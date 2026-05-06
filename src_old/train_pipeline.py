import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.config import (
    FORMS_METADATA,
    IMAGE_DIRECTORY,
    TOTAL_EPOCHS,
    INITIAL_LEARNING_RATE,
    MODEL_DIRECTORY,
    VOCABULARY_LIST,
    RANDOM_SEED,
    SPLIT_INDEX_FILE,
    MAX_GRAD_NORM,
    SPLIT_TRAIN_RATIO,
    SPLIT_VAL_RATIO,
    SPLIT_TEST_RATIO,
    STRICT_SPLIT_POLICY,
    STRICT_CHARSET_POLICY,
    LM_CORPUS_TEXT_PATH,
    UNIGRAMS_PATH,
    KENLM_MODEL_PATH,
)
from src.dataset_parser import parse_iam_metadata, OptimizedHTRGenerator
from src.architecture import compile_hybrid_network
from src.inference_engine import execute_ctc_decoding
from src.metrics import aggregate_corpus_metrics
from src.split_utils import load_or_create_split_indices
from src.logger import initialize_logger

log = initialize_logger(__name__)


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def export_lm_assets(training_subset: list):
    os.makedirs(os.path.dirname(LM_CORPUS_TEXT_PATH), exist_ok=True)

    with open(LM_CORPUS_TEXT_PATH, "w", encoding="utf-8") as corpus_file:
        for sample in training_subset:
            text = sample["transcription"].strip()
            if text:
                corpus_file.write(text + "\n")

    unigram_set = set()
    for sample in training_subset:
        for token in sample["transcription"].split():
            if token:
                unigram_set.add(token)

    with open(UNIGRAMS_PATH, "w", encoding="utf-8") as unigram_file:
        for token in sorted(unigram_set):
            unigram_file.write(token + "\n")

    log.info("LM corpus exported to %s", LM_CORPUS_TEXT_PATH)
    log.info("LM unigrams exported to %s (%d entries)", UNIGRAMS_PATH, len(unigram_set))
    log.info(
        "Build KenLM binary from corpus and place it at %s before strict evaluation.",
        KENLM_MODEL_PATH,
    )


def decode_truth_batch(labels: np.ndarray, label_lengths: np.ndarray) -> list:
    vocab_size = len(VOCABULARY_LIST)
    lengths = label_lengths.reshape(-1).astype(np.int32)
    truths = []

    for row, ln in zip(labels, lengths):
        chars = []
        for token in row[:int(ln)]:
            token = int(token)
            if 0 <= token < vocab_size:
                chars.append(VOCABULARY_LIST[token])
        truths.append("".join(chars))

    return truths


class DecodeMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator, inference_model, save_path, max_batches=200):
        super().__init__()
        self.val_generator = val_generator
        self.inference_model = inference_model
        self.save_path = save_path
        self.max_batches = max_batches
        self.best_wer = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        preds, truths = [], []
        steps = min(len(self.val_generator), self.max_batches)

        for i in range(steps):
            batch_x, _ = self.val_generator[i]
            probs = self.inference_model.predict(batch_x["image_input"], verbose=0)

            preds.extend(
                execute_ctc_decoding(
                    probs,
                    lm_decoder=None,
                    require_lm=False,
                )
            )
            truths.extend(decode_truth_batch(batch_x["labels"], batch_x["label_length"]))

        metric_bundle = aggregate_corpus_metrics(preds, truths)

        logs["val_decode_word_accuracy"] = metric_bundle["word_accuracy"]
        logs["val_decode_seq_accuracy"] = metric_bundle["sequence_accuracy"]
        logs["val_decode_wer"] = metric_bundle["wer"]
        logs["val_decode_cer"] = metric_bundle["cer"]

        log.info(
            "Epoch %d | val_decode_word_accuracy=%.2f%% | val_decode_wer=%.4f | val_decode_cer=%.4f",
            epoch + 1,
            metric_bundle["word_accuracy"],
            metric_bundle["wer"],
            metric_bundle["cer"],
        )

        if metric_bundle["wer"] < self.best_wer:
            self.best_wer = metric_bundle["wer"]
            self.model.save_weights(self.save_path)
            log.info("Saved decode-best checkpoint to %s", self.save_path)


def execute_training_lifecycle():
    try:
        set_global_seed(RANDOM_SEED)
        log.info("Bootstrapping replication-mode training lifecycle...")

        verified_corpus = parse_iam_metadata(
            FORMS_METADATA,
            IMAGE_DIRECTORY,
            strict_charset=STRICT_CHARSET_POLICY,
            strict_integrity=True,
        )

        train_idx, val_idx, test_idx = load_or_create_split_indices(
            total_size=len(verified_corpus),
            split_path=SPLIT_INDEX_FILE,
            train_ratio=SPLIT_TRAIN_RATIO,
            val_ratio=SPLIT_VAL_RATIO,
            test_ratio=SPLIT_TEST_RATIO,
            seed=RANDOM_SEED,
            strict=STRICT_SPLIT_POLICY,
        )

        subset_training = [verified_corpus[i] for i in train_idx]
        subset_validation = [verified_corpus[i] for i in val_idx]
        subset_test = [verified_corpus[i] for i in test_idx]

        log.info(
            "Partitioned Volumes -> Train: %d, Valid: %d, Test: %d",
            len(subset_training),
            len(subset_validation),
            len(subset_test),
        )

        export_lm_assets(subset_training)

        generator_train = OptimizedHTRGenerator(
            subset_training,
            shuffle_data=True,
            strict_mode=True,
        )
        generator_valid = OptimizedHTRGenerator(
            subset_validation,
            shuffle_data=False,
            strict_mode=True,
        )

        model_training, model_inference = compile_hybrid_network()
        optimizer = Adam(learning_rate=INITIAL_LEARNING_RATE, clipnorm=MAX_GRAD_NORM)
        model_training.compile(optimizer=optimizer)

        os.makedirs(MODEL_DIRECTORY, exist_ok=True)
        by_loss_path = os.path.join(MODEL_DIRECTORY, "optimal_hybrid_by_loss.weights.h5")
        by_decode_path = os.path.join(MODEL_DIRECTORY, "optimal_hybrid_weights.weights.h5")

        callbacks = [
            ModelCheckpoint(
                filepath=by_loss_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=12,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1,
            ),
            DecodeMetricCallback(
                val_generator=generator_valid,
                inference_model=model_inference,
                save_path=by_decode_path,
                max_batches=200,
            ),
        ]

        log.info("Commencing training...")
        history = model_training.fit(
            generator_train,
            validation_data=generator_valid,
            epochs=TOTAL_EPOCHS,
            callbacks=callbacks,
        )

        log.info("Training completed.")
        return history

    except Exception as exception_trace:
        log.critical(f"Abrupt termination of training protocol: {str(exception_trace)}")
        raise


if __name__ == "__main__":
    execute_training_lifecycle()