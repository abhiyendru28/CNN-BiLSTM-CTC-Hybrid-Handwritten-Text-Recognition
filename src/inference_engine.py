import os
import numpy as np
import tensorflow as tf

from src.config import (
    FORMS_METADATA,
    IMAGE_DIRECTORY,
    VOCABULARY_LIST,
    MODEL_DIRECTORY,
    SPLIT_INDEX_FILE,
    RANDOM_SEED,
    SPLIT_TRAIN_RATIO,
    SPLIT_VAL_RATIO,
    SPLIT_TEST_RATIO,
    STRICT_SPLIT_POLICY,
    STRICT_CHARSET_POLICY,
    STRICT_LM_DECODER,
    KENLM_MODEL_PATH,
    UNIGRAMS_PATH,
    LM_ALPHA,
    LM_BETA,
    CTC_GREEDY,
    CTC_BEAM_WIDTH,
    CTC_TOP_PATHS,
)
from src.dataset_parser import parse_iam_metadata, OptimizedHTRGenerator
from src.architecture import compile_hybrid_network
from src.metrics import aggregate_corpus_metrics
from src.split_utils import load_or_create_split_indices
from src.logger import initialize_logger

try:
    from pyctcdecode.decoder import build_ctcdecoder
except Exception:
    build_ctcdecoder = None

log = initialize_logger(__name__)


def build_replication_lm_decoder(required: bool = STRICT_LM_DECODER):


    if not os.path.exists(KENLM_MODEL_PATH):
        if required:
            raise FileNotFoundError(
                f"Missing KenLM binary at {KENLM_MODEL_PATH}. "
                "Replication mode requires corpus-backed decoding."
            )
        return None

    if not os.path.exists(UNIGRAMS_PATH):
        if required:
            raise FileNotFoundError(
                f"Missing unigram list at {UNIGRAMS_PATH}. "
                "Replication mode requires corpus-backed decoding."
            )
        unigrams = None
    else:
        with open(UNIGRAMS_PATH, "r", encoding="utf-8") as file_buffer:
            unigrams = [line.strip() for line in file_buffer if line.strip()]

    labels = list(VOCABULARY_LIST) + [""]

    return build_ctcdecoder(
        labels=labels,
        kenlm_model_path=KENLM_MODEL_PATH,
        unigrams=unigrams,
        alpha=LM_ALPHA,
        beta=LM_BETA,
    )


def execute_ctc_decoding(
    softmax_probability_matrix: np.ndarray,
    lm_decoder=None,
    require_lm: bool = STRICT_LM_DECODER,
) -> list:
    if require_lm and lm_decoder is None:
        raise RuntimeError(
            "Language-model decoding is required in replication mode, "
            "but no LM decoder was supplied."
        )

    if lm_decoder is not None:
        translated = []
        for probs in softmax_probability_matrix:
            log_probs = np.log(np.clip(probs, 1e-8, 1.0))
            translated.append(lm_decoder.decode(log_probs))
        return translated

    batch_size = softmax_probability_matrix.shape[0]
    time_steps = softmax_probability_matrix.shape[1]
    input_lengths = np.full((batch_size,), time_steps, dtype=np.int32)

    # Use config-driven CTC decode parameters so beam/greedy can be tuned.
    decoded_sequences, _ = tf.keras.backend.ctc_decode(
        softmax_probability_matrix,
        input_length=input_lengths,
        greedy=CTC_GREEDY,
        beam_width=CTC_BEAM_WIDTH,
        top_paths=CTC_TOP_PATHS,
    )

    decoded_dense = decoded_sequences[0].numpy()
    translated_texts = []
    vocab_size = len(VOCABULARY_LIST)

    for row in decoded_dense:
        chars = []
        for token in row:
            token = int(token)
            if 0 <= token < vocab_size:
                chars.append(VOCABULARY_LIST[token])
        translated_texts.append("".join(chars))

    return translated_texts


def validate_production_system():
    try:
        log.info("Initializing replication-mode inference engine...")

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
        _ = train_idx, val_idx

        subset_testing = [verified_corpus[i] for i in test_idx]
        generator_test = OptimizedHTRGenerator(
            subset_testing,
            shuffle_data=False,
            strict_mode=True,
        )

        _, model_inference = compile_hybrid_network()
        weight_filepath = os.path.join(MODEL_DIRECTORY, "optimal_hybrid_weights.weights.h5")

        if not os.path.exists(weight_filepath):
            raise FileNotFoundError(f"Missing model weights at {weight_filepath}. Train first.")

        model_inference.load_weights(weight_filepath)
        lm_decoder = build_replication_lm_decoder(required=STRICT_LM_DECODER)

        list_ground_truths = [element["transcription"] for element in subset_testing]
        list_predictions = []

        for tensor_batch_x, _ in generator_test:
            softmax_outputs = model_inference.predict(tensor_batch_x["image_input"], verbose=0)
            decoded_batch_strings = execute_ctc_decoding(
                softmax_outputs,
                lm_decoder=lm_decoder,
                require_lm=STRICT_LM_DECODER,
            )
            list_predictions.extend(decoded_batch_strings)

        if len(list_predictions) != len(list_ground_truths):
            raise RuntimeError(
                f"Prediction/ground-truth mismatch: {len(list_predictions)} vs {len(list_ground_truths)}"
            )

        metric_bundle = aggregate_corpus_metrics(list_predictions, list_ground_truths)

        log.info(
            "Final Metrics | WordAcc: %.2f%% | SeqAcc: %.2f%% | WER: %.4f | CER: %.4f",
            metric_bundle["word_accuracy"],
            metric_bundle["sequence_accuracy"],
            metric_bundle["wer"],
            metric_bundle["cer"],
        )

        if metric_bundle["word_accuracy"] >= 98.50 and metric_bundle["wer"] <= 0.015:
            log.info("SUCCESS: Replication target reached for IAM benchmark threshold.")
        else:
            log.warning("Replication target not reached yet. Continue tuning and data checks.")

        return metric_bundle

    except Exception as exception_trace:
        log.critical(f"Verification engine failure: {str(exception_trace)}")
        raise


if __name__ == "__main__":
    validate_production_system()