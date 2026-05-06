import argparse
import os

import numpy as np
from typing import Dict

from src.architecture import compile_hybrid_network
from src.config import (
    FORMS_METADATA,
    IMAGE_DIRECTORY,
    MODEL_DIRECTORY,
    RANDOM_SEED,
    SPLIT_INDEX_FILE,
    SPLIT_TEST_RATIO,
    SPLIT_TRAIN_RATIO,
    SPLIT_VAL_RATIO,
    STRICT_CHARSET_POLICY,
    STRICT_LM_DECODER,
    STRICT_SPLIT_POLICY,
    VOCABULARY_LIST,
)
from src.dataset_parser import OptimizedHTRGenerator, parse_iam_metadata
from src.inference_engine import build_replication_lm_decoder, execute_ctc_decoding
from src.logger import initialize_logger
from src.metrics import aggregate_corpus_metrics
from src.split_utils import load_or_create_split_indices

log = initialize_logger(__name__)


def _decode_truth_batch(labels: np.ndarray, label_lengths: np.ndarray) -> list:
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


def evaluate_words_test_dataset(weights_path: str, require_lm: bool) -> Dict[str, float]:
    verified_corpus = parse_iam_metadata(
        FORMS_METADATA,
        IMAGE_DIRECTORY,
        strict_charset=STRICT_CHARSET_POLICY,
        strict_integrity=True,
    )

    _, _, test_idx = load_or_create_split_indices(
        total_size=len(verified_corpus),
        split_path=SPLIT_INDEX_FILE,
        train_ratio=SPLIT_TRAIN_RATIO,
        val_ratio=SPLIT_VAL_RATIO,
        test_ratio=SPLIT_TEST_RATIO,
        seed=RANDOM_SEED,
        strict=STRICT_SPLIT_POLICY,
    )

    test_subset = [verified_corpus[i] for i in test_idx]
    if not test_subset:
        raise RuntimeError("Test subset is empty. Cannot evaluate metrics.")

    _, model_inference = compile_hybrid_network()
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing model weights at {weights_path}")

    model_inference.load_weights(weights_path)

    lm_decoder = build_replication_lm_decoder(required=require_lm)

    generator_test = OptimizedHTRGenerator(
        test_subset,
        shuffle_data=False,
        strict_mode=True,
    )

    predictions = []
    truths = []

    for batch_index in range(len(generator_test)):
        batch_x, _ = generator_test[batch_index]

        if batch_x["image_input"].shape[0] == 0:
            continue

        model_output = model_inference(batch_x["image_input"], training=False)
        if isinstance(model_output, (list, tuple)):
            model_output = model_output[0]

        probs = np.asarray(model_output, dtype=np.float32)

        batch_predictions = execute_ctc_decoding(
            probs,
            lm_decoder=lm_decoder,
            require_lm=require_lm,
        )
        batch_truths = _decode_truth_batch(batch_x["labels"], batch_x["label_length"])

        if len(batch_predictions) != len(batch_truths):
            raise RuntimeError(
                "Batch prediction/ground-truth mismatch: "
                f"{len(batch_predictions)} vs {len(batch_truths)}"
            )

        predictions.extend(batch_predictions)
        truths.extend(batch_truths)

    if not predictions:
        raise RuntimeError("No predictions produced for test subset.")

    metric_bundle = aggregate_corpus_metrics(predictions, truths)

    print(f"Test samples evaluated: {len(truths)}")
    print(f"Accuracy (word-level): {metric_bundle['word_accuracy']:.2f}%")
    print(f"Sequence accuracy (exact match): {metric_bundle['sequence_accuracy']:.2f}%")
    print(f"WER: {metric_bundle['wer']:.4f}")
    print(f"CER: {metric_bundle['cer']:.4f}")

    return metric_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate src model on the pre-split IAM words test dataset."
    )
    parser.add_argument(
        "--weights",
        default=os.path.join(MODEL_DIRECTORY, "optimal_hybrid_weights.weights.h5"),
        help="Path to model weights file (.weights.h5).",
    )
    parser.add_argument(
        "--no-lm",
        action="store_true",
        help="Disable strict LM requirement and decode without LM if needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_lm = False if args.no_lm else STRICT_LM_DECODER

    log.info("Evaluating words test split using weights: %s", args.weights)
    log.info("LM decoding required: %s", str(require_lm))

    metrics = evaluate_words_test_dataset(args.weights, require_lm=require_lm)
    log.info(
        "Test metrics | WordAcc: %.2f%% | SeqAcc: %.2f%% | WER: %.4f | CER: %.4f",
        metrics["word_accuracy"],
        metrics["sequence_accuracy"],
        metrics["wer"],
        metrics["cer"],
    )


if __name__ == "__main__":
    main()