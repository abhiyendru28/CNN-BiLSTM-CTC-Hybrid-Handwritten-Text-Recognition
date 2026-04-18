from src.logger import initialize_logger

log = initialize_logger(__name__)


def levenshtein_distance(seq1, seq2) -> int:
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    previous_row = list(range(len(seq2) + 1))

    for i, c1 in enumerate(seq1, start=1):
        current_row = [i]
        for j, c2 in enumerate(seq2, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (0 if c1 == c2 else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def evaluate_cer(predicted_string: str, ground_truth_string: str) -> float:
    dist = levenshtein_distance(predicted_string, ground_truth_string)
    return dist / max(len(ground_truth_string), 1)


def evaluate_wer(predicted_string: str, ground_truth_string: str) -> float:
    predicted_tokens = predicted_string.split()
    truth_tokens = ground_truth_string.split()
    dist = levenshtein_distance(predicted_tokens, truth_tokens)
    return dist / max(len(truth_tokens), 1)


def aggregate_corpus_metrics(predictions_list: list, truth_list: list) -> dict:
    if len(predictions_list) != len(truth_list):
        raise ValueError(
            f"Prediction/truth length mismatch: {len(predictions_list)} vs {len(truth_list)}"
        )

    exact_sequence_matches = 0
    cumulative_sample_wer = 0.0
    cumulative_sample_cer = 0.0

    total_word_edits = 0
    total_word_ref = 0
    total_char_edits = 0
    total_char_ref = 0

    n = max(len(predictions_list), 1)

    for pred, truth in zip(predictions_list, truth_list):
        pred = str(pred)
        truth = str(truth)

        if pred == truth:
            exact_sequence_matches += 1

        pred_words = pred.split()
        truth_words = truth.split()

        word_edits = levenshtein_distance(pred_words, truth_words)
        char_edits = levenshtein_distance(pred, truth)

        total_word_edits += word_edits
        total_word_ref += max(len(truth_words), 1)

        total_char_edits += char_edits
        total_char_ref += max(len(truth), 1)

        cumulative_sample_wer += word_edits / max(len(truth_words), 1)
        cumulative_sample_cer += char_edits / max(len(truth), 1)

    average_wer = cumulative_sample_wer / n
    average_cer = cumulative_sample_cer / n
    global_wer = total_word_edits / max(total_word_ref, 1)
    global_cer = total_char_edits / max(total_char_ref, 1)

    sequence_accuracy = (exact_sequence_matches / n) * 100.0
    word_accuracy = max(0.0, (1.0 - global_wer) * 100.0)
    character_accuracy = max(0.0, (1.0 - global_cer) * 100.0)

    metrics = {
        "sequence_accuracy": sequence_accuracy,
        "word_accuracy": word_accuracy,
        "character_accuracy": character_accuracy,
        "wer": average_wer,
        "cer": average_cer,
        "global_wer": global_wer,
        "global_cer": global_cer,
    }

    log.info(
        "SeqAcc: %.2f%% | WordAcc: %.2f%% | CharAcc: %.2f%% | "
        "WER(avg): %.4f | CER(avg): %.4f | WER(global): %.4f | CER(global): %.4f",
        metrics["sequence_accuracy"],
        metrics["word_accuracy"],
        metrics["character_accuracy"],
        metrics["wer"],
        metrics["cer"],
        metrics["global_wer"],
        metrics["global_cer"],
    )

    return metrics