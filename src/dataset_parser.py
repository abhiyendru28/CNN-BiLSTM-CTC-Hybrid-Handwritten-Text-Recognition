import os
import numpy as np
import tensorflow as tf
from src.config import (
    VOCABULARY_LIST,
    SEQUENCE_TIME_STEPS,
    TRAIN_BATCH_SIZE,
    STRICT_CHARSET_POLICY,
)
from src.preprocessing import execute_morphological_preprocessing
from src.logger import initialize_logger

log = initialize_logger(__name__)

char_to_integer_map = {character: index for index, character in enumerate(VOCABULARY_LIST)}


def _apply_charset_policy(raw_text: str, strict_charset: bool) -> str:
    unknown_chars = sorted({ch for ch in raw_text if ch not in char_to_integer_map})

    if unknown_chars and strict_charset:
        raise ValueError(
            "Unknown characters found in transcription. "
            f"Chars: {''.join(unknown_chars)} | Text: {raw_text}"
        )

    if strict_charset:
        return raw_text

    return "".join(ch for ch in raw_text if ch in char_to_integer_map)


def encode_ground_truth(text_string: str, strict_charset: bool = STRICT_CHARSET_POLICY) -> list:
    encoded = []
    for char in text_string:
        if char in char_to_integer_map:
            encoded.append(char_to_integer_map[char])
        elif strict_charset:
            raise ValueError(f"Out-of-vocabulary character during encoding: {repr(char)}")
    return encoded


def parse_iam_metadata(
    metadata_filepath: str,
    images_directory: str,
    strict_charset: bool = STRICT_CHARSET_POLICY,
    strict_integrity: bool = True,
) -> list:
    verified_dataset = []
    try:
        with open(metadata_filepath, "r", encoding="utf-8") as file_buffer:
            for textual_line in file_buffer:
                if textual_line.startswith("#") or not textual_line.strip():
                    continue

                tokens = textual_line.strip().split()
                if len(tokens) < 8:
                    continue

                word_id = tokens[0]
                segmentation_status = tokens[1]
                if segmentation_status != "ok":
                    continue

                tag_index = 2
                while tag_index < len(tokens) and tokens[tag_index].lstrip("-").isdigit():
                    tag_index += 1

                if tag_index + 1 >= len(tokens):
                    continue

                raw_transcription = " ".join(tokens[tag_index + 1:])

                try:
                    clean_transcription = _apply_charset_policy(raw_transcription, strict_charset)
                except ValueError as charset_error:
                    if strict_charset:
                        raise ValueError(f"{word_id}: {charset_error}") from charset_error
                    continue

                if len(clean_transcription) < 1 or len(clean_transcription) > SEQUENCE_TIME_STEPS:
                    continue

                directory_parts = word_id.split("-")
                if len(directory_parts) < 2:
                    continue

                folder_1 = directory_parts[0]
                folder_2 = f"{directory_parts[0]}-{directory_parts[1]}"
                absolute_img_path = os.path.join(images_directory, folder_1, folder_2, word_id + ".png")

                if not os.path.isfile(absolute_img_path):
                    if strict_integrity:
                        raise FileNotFoundError(f"Missing image: {absolute_img_path}")
                    continue

                verified_dataset.append(
                    {
                        "path": absolute_img_path,
                        "transcription": clean_transcription,
                    }
                )

        log.info(f"Word metadata ingestion complete. Total verified samples: {len(verified_dataset)}")
        return verified_dataset

    except Exception as exception_trace:
        log.critical(f"Metadata parsing collapsed: {str(exception_trace)}")
        raise


class OptimizedHTRGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_corpus, batch_size=TRAIN_BATCH_SIZE, shuffle_data=True, strict_mode=False):
        self.data_corpus = data_corpus
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.strict_mode = strict_mode
        self.dataset_indices = np.arange(len(self.data_corpus))
        if self.shuffle_data:
            np.random.shuffle(self.dataset_indices)

    def __len__(self):
        return int(np.ceil(len(self.data_corpus) / self.batch_size))

    def __getitem__(self, batch_index):
        current_indices = self.dataset_indices[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        current_batch_metadata = [self.data_corpus[i] for i in current_indices]

        tensor_images, tensor_labels = [], []
        seq_label_lengths, seq_input_lengths = [], []

        for element in current_batch_metadata:
            processed_img = execute_morphological_preprocessing(element["path"])

            try:
                encoded_array = encode_ground_truth(
                    element["transcription"],
                    strict_charset=self.strict_mode or STRICT_CHARSET_POLICY,
                )
            except ValueError as encoding_error:
                log.warning("Encoding error for %s: %s. Skipping sample.", element["path"], str(encoding_error))
                continue

            if processed_img is None or len(encoded_array) == 0:
                log.warning("Invalid sample during preprocessing, skipping: %s", element["path"])
                continue

            if len(encoded_array) > SEQUENCE_TIME_STEPS:
                log.warning(
                    "Label length exceeds sequence steps for %s: %d > %d. Skipping.",
                    element["path"],
                    len(encoded_array),
                    SEQUENCE_TIME_STEPS,
                )
                continue

            blank_token_index = len(VOCABULARY_LIST)
            padded_array = encoded_array + [blank_token_index] * (SEQUENCE_TIME_STEPS - len(encoded_array))

            tensor_images.append(processed_img)
            tensor_labels.append(padded_array)
            seq_label_lengths.append(len(encoded_array))
            seq_input_lengths.append(SEQUENCE_TIME_STEPS)

        if self.strict_mode and len(tensor_images) == 0:
            raise ValueError(f"Empty batch produced at batch_index={batch_index}")

        X_dict = {
            "image_input": np.array(tensor_images, dtype=np.float32),
            "labels": np.array(tensor_labels, dtype=np.int32),
            "input_length": np.array(seq_input_lengths, dtype=np.int32).reshape(-1, 1),
            "label_length": np.array(seq_label_lengths, dtype=np.int32).reshape(-1, 1),
        }

        Y_dummy = np.zeros((len(tensor_images),), dtype=np.float32)
        return X_dict, Y_dummy

    def on_epoch_end(self):
        if self.shuffle_data:
            np.random.shuffle(self.dataset_indices)