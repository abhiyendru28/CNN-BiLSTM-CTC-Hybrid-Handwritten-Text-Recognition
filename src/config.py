import os

# Absolute Pathing
BASE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, "data")
FORMS_METADATA = os.path.join(DATA_DIRECTORY, "words.txt")
LINES_METADATA = os.path.join(DATA_DIRECTORY, "lines.txt")
IMAGE_DIRECTORY = os.path.join(DATA_DIRECTORY, "words")
LOG_DIRECTORY = os.path.join(BASE_DIRECTORY, "logs")
MODEL_DIRECTORY = os.path.join(BASE_DIRECTORY, "saved_models")

# Input geometry
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32
COLOR_CHANNELS = 1

# Training
TRAIN_BATCH_SIZE = 32
TOTAL_EPOCHS = 80
INITIAL_LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 5.0
SEQUENCE_TIME_STEPS = 32

# Replication split (train/val/test).
# 72/8/20 keeps paper-style 80/20 train-test while reserving validation.
SPLIT_TRAIN_RATIO = 0.72
SPLIT_VAL_RATIO = 0.08
SPLIT_TEST_RATIO = 0.20
if abs((SPLIT_TRAIN_RATIO + SPLIT_VAL_RATIO + SPLIT_TEST_RATIO) - 1.0) > 1e-9:
    raise ValueError("Split ratios must sum to 1.0")

# Backward-compatible alias used in a few paths.
TRAINING_SPLIT_RATIO = SPLIT_TRAIN_RATIO + SPLIT_VAL_RATIO

# Charset
VOCABULARY_LIST = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-()'\"#&*+/:; "
TOTAL_CLASSES = len(VOCABULARY_LIST) + 1

# Reproducibility and frozen split manifest
RANDOM_SEED = 42
SPLIT_INDEX_FILE = os.path.join(MODEL_DIRECTORY, "replication_split_indices.npz")

# Replication policy flags
REPLICATION_MODE = True
STRICT_SPLIT_POLICY = True
STRICT_CHARSET_POLICY = True
STRICT_LM_DECODER = True

# LM assets for corpus-backed decoding
LM_CORPUS_TEXT_PATH = os.path.join(MODEL_DIRECTORY, "iam_corpus.txt")
UNIGRAMS_PATH = os.path.join(MODEL_DIRECTORY, "iam_words_unigrams.txt")
KENLM_MODEL_PATH = os.path.join(BASE_DIRECTORY, "language_model", "iam_words.arpa.bin")
LM_ALPHA = 0.6
LM_BETA = 1.5

# CTC decoding parameters (used when no external LM is available)
# `CTC_GREEDY=True` will use a greedy decode; otherwise beam search is used.
CTC_GREEDY = False
# Beam width for tf.keras.backend.ctc_decode when greedy=False
CTC_BEAM_WIDTH = 50
# Number of top paths to return from CTC decode
CTC_TOP_PATHS = 1