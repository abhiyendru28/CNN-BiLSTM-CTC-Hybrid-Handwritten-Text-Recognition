import os

BASE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, "data")
FORMS_METADATA = os.path.join(DATA_DIRECTORY, "words.txt")
IMAGE_DIRECTORY = os.path.join(DATA_DIRECTORY, "words")
LOG_DIRECTORY = os.path.join(BASE_DIRECTORY, "logs")
MODEL_DIRECTORY = os.path.join(BASE_DIRECTORY, "saved_models")

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32
COLOR_CHANNELS = 1

# Training
TRAIN_BATCH_SIZE = 32
TOTAL_EPOCHS = 80
INITIAL_LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 5.0
SEQUENCE_TIME_STEPS = 32

# Multiprocessing controls for data loading (training + inference)
CPU_CORE_COUNT = max(1, os.cpu_count() or 1)
ENABLE_MULTIPROCESSING = CPU_CORE_COUNT > 1
TRAINING_WORKERS = max(1, min(8, CPU_CORE_COUNT - 1)) if ENABLE_MULTIPROCESSING else 1
INFERENCE_WORKERS = TRAINING_WORKERS
MULTIPROCESSING_QUEUE_SIZE = 16
USE_PROCESS_BASED_WORKERS = ENABLE_MULTIPROCESSING and (os.name != "nt")

ENABLE_MIXED_PRECISION = True
ENABLE_XLA_JIT = False
TRAIN_STEPS_PER_EXECUTION = 16

DECODE_EVAL_MAX_BATCHES = 200
DECODE_EVAL_EVERY_N_EPOCHS = 2

# split (train/val/test).
SPLIT_TRAIN_RATIO = 0.72
SPLIT_VAL_RATIO = 0.08
SPLIT_TEST_RATIO = 0.20
if abs((SPLIT_TRAIN_RATIO + SPLIT_VAL_RATIO + SPLIT_TEST_RATIO) - 1.0) > 1e-9:
    raise ValueError("Split ratios must sum to 1.0")

TRAINING_SPLIT_RATIO = SPLIT_TRAIN_RATIO + SPLIT_VAL_RATIO

# Charset
VOCABULARY_LIST = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-()'\"#&*+/:; "
TOTAL_CLASSES = len(VOCABULARY_LIST) + 1

RANDOM_SEED = 42
SPLIT_INDEX_FILE = os.path.join(MODEL_DIRECTORY, "replication_split_indices.npz")

REPLICATION_MODE = True
STRICT_SPLIT_POLICY = True
STRICT_CHARSET_POLICY = True
STRICT_LM_DECODER = True

# LM assets 
LM_CORPUS_TEXT_PATH = os.path.join(MODEL_DIRECTORY, "iam_corpus.txt")
UNIGRAMS_PATH = os.path.join(MODEL_DIRECTORY, "iam_words_unigrams.txt")
KENLM_MODEL_PATH = os.path.join(BASE_DIRECTORY, "language_model", "iam_words.arpa.bin")
LM_ALPHA = 0.6
LM_BETA = 1.5

CTC_GREEDY = False
CTC_BEAM_WIDTH = 50
CTC_TOP_PATHS = 1