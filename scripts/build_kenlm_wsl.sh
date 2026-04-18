#!/usr/bin/env bash
# Build KenLM and create binary LM from the project's corpus (WSL/Ubuntu)
# Usage: run inside WSL from the repo root or provide REPO_DPATH env var
set -euo pipefail

# If you want to override, set REPO_DPATH to the Windows path (e.g. /mnt/d/HTR/CNN-BiLSTM-CTC-HTR)
REPO_DPATH=${REPO_DPATH:-$(pwd)}
MODEL_DIR="$REPO_DPATH/saved_models"
CORPUS_PATH="$MODEL_DIR/iam_corpus.txt"
TARGET_DIR="$REPO_DPATH/language_model"
ARPA_NAME="iam.arpa"
BIN_NAME="iam_words.arpa.bin"
KENLM_DIR="$REPO_DPATH/kenlm"

echo "Repository path: $REPO_DPATH"
echo "Corpus path: $CORPUS_PATH"

if [ ! -f "$CORPUS_PATH" ]; then
  echo "ERROR: Corpus not found at $CORPUS_PATH"
  echo "Run your training once or ensure the file exists at saved_models/iam_corpus.txt"
  exit 2
fi

# Install build deps
echo "Installing build dependencies (apt)..."
sudo apt update
# KenLM requires Boost (program_options, system, thread, unit_test_framework)
sudo apt install -y build-essential cmake libbz2-dev liblzma-dev zlib1g-dev git \
  libboost-all-dev

# Clone KenLM if missing
if [ ! -d "$KENLM_DIR" ]; then
  echo "Cloning KenLM into $KENLM_DIR"
  git clone https://github.com/kpu/kenlm.git "$KENLM_DIR"
fi

# Build KenLM
echo "Building KenLM..."
mkdir -p "$KENLM_DIR/build"
pushd "$KENLM_DIR/build" >/dev/null
cmake ..
make -j$(nproc)
popd >/dev/null

# Create ARPA using lmplz
echo "Creating ARPA model (order 5)"
LMPLZ="$KENLM_DIR/build/bin/lmplz"
BUILD_BINARY="$KENLM_DIR/build/bin/build_binary"

if [ ! -x "$LMPLZ" ] || [ ! -x "$BUILD_BINARY" ]; then
  echo "ERROR: KenLM build tools not found in $KENLM_DIR/build/bin"
  exit 3
fi

pushd "$REPO_DPATH" >/dev/null
mkdir -p "$TARGET_DIR"

# Generate arpa then binary
echo "Preparing corpus (deduplicate -> temporary file)"
TMP_CORPUS="$MODEL_DIR/iam_corpus.dedup.txt"
sort -u "$CORPUS_PATH" > "$TMP_CORPUS"

echo "Attempting to generate ARPA model from deduplicated corpus"
SUCCESS=0
# Try orders 5,4,3 and enable discount_fallback to avoid Kneser-Ney errors on small/degenerate data
for ORDER in 5 4 3; do
  echo "Trying n-gram order=${ORDER} with discount_fallback"
  if "$LMPLZ" -o "$ORDER" --discount_fallback < "$TMP_CORPUS" > "$TARGET_DIR/$ARPA_NAME"; then
    echo "ARPA generation succeeded with order=${ORDER}"
    SUCCESS=1
    break
  else
    echo "ARPA generation failed for order=${ORDER}, trying lower order..."
  fi
done

if [ "$SUCCESS" -ne 1 ]; then
  echo "ERROR: Could not generate ARPA model with fallback orders. Inspect corpus or run with a larger dataset."
  exit 4
fi

echo "Building binary -> $TARGET_DIR/$BIN_NAME"
"$BUILD_BINARY" "$TARGET_DIR/$ARPA_NAME" "$TARGET_DIR/$BIN_NAME"

popd >/dev/null

echo "KenLM binary created at: $TARGET_DIR/$BIN_NAME"

echo "Done. Ensure your config KENLM_MODEL_PATH points to: $TARGET_DIR/$BIN_NAME"
