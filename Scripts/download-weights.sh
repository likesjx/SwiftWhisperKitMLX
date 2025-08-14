#!/usr/bin/env bash
set -euo pipefail

# Simple helper to download a Whisper model weight bundle into Application Support.
# Usage: ./Scripts/download-weights.sh <variant> <base-url>
# Example: ./Scripts/download-weights.sh small https://example.com/whisper

VARIANT="${1:-}"
BASE_URL="${2:-}"
if [[ -z "$VARIANT" || -z "$BASE_URL" ]]; then
  echo "Usage: $0 <variant> <base-url>" >&2
  exit 1
fi

DEST="$HOME/Library/Application Support/$VARIANT"
mkdir -p "$DEST"

echo "Downloading manifest..."
curl -fsSL "$BASE_URL/$VARIANT/manifest.json" -o "$DEST/manifest.json"

if [[ -f "$DEST/manifest.json" ]]; then
  # Extract tensor file list (very naive jq-less parse expecting '"file":"..."')
  grep -o '"file" *: *"[^"]*"' "$DEST/manifest.json" | sed -E 's/.*"file" *: *"([^"]*)"/\1/' | while read -r f; do
    if [[ ! -f "$DEST/$f" ]]; then
      echo "Downloading tensor $f"
      curl -fsSL "$BASE_URL/$VARIANT/$f" -o "$DEST/$f"
    fi
  done
fi

for extra in vocab.json tokenizer.vocab.json tokenizer.merges.txt; do
  if [[ ! -f "$DEST/$extra" ]]; then
    echo "Attempting to download $extra"
    curl -fsSL "$BASE_URL/$VARIANT/$extra" -o "$DEST/$extra" || true
  fi
done

echo "Done. Files in $DEST:"; ls -1 "$DEST"
