#!/usr/bin/env bash
set -euo pipefail

NAOMI_PATH="/mnt/c/Users/Bo Li/Desktop/adamshch-naomi_sim-20250411"
NAOMI_PATH_WIN="C:/Users/Bo Li/Desktop/adamshch-naomi_sim-20250411"

usage() {
    echo "Usage: $0 <config.json> <output_directory>" >&2
    exit 1
}

if [[ "$#" -ne 2 ]]; then
    usage
fi

CONFIG_JSON="$1"
OUTPUT_DIR="$2"

if [[ ! -f "$CONFIG_JSON" ]]; then
    echo "Error: config file not found: $CONFIG_JSON" >&2
    exit 1
fi

if ! python3 -c "import json" 2>/dev/null; then
    echo "Error: python3 is required but not available" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python3 generate_naomi_m.py "$CONFIG_JSON" naomi_generate.m

cp ./naomi_generate.m "${NAOMI_PATH}/code/"

ssh win "matlab -batch \"cd ('${NAOMI_PATH_WIN}/code'); naomi_generate;\""

TIFF_FILE=$(find "${NAOMI_PATH}/code/" -maxdepth 1 -type f -name '*_00001.tif' -print -quit)

if [[ -z "$TIFF_FILE" ]]; then
    echo "Error: no *_00001.tif file found in ${NAOMI_PATH}/code/" >&2
    exit 1
fi

if [[ $(find "${NAOMI_PATH}/code/" -maxdepth 1 -type f -name '*_00001.tif' | wc -l) -ne 1 ]]; then
    echo "Error: expected exactly one *_00001.tif file in ${NAOMI_PATH}/code/" >&2
    exit 1
fi

mv "$TIFF_FILE" "$OUTPUT_DIR/"

TIFF_BASENAME=$(basename "$TIFF_FILE")
BASE_NAME="${TIFF_BASENAME%_00001.tif}"

cp "$CONFIG_JSON" "${OUTPUT_DIR}/${BASE_NAME}.json"

echo "Generated:"
echo "  TIFF: ${OUTPUT_DIR}/${TIFF_BASENAME}"
echo "  JSON: ${OUTPUT_DIR}/${BASE_NAME}.json"
