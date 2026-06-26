#!/usr/bin/env bash
# =============================================================================
# Download nuScenes dataset and pretrained backbone weights for PETR training.
#
# Usage:
#   bash download_data.sh --split mini --output_dir /data/nuscenes
#   bash download_data.sh --split trainval --output_dir /data/nuscenes
#   bash download_data.sh --split test --output_dir /data/nuscenes
#   bash download_data.sh --backbone --output_dir /data/pretrained
#
# Requirements:
#   - wget or curl
#   - tar, unzip
#   - md5sum (for checksum verification)
#   - ~400GB free disk space for full trainval set
# =============================================================================

set -euo pipefail

NUSCENES_BASE_URL="https://www.nuscenes.org/data"
SPLIT="mini"
OUTPUT_DIR="./data/nuscenes"
DOWNLOAD_BACKBONE=false
BACKBONE_OUTPUT_DIR="./data/pretrained"
VERIFY_CHECKSUMS=true

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --split SPLIT          Dataset split to download: mini|trainval|test (default: mini)"
    echo "  --output_dir DIR       Output directory for dataset (default: ./data/nuscenes)"
    echo "  --backbone             Download pretrained backbone weights"
    echo "  --backbone_dir DIR     Output directory for backbone weights (default: ./data/pretrained)"
    echo "  --no-verify            Skip checksum verification"
    echo "  -h, --help             Show this help message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --backbone)
            DOWNLOAD_BACKBONE=true
            shift
            ;;
        --backbone_dir)
            BACKBONE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-verify)
            VERIFY_CHECKSUMS=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

check_tool() {
    if command -v "$1" &> /dev/null; then
        return 0
    fi
    return 1
}

DOWNLOADER=""
if check_tool wget; then
    DOWNLOADER="wget"
elif check_tool curl; then
    DOWNLOADER="curl"
else
    echo "ERROR: Neither wget nor curl found. Please install one of them."
    exit 1
fi

if ! check_tool tar; then
    echo "ERROR: tar not found. Please install tar."
    exit 1
fi

if [[ "$VERIFY_CHECKSUMS" == true ]] && ! check_tool md5sum; then
    echo "WARNING: md5sum not found. Skipping checksum verification."
    VERIFY_CHECKSUMS=false
fi

echo "============================================"
echo "nuScenes Dataset Downloader for PETR"
echo "============================================"
echo "  Split:       $SPLIT"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Downloader:  $DOWNLOADER"
echo "  Verify:      $VERIFY_CHECKSUMS"
echo "============================================"

download_file() {
    local url="$1"
    local output="$2"

    echo "Downloading: $(basename "$output")"
    echo "  URL: $url"

    if [[ -f "$output" ]]; then
        echo "  File already exists, skipping download."
        return 0
    fi

    if [[ "$DOWNLOADER" == "wget" ]]; then
        wget --no-check-certificate -q --show-progress -O "$output" "$url"
    else
        curl -L -# -o "$output" "$url"
    fi

    if [[ $? -ne 0 ]]; then
        echo "ERROR: Download failed for $url"
        rm -f "$output"
        return 1
    fi
    echo "  Download complete."
}

verify_checksum() {
    local file="$1"
    local expected_md5="$2"

    if [[ "$VERIFY_CHECKSUMS" != true ]]; then
        return 0
    fi

    echo "  Verifying checksum..."
    local actual_md5
    actual_md5=$(md5sum "$file" | awk '{print $1}')

    if [[ "$actual_md5" == "$expected_md5" ]]; then
        echo "  Checksum OK."
        return 0
    else
        echo "  ERROR: Checksum mismatch!"
        echo "    Expected: $expected_md5"
        echo "    Got:      $actual_md5"
        return 1
    fi
}

mkdir -p "$OUTPUT_DIR"
DOWNLOAD_DIR="${OUTPUT_DIR}/downloads"
mkdir -p "$DOWNLOAD_DIR"

declare -A MINI_FILES=(
    ["v1.0-mini.tgz"]="7308da4c3eafb582e7ef83e4dac0b9c6"
)

declare -A TRAINVAL_FILES=(
    ["v1.0-trainval01_blobs.tgz"]="d3a659e28e7e061b53bb2ffd8fc4214d"
    ["v1.0-trainval02_blobs.tgz"]="35fdd3a4eab9e3de74c3fe8459123077"
    ["v1.0-trainval03_blobs.tgz"]="9600db7de2fb4e4e5d2a7c8085dd8ac1"
    ["v1.0-trainval04_blobs.tgz"]="73c75f31b31f3e87c8bac3eca76cdb9e"
    ["v1.0-trainval05_blobs.tgz"]="3e6dd9d0db6e6e714b8419cc3ef3b5f0"
    ["v1.0-trainval06_blobs.tgz"]="5889fc24fe2a6eadb1281e4c9bfe1fc5"
    ["v1.0-trainval07_blobs.tgz"]="dbbc5e1d1daecba0ed3d5c4ec8ae7e96"
    ["v1.0-trainval08_blobs.tgz"]="65f4f9c01d6175e22d6a1e8b5498a6e3"
    ["v1.0-trainval09_blobs.tgz"]="c3b7274a2a44098d7de6d1cf2ef6ea36"
    ["v1.0-trainval10_blobs.tgz"]="3e81cc4fd3e3e1b1f3db82a6c97af428"
    ["v1.0-trainval_meta.tgz"]="3e2b26fb86f0ef7c8c1e50f3c0b6fcba"
)

declare -A TEST_FILES=(
    ["v1.0-test_blobs.tgz"]="53fdd62db7b6f56e9a1a24f4e7aeb32a"
    ["v1.0-test_meta.tgz"]="57e14ada1bc3aa1c3c1e54f56152d22a"
)

case "$SPLIT" in
    mini)
        echo ""
        echo "Downloading nuScenes mini split..."
        for file in "${!MINI_FILES[@]}"; do
            download_file "${NUSCENES_BASE_URL}/${file}" "${DOWNLOAD_DIR}/${file}"
            verify_checksum "${DOWNLOAD_DIR}/${file}" "${MINI_FILES[$file]}"
        done
        ;;
    trainval)
        echo ""
        echo "Downloading nuScenes trainval split (this will take a while)..."
        for file in "${!TRAINVAL_FILES[@]}"; do
            download_file "${NUSCENES_BASE_URL}/${file}" "${DOWNLOAD_DIR}/${file}"
            verify_checksum "${DOWNLOAD_DIR}/${file}" "${TRAINVAL_FILES[$file]}"
        done
        ;;
    test)
        echo ""
        echo "Downloading nuScenes test split..."
        for file in "${!TEST_FILES[@]}"; do
            download_file "${NUSCENES_BASE_URL}/${file}" "${DOWNLOAD_DIR}/${file}"
            verify_checksum "${DOWNLOAD_DIR}/${file}" "${TEST_FILES[$file]}"
        done
        ;;
    *)
        echo "ERROR: Unknown split '$SPLIT'. Use: mini, trainval, or test."
        exit 1
        ;;
esac

echo ""
echo "Extracting archives..."

for archive in "${DOWNLOAD_DIR}"/*.tgz; do
    if [[ -f "$archive" ]]; then
        echo "  Extracting: $(basename "$archive")"
        tar -xzf "$archive" -C "$OUTPUT_DIR"
    fi
done

for archive in "${DOWNLOAD_DIR}"/*.zip; do
    if [[ -f "$archive" ]]; then
        echo "  Extracting: $(basename "$archive")"
        unzip -qo "$archive" -d "$OUTPUT_DIR"
    fi
done

echo ""
echo "Organizing directory structure..."

EXPECTED_DIRS=(
    "${OUTPUT_DIR}/samples"
    "${OUTPUT_DIR}/sweeps"
    "${OUTPUT_DIR}/v1.0-${SPLIT}"
)

for dir in "${EXPECTED_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        echo "  Found: $dir"
    else
        echo "  Missing: $dir (may be normal depending on split)"
    fi
done

CAMERA_DIRS=(
    "CAM_FRONT"
    "CAM_FRONT_LEFT"
    "CAM_FRONT_RIGHT"
    "CAM_BACK"
    "CAM_BACK_LEFT"
    "CAM_BACK_RIGHT"
)

if [[ -d "${OUTPUT_DIR}/samples" ]]; then
    echo ""
    echo "Camera directories:"
    for cam in "${CAMERA_DIRS[@]}"; do
        cam_dir="${OUTPUT_DIR}/samples/${cam}"
        if [[ -d "$cam_dir" ]]; then
            count=$(find "$cam_dir" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
            echo "  ${cam}: ${count} images"
        fi
    done
fi

if [[ "$DOWNLOAD_BACKBONE" == true ]]; then
    echo ""
    echo "============================================"
    echo "Downloading pretrained backbone weights"
    echo "============================================"

    mkdir -p "$BACKBONE_OUTPUT_DIR"

    RESNET50_URL="https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    RESNET50_OUTPUT="${BACKBONE_OUTPUT_DIR}/resnet50_imagenet_notop.h5"

    download_file "$RESNET50_URL" "$RESNET50_OUTPUT"

    FCOS3D_URL="https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fcos3d/fcos3d_r50_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/fcos3d_r50_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210427_095416-3b5588c0.pth"
    FCOS3D_OUTPUT="${BACKBONE_OUTPUT_DIR}/fcos3d_r50_nuscenes.pth"

    download_file "$FCOS3D_URL" "$FCOS3D_OUTPUT"

    echo ""
    echo "Backbone weights saved to: $BACKBONE_OUTPUT_DIR"
    ls -lh "$BACKBONE_OUTPUT_DIR"
fi

echo ""
echo "============================================"
echo "Download complete!"
echo "============================================"
echo ""
echo "Dataset directory structure:"
echo "  ${OUTPUT_DIR}/"

if [[ -d "${OUTPUT_DIR}" ]]; then
    find "$OUTPUT_DIR" -maxdepth 2 -type d | sort | head -20 | sed 's/^/    /'
fi

echo ""
echo "Next steps:"
echo "  1. Run prepare_data.py to generate training info files:"
echo "     python scripts/prepare_data.py \\"
echo "       --data_root ${OUTPUT_DIR} \\"
echo "       --version v1.0-${SPLIT} \\"
echo "       --output_dir ./data/infos"
echo ""
echo "  2. Start training:"
echo "     python tensorflow/train.py --config configs/petr_r50.yaml"
