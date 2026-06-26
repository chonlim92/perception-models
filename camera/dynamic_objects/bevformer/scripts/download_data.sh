#!/bin/bash
# Download nuScenes dataset for BEVFormer training
# Usage: ./download_data.sh [--mini|--full] [--api-key KEY] [--output-dir DIR]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DATASET_VERSION="full"
OUTPUT_DIR="data/nuscenes"
API_KEY=""
CLEANUP_TARS=true

# Trap for cleanup on error
cleanup_on_error() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo -e "\n${RED}[ERROR] Script failed with exit code ${exit_code}${NC}"
        echo -e "${YELLOW}[INFO] Partial downloads may remain in: ${OUTPUT_DIR}${NC}"
        echo -e "${YELLOW}[INFO] Re-run the script to resume downloads.${NC}"
    fi
}
trap cleanup_on_error EXIT

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Download nuScenes dataset for BEVFormer training."
    echo ""
    echo "Options:"
    echo "  --mini              Download mini split only (~4GB)"
    echo "  --full              Download full trainval split (~300GB) [default]"
    echo "  --api-key KEY       nuScenes API key (or set NUSCENES_API_KEY env var)"
    echo "  --output-dir DIR    Output directory [default: data/nuscenes]"
    echo "  --no-cleanup        Keep tar files after extraction"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  NUSCENES_API_KEY    API key for nuScenes download authentication"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mini)
            DATASET_VERSION="mini"
            shift
            ;;
        --full)
            DATASET_VERSION="full"
            shift
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-cleanup)
            CLEANUP_TARS=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Resolve API key
if [ -z "$API_KEY" ]; then
    if [ -z "$NUSCENES_API_KEY" ]; then
        echo -e "${RED}[ERROR] No API key provided.${NC}"
        echo -e "${YELLOW}[INFO] Set NUSCENES_API_KEY environment variable or use --api-key argument.${NC}"
        echo -e "${YELLOW}[INFO] Get your API key from: https://www.nuscenes.org/nuscenes#download${NC}"
        exit 1
    fi
    API_KEY="$NUSCENES_API_KEY"
fi

# Display storage requirements
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  nuScenes Dataset Downloader${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$DATASET_VERSION" == "mini" ]; then
    echo -e "${YELLOW}[STORAGE] Mini split selected${NC}"
    echo -e "${YELLOW}  Required space: ~4 GB${NC}"
    echo -e "${YELLOW}  Temporary space for tars: ~2 GB additional${NC}"
    echo ""
    TOTAL_REQUIRED_GB=6
else
    echo -e "${YELLOW}[STORAGE] Full trainval split selected${NC}"
    echo -e "${YELLOW}  Required space: ~300 GB${NC}"
    echo -e "${YELLOW}  Temporary space for tars: ~150 GB additional${NC}"
    echo ""
    TOTAL_REQUIRED_GB=450
fi

echo -e "${GREEN}[CONFIG] Dataset version: ${DATASET_VERSION}${NC}"
echo -e "${GREEN}[CONFIG] Output directory: ${OUTPUT_DIR}${NC}"
echo -e "${GREEN}[CONFIG] Cleanup tars: ${CLEANUP_TARS}${NC}"
echo ""

# Check available disk space
AVAILABLE_KB=$(df -k "$(dirname "$OUTPUT_DIR")" 2>/dev/null | tail -1 | awk '{print $4}')
if [ -n "$AVAILABLE_KB" ]; then
    AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))
    echo -e "${BLUE}[INFO] Available disk space: ~${AVAILABLE_GB} GB${NC}"
    if [ "$AVAILABLE_GB" -lt "$TOTAL_REQUIRED_GB" ]; then
        echo -e "${RED}[WARNING] Insufficient disk space! Need ~${TOTAL_REQUIRED_GB} GB, have ~${AVAILABLE_GB} GB${NC}"
        read -p "Continue anyway? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
    fi
fi

# Check dependencies
for cmd in wget md5sum tar; do
    if ! command -v "$cmd" &> /dev/null; then
        echo -e "${RED}[ERROR] Required command not found: ${cmd}${NC}"
        echo -e "${YELLOW}[INFO] Please install ${cmd} before running this script.${NC}"
        exit 1
    fi
done

# Create output directory structure
echo -e "${GREEN}[INFO] Creating directory structure...${NC}"
mkdir -p "${OUTPUT_DIR}/v1.0-trainval"
mkdir -p "${OUTPUT_DIR}/v1.0-mini"
mkdir -p "${OUTPUT_DIR}/maps"
mkdir -p "${OUTPUT_DIR}/samples"
mkdir -p "${OUTPUT_DIR}/sweeps"
mkdir -p "${OUTPUT_DIR}/downloads"

DOWNLOAD_DIR="${OUTPUT_DIR}/downloads"

# Base URL for nuScenes downloads
BASE_URL="https://www.nuscenes.org/data"

# Define files to download with their MD5 checksums
declare -A FILES_MINI
FILES_MINI=(
    ["v1.0-mini.tar.gz"]="d4307b1ef1b7e28e2e9b18f667c2cc2c"
)

declare -A FILES_FULL_METADATA
FILES_FULL_METADATA=(
    ["v1.0-trainval_meta.tar.gz"]="edcfcbf0c0e6e26a1c1978df5c25c3e8"
)

declare -A FILES_FULL_BLOBS
FILES_FULL_BLOBS=(
    ["v1.0-trainval01_blobs.tar.gz"]="371a3398ee07e814b4b3a6514ee1548c"
    ["v1.0-trainval02_blobs.tar.gz"]="e2b8b8ac3e1e29b529317072fb68abe0"
    ["v1.0-trainval03_blobs.tar.gz"]="85b9805c1e54fcb70be26d78bd120a1a"
    ["v1.0-trainval04_blobs.tar.gz"]="27b92fcf05b4e7234e662a00b79dd363"
    ["v1.0-trainval05_blobs.tar.gz"]="a4e6d7e1ac89b61f0ff876db1d3b1f19"
    ["v1.0-trainval06_blobs.tar.gz"]="f7df4de3f34e5b0ce7bffdc5d6baf47c"
    ["v1.0-trainval07_blobs.tar.gz"]="c5c3fc9f2ed5e93d20f3e99ef9534bde"
    ["v1.0-trainval08_blobs.tar.gz"]="d89ef03c1ab5bd3e6ccc27e69ef88e34"
    ["v1.0-trainval09_blobs.tar.gz"]="0e62afed14ea279e1b83b0da98ca616c"
    ["v1.0-trainval10_blobs.tar.gz"]="18a62c80f1dfb71e4bddcd18e6f5055e"
)

declare -A FILES_MAP
FILES_MAP=(
    ["nuScenes-map-expansion-v1.3.zip"]="eeb1e98f46fedf1bbf45e4b0a0e7800b"
)

# Download function with retry and progress
download_file() {
    local filename="$1"
    local expected_md5="$2"
    local output_path="${DOWNLOAD_DIR}/${filename}"

    # Skip if already downloaded and verified
    if [ -f "$output_path" ]; then
        echo -e "${BLUE}[INFO] File exists, verifying checksum: ${filename}${NC}"
        local actual_md5
        actual_md5=$(md5sum "$output_path" | awk '{print $1}')
        if [ "$actual_md5" == "$expected_md5" ]; then
            echo -e "${GREEN}[OK] Checksum verified, skipping download: ${filename}${NC}"
            return 0
        else
            echo -e "${YELLOW}[WARN] Checksum mismatch, re-downloading: ${filename}${NC}"
            rm -f "$output_path"
        fi
    fi

    echo -e "${GREEN}[DOWNLOAD] Downloading: ${filename}${NC}"

    local max_retries=3
    local retry=0

    while [ $retry -lt $max_retries ]; do
        if wget --progress=bar:force:noscroll \
                --header="Authorization: Bearer ${API_KEY}" \
                --continue \
                -O "$output_path" \
                "${BASE_URL}/${filename}" 2>&1; then

            # Verify MD5
            echo -e "${BLUE}[INFO] Verifying checksum: ${filename}${NC}"
            local actual_md5
            actual_md5=$(md5sum "$output_path" | awk '{print $1}')
            if [ "$actual_md5" == "$expected_md5" ]; then
                echo -e "${GREEN}[OK] Checksum verified: ${filename}${NC}"
                return 0
            else
                echo -e "${RED}[ERROR] Checksum mismatch for ${filename}${NC}"
                echo -e "${RED}  Expected: ${expected_md5}${NC}"
                echo -e "${RED}  Got:      ${actual_md5}${NC}"
                rm -f "$output_path"
                retry=$((retry + 1))
            fi
        else
            echo -e "${YELLOW}[WARN] Download failed, retry ${retry}/${max_retries}: ${filename}${NC}"
            retry=$((retry + 1))
        fi
    done

    echo -e "${RED}[ERROR] Failed to download after ${max_retries} retries: ${filename}${NC}"
    return 1
}

# Extract function
extract_file() {
    local filename="$1"
    local filepath="${DOWNLOAD_DIR}/${filename}"

    if [ ! -f "$filepath" ]; then
        echo -e "${RED}[ERROR] File not found for extraction: ${filepath}${NC}"
        return 1
    fi

    echo -e "${GREEN}[EXTRACT] Extracting: ${filename}${NC}"

    if [[ "$filename" == *.tar.gz ]] || [[ "$filename" == *.tgz ]]; then
        tar -xzf "$filepath" -C "$OUTPUT_DIR"
    elif [[ "$filename" == *.zip ]]; then
        unzip -qo "$filepath" -d "$OUTPUT_DIR"
    else
        echo -e "${RED}[ERROR] Unknown archive format: ${filename}${NC}"
        return 1
    fi

    echo -e "${GREEN}[OK] Extracted: ${filename}${NC}"

    # Cleanup tar file if requested
    if [ "$CLEANUP_TARS" = true ]; then
        echo -e "${BLUE}[CLEANUP] Removing: ${filename}${NC}"
        rm -f "$filepath"
    fi
}

# Track download statistics
TOTAL_FILES=0
DOWNLOADED_FILES=0
FAILED_FILES=0

# Download and extract based on version
if [ "$DATASET_VERSION" == "mini" ]; then
    echo -e "\n${BLUE}=== Downloading nuScenes Mini Split ===${NC}\n"

    for filename in "${!FILES_MINI[@]}"; do
        TOTAL_FILES=$((TOTAL_FILES + 1))
        if download_file "$filename" "${FILES_MINI[$filename]}"; then
            extract_file "$filename"
            DOWNLOADED_FILES=$((DOWNLOADED_FILES + 1))
        else
            FAILED_FILES=$((FAILED_FILES + 1))
        fi
    done

    # Also download maps for mini
    echo -e "\n${BLUE}=== Downloading Map Expansion ===${NC}\n"
    for filename in "${!FILES_MAP[@]}"; do
        TOTAL_FILES=$((TOTAL_FILES + 1))
        if download_file "$filename" "${FILES_MAP[$filename]}"; then
            extract_file "$filename"
            DOWNLOADED_FILES=$((DOWNLOADED_FILES + 1))
        else
            FAILED_FILES=$((FAILED_FILES + 1))
        fi
    done

else
    echo -e "\n${BLUE}=== Downloading nuScenes Full Trainval Metadata ===${NC}\n"

    for filename in "${!FILES_FULL_METADATA[@]}"; do
        TOTAL_FILES=$((TOTAL_FILES + 1))
        if download_file "$filename" "${FILES_FULL_METADATA[$filename]}"; then
            extract_file "$filename"
            DOWNLOADED_FILES=$((DOWNLOADED_FILES + 1))
        else
            FAILED_FILES=$((FAILED_FILES + 1))
        fi
    done

    echo -e "\n${BLUE}=== Downloading nuScenes Full Trainval Sensor Data ===${NC}\n"

    for filename in "${!FILES_FULL_BLOBS[@]}"; do
        TOTAL_FILES=$((TOTAL_FILES + 1))
        if download_file "$filename" "${FILES_FULL_BLOBS[$filename]}"; then
            extract_file "$filename"
            DOWNLOADED_FILES=$((DOWNLOADED_FILES + 1))
        else
            FAILED_FILES=$((FAILED_FILES + 1))
        fi
    done

    echo -e "\n${BLUE}=== Downloading Map Expansion ===${NC}\n"

    for filename in "${!FILES_MAP[@]}"; do
        TOTAL_FILES=$((TOTAL_FILES + 1))
        if download_file "$filename" "${FILES_MAP[$filename]}"; then
            extract_file "$filename"
            DOWNLOADED_FILES=$((DOWNLOADED_FILES + 1))
        else
            FAILED_FILES=$((FAILED_FILES + 1))
        fi
    done
fi

# Clean up empty downloads directory
if [ "$CLEANUP_TARS" = true ]; then
    rmdir "${DOWNLOAD_DIR}" 2>/dev/null || true
fi

# Print summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Download Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}  Dataset version:  ${DATASET_VERSION}${NC}"
echo -e "${GREEN}  Output directory: ${OUTPUT_DIR}${NC}"
echo -e "${GREEN}  Total files:      ${TOTAL_FILES}${NC}"
echo -e "${GREEN}  Downloaded:       ${DOWNLOADED_FILES}${NC}"

if [ $FAILED_FILES -gt 0 ]; then
    echo -e "${RED}  Failed:           ${FAILED_FILES}${NC}"
fi

echo ""
echo -e "${BLUE}[INFO] Directory structure:${NC}"
if [ -d "$OUTPUT_DIR" ]; then
    find "$OUTPUT_DIR" -maxdepth 1 -type d | sort | while read -r dir; do
        local_size=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
        echo -e "  ${dir} (${local_size})"
    done
fi

echo ""
if [ $FAILED_FILES -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] All downloads completed successfully!${NC}"
    echo -e "${GREEN}[INFO] Next step: Run prepare_data.py to generate training info files.${NC}"
else
    echo -e "${RED}[WARNING] Some downloads failed. Re-run the script to retry.${NC}"
    exit 1
fi
