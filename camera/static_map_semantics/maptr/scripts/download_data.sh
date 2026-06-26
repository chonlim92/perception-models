#!/bin/bash
# ==============================================================================
# MapTR Data Download Script
# ==============================================================================
# Downloads nuScenes dataset with map expansion and optionally Argoverse 2.
#
# Usage:
#   bash scripts/download_data.sh --dataset nuscenes --output_dir /data/nuscenes
#   bash scripts/download_data.sh --dataset argoverse2 --output_dir /data/argo2
#   bash scripts/download_data.sh --dataset all --output_dir /data
#
# Prerequisites:
#   - wget or curl installed
#   - For nuScenes: Create account at https://www.nuscenes.org/ and obtain token
#   - For Argoverse 2: Install s5cmd or aws-cli for S3 download
#   - Sufficient disk space (~400GB for nuScenes full, ~1TB for Argoverse 2)
#
# Environment Variables:
#   NUSCENES_TOKEN  - Authentication token for nuScenes API
#   AWS_PROFILE     - AWS profile for Argoverse 2 S3 access (optional)
# ==============================================================================

set -euo pipefail

# Default configuration
DATASET="nuscenes"
OUTPUT_DIR="/data"
VERSION="v1.0-trainval"
NUM_PARALLEL=4
VERIFY_CHECKSUMS=true

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --dataset      Dataset to download: nuscenes, argoverse2, all (default: nuscenes)
  --output_dir   Base output directory (default: /data)
  --version      nuScenes version: v1.0-trainval, v1.0-mini (default: v1.0-trainval)
  --parallel     Number of parallel downloads (default: 4)
  --no-verify    Skip checksum verification
  --help         Show this help message

Examples:
  $0 --dataset nuscenes --output_dir /data/nuscenes --version v1.0-trainval
  $0 --dataset nuscenes --output_dir /data/nuscenes --version v1.0-mini
  $0 --dataset argoverse2 --output_dir /data/argoverse2
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --parallel)
            NUM_PARALLEL="$2"
            shift 2
            ;;
        --no-verify)
            VERIFY_CHECKSUMS=false
            shift
            ;;
        --help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required tools
check_dependencies() {
    local missing=()

    if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
        missing+=("wget or curl")
    fi

    if ! command -v md5sum &> /dev/null && ! command -v md5 &> /dev/null; then
        missing+=("md5sum")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing[*]}"
        exit 1
    fi
}

# Download a file with resume support
download_file() {
    local url="$1"
    local output_path="$2"
    local description="${3:-}"

    if [[ -f "$output_path" ]]; then
        log_info "File already exists, skipping: $output_path"
        return 0
    fi

    local dir
    dir=$(dirname "$output_path")
    mkdir -p "$dir"

    log_info "Downloading: ${description:-$url}"

    if command -v wget &> /dev/null; then
        wget -c -q --show-progress -O "$output_path" "$url"
    else
        curl -L -C - -o "$output_path" "$url"
    fi
}

# Verify file checksum
verify_checksum() {
    local filepath="$1"
    local expected_md5="$2"

    if [[ "$VERIFY_CHECKSUMS" != "true" ]]; then
        return 0
    fi

    log_info "Verifying checksum: $(basename "$filepath")"

    local actual_md5
    if command -v md5sum &> /dev/null; then
        actual_md5=$(md5sum "$filepath" | awk '{print $1}')
    else
        actual_md5=$(md5 -q "$filepath")
    fi

    if [[ "$actual_md5" == "$expected_md5" ]]; then
        log_info "Checksum OK: $(basename "$filepath")"
        return 0
    else
        log_error "Checksum mismatch for $(basename "$filepath")"
        log_error "  Expected: $expected_md5"
        log_error "  Actual:   $actual_md5"
        return 1
    fi
}

# ==============================================================================
# nuScenes Download
# ==============================================================================
download_nuscenes() {
    local nuscenes_dir="$OUTPUT_DIR/nuscenes"

    log_info "=========================================="
    log_info "Downloading nuScenes dataset ($VERSION)"
    log_info "Output directory: $nuscenes_dir"
    log_info "=========================================="

    # Check for authentication token
    if [[ -z "${NUSCENES_TOKEN:-}" ]]; then
        log_warn "NUSCENES_TOKEN not set."
        log_warn ""
        log_warn "To download nuScenes, you need to:"
        log_warn "  1. Create an account at https://www.nuscenes.org/"
        log_warn "  2. Accept the Terms of Use"
        log_warn "  3. Get your download token from the download page"
        log_warn "  4. Set: export NUSCENES_TOKEN=your_token_here"
        log_warn ""
        log_warn "Alternatively, download manually from:"
        log_warn "  https://www.nuscenes.org/nuscenes#download"
        log_warn ""
        log_warn "Required files for MapTR:"
        log_warn "  - Metadata (v1.0-trainval or v1.0-mini)"
        log_warn "  - Camera blobs (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,"
        log_warn "    CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT)"
        log_warn "  - Map expansion pack v1.3"
        log_warn ""
        log_error "Set NUSCENES_TOKEN and re-run, or download manually."
        exit 1
    fi

    local base_url="https://www.nuscenes.org/data"
    local headers="Authorization: Bearer ${NUSCENES_TOKEN}"

    # Create directory structure
    mkdir -p "$nuscenes_dir"/{maps,samples,sweeps,v1.0-trainval}

    # Define files to download based on version
    local -a files_to_download=()

    if [[ "$VERSION" == "v1.0-mini" ]]; then
        files_to_download=(
            "v1.0-mini.tgz"
        )
    else
        files_to_download=(
            "v1.0-trainval_meta.tgz"
            "v1.0-trainval01_blobs.tgz"
            "v1.0-trainval02_blobs.tgz"
            "v1.0-trainval03_blobs.tgz"
            "v1.0-trainval04_blobs.tgz"
            "v1.0-trainval05_blobs.tgz"
            "v1.0-trainval06_blobs.tgz"
            "v1.0-trainval07_blobs.tgz"
            "v1.0-trainval08_blobs.tgz"
            "v1.0-trainval09_blobs.tgz"
            "v1.0-trainval10_blobs.tgz"
        )
    fi

    # Download main dataset files
    for fname in "${files_to_download[@]}"; do
        local url="${base_url}/${fname}"
        local output="${nuscenes_dir}/${fname}"

        if [[ -f "$output" ]]; then
            log_info "Already downloaded: $fname"
        else
            log_info "Downloading: $fname"
            if command -v wget &> /dev/null; then
                wget -c --header="$headers" -O "$output" "$url"
            else
                curl -L -H "$headers" -C - -o "$output" "$url"
            fi
        fi
    done

    # Download map expansion pack (critical for MapTR)
    log_info "Downloading nuScenes Map Expansion v1.3..."
    local map_expansion_url="${base_url}/nuScenes-map-expansion-v1.3.zip"
    local map_expansion_file="${nuscenes_dir}/nuScenes-map-expansion-v1.3.zip"

    if [[ ! -f "$map_expansion_file" ]]; then
        if command -v wget &> /dev/null; then
            wget -c --header="$headers" -O "$map_expansion_file" "$map_expansion_url"
        else
            curl -L -H "$headers" -C - -o "$map_expansion_file" "$map_expansion_url"
        fi
    fi

    # Extract files
    log_info "Extracting dataset files..."
    for fname in "${files_to_download[@]}"; do
        local archive="${nuscenes_dir}/${fname}"
        if [[ -f "$archive" ]]; then
            log_info "Extracting: $fname"
            tar -xzf "$archive" -C "$nuscenes_dir"
        fi
    done

    # Extract map expansion
    log_info "Extracting map expansion pack..."
    if [[ -f "$map_expansion_file" ]]; then
        unzip -o -q "$map_expansion_file" -d "$nuscenes_dir/maps"
    fi

    # Verify directory structure
    log_info "Verifying directory structure..."
    local expected_dirs=("maps" "samples" "v1.0-trainval")
    local all_present=true

    for dir in "${expected_dirs[@]}"; do
        if [[ ! -d "$nuscenes_dir/$dir" ]]; then
            log_warn "Missing directory: $nuscenes_dir/$dir"
            all_present=false
        fi
    done

    # Check for map expansion files
    local map_files=("$nuscenes_dir"/maps/expansion/*.json)
    if [[ -e "${map_files[0]}" ]]; then
        log_info "Map expansion files found: $(ls "$nuscenes_dir"/maps/expansion/*.json 2>/dev/null | wc -l) files"
    else
        # Check alternative location
        map_files=("$nuscenes_dir"/maps/*.json)
        if [[ -e "${map_files[0]}" ]]; then
            log_info "Map files found in maps/ directory"
        else
            log_warn "Map expansion files not found. Please verify extraction."
        fi
    fi

    if [[ "$all_present" == "true" ]]; then
        log_info "nuScenes download complete!"
    else
        log_warn "Some directories missing. Please check extraction."
    fi

    # Print expected structure
    log_info ""
    log_info "Expected directory structure:"
    log_info "  $nuscenes_dir/"
    log_info "  ├── maps/"
    log_info "  │   ├── expansion/"
    log_info "  │   │   ├── singapore-onenorth.json"
    log_info "  │   │   ├── singapore-hollandvillage.json"
    log_info "  │   │   ├── singapore-queenstown.json"
    log_info "  │   │   └── boston-seaport.json"
    log_info "  │   ├── 36092f0b03a857c6.png"
    log_info "  │   └── ..."
    log_info "  ├── samples/"
    log_info "  │   ├── CAM_FRONT/"
    log_info "  │   ├── CAM_FRONT_LEFT/"
    log_info "  │   ├── CAM_FRONT_RIGHT/"
    log_info "  │   ├── CAM_BACK/"
    log_info "  │   ├── CAM_BACK_LEFT/"
    log_info "  │   └── CAM_BACK_RIGHT/"
    log_info "  └── v1.0-trainval/"
    log_info "      ├── sample.json"
    log_info "      ├── sample_data.json"
    log_info "      ├── ego_pose.json"
    log_info "      ├── calibrated_sensor.json"
    log_info "      ├── map.json"
    log_info "      └── ..."
}

# ==============================================================================
# Argoverse 2 Download
# ==============================================================================
download_argoverse2() {
    local argo_dir="$OUTPUT_DIR/argoverse2"

    log_info "=========================================="
    log_info "Downloading Argoverse 2 Sensor Dataset"
    log_info "Output directory: $argo_dir"
    log_info "=========================================="

    # Check for s5cmd or aws cli
    if ! command -v s5cmd &> /dev/null && ! command -v aws &> /dev/null; then
        log_warn "Neither s5cmd nor aws-cli found."
        log_warn ""
        log_warn "To download Argoverse 2, install one of:"
        log_warn "  - s5cmd (recommended, faster): https://github.com/peak/s5cmd"
        log_warn "  - aws-cli: pip install awscli"
        log_warn ""
        log_warn "Then run:"
        log_warn "  s5cmd --no-sign-request cp 's3://argoverse/datasets/av2/sensor/*' $argo_dir/"
        log_warn ""
        log_warn "Or download from: https://www.argoverse.org/av2.html"
        exit 1
    fi

    mkdir -p "$argo_dir"

    local s3_base="s3://argoverse/datasets/av2/sensor"

    if command -v s5cmd &> /dev/null; then
        log_info "Using s5cmd for parallel download..."

        # Download train split
        log_info "Downloading train split..."
        s5cmd --no-sign-request cp "${s3_base}/train/*" "$argo_dir/train/"

        # Download val split
        log_info "Downloading val split..."
        s5cmd --no-sign-request cp "${s3_base}/val/*" "$argo_dir/val/"

        # Download test split
        log_info "Downloading test split..."
        s5cmd --no-sign-request cp "${s3_base}/test/*" "$argo_dir/test/"

    elif command -v aws &> /dev/null; then
        log_info "Using aws-cli for download..."

        aws s3 sync --no-sign-request "${s3_base}/train/" "$argo_dir/train/"
        aws s3 sync --no-sign-request "${s3_base}/val/" "$argo_dir/val/"
        aws s3 sync --no-sign-request "${s3_base}/test/" "$argo_dir/test/"
    fi

    # Download map data
    log_info "Downloading map data..."
    local map_s3="s3://argoverse/datasets/av2/map"

    if command -v s5cmd &> /dev/null; then
        s5cmd --no-sign-request cp "${map_s3}/*" "$argo_dir/map/"
    else
        aws s3 sync --no-sign-request "${map_s3}/" "$argo_dir/map/"
    fi

    log_info "Argoverse 2 download complete!"
    log_info ""
    log_info "Expected directory structure:"
    log_info "  $argo_dir/"
    log_info "  ├── train/"
    log_info "  │   ├── <log_id>/"
    log_info "  │   │   ├── sensors/cameras/"
    log_info "  │   │   ├── calibration/"
    log_info "  │   │   └── city_SE3_egovehicle.feather"
    log_info "  │   └── ..."
    log_info "  ├── val/"
    log_info "  ├── test/"
    log_info "  └── map/"
    log_info "      ├── <log_id>/"
    log_info "      │   └── map_<city>.json"
    log_info "      └── ..."
}

# ==============================================================================
# Main
# ==============================================================================
main() {
    check_dependencies

    log_info "MapTR Data Download Script"
    log_info "Dataset: $DATASET"
    log_info "Output: $OUTPUT_DIR"
    log_info ""

    case "$DATASET" in
        nuscenes)
            download_nuscenes
            ;;
        argoverse2)
            download_argoverse2
            ;;
        all)
            download_nuscenes
            download_argoverse2
            ;;
        *)
            log_error "Unknown dataset: $DATASET"
            log_error "Valid options: nuscenes, argoverse2, all"
            exit 1
            ;;
    esac

    log_info ""
    log_info "=========================================="
    log_info "Download complete!"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Run data preparation:"
    log_info "     python scripts/prepare_data.py --nuscenes_root $OUTPUT_DIR/nuscenes --output_dir data/processed"
    log_info ""
    log_info "  2. Start training:"
    log_info "     python pytorch/train.py --data_root data/processed --config configs/maptr_r50_nuscenes.yaml"
    log_info "=========================================="
}

main "$@"
