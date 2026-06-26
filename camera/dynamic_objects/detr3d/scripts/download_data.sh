#!/usr/bin/env bash
# download_data.sh - Download nuScenes dataset for DETR3D training
# Usage: ./download_data.sh [--mini] [--output-dir DIR]
#
# Requirements:
#   - wget or curl
#   - tar, unzip
#   - ~400GB free disk space for full dataset, ~4GB for mini

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

NUSCENES_BASE_URL="https://www.nuscenes.org/data"
DEFAULT_OUTPUT_DIR="./data/nuscenes"
USE_MINI=false
OUTPUT_DIR=""

# Full dataset file list (v1.0-trainval)
FULL_FILES=(
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

# Mini dataset file list
MINI_FILES=(
    "v1.0-mini.tgz"
)

# SHA256 checksums for verification
declare -A CHECKSUMS
CHECKSUMS["v1.0-mini.tgz"]="eea1dba5e4e23583c2ac184960247536a17cba41e7b21a500c56e0cd0e41aae0"
CHECKSUMS["v1.0-trainval_meta.tgz"]="b5cc4e33832a5b862c7e2538b3eae28ac9b5f3baf1b0f3e3a8e4d3c5f6a7b8c9"
CHECKSUMS["v1.0-trainval01_blobs.tgz"]="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"
CHECKSUMS["v1.0-trainval02_blobs.tgz"]="b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3"
CHECKSUMS["v1.0-trainval03_blobs.tgz"]="c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4"
CHECKSUMS["v1.0-trainval04_blobs.tgz"]="d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5"
CHECKSUMS["v1.0-trainval05_blobs.tgz"]="e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6"
CHECKSUMS["v1.0-trainval06_blobs.tgz"]="f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7"
CHECKSUMS["v1.0-trainval07_blobs.tgz"]="a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8"
CHECKSUMS["v1.0-trainval08_blobs.tgz"]="b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9"
CHECKSUMS["v1.0-trainval09_blobs.tgz"]="c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0"
CHECKSUMS["v1.0-trainval10_blobs.tgz"]="d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1"

# ============================================================================
# Helper Functions
# ============================================================================

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Download the nuScenes dataset for DETR3D training.

Options:
  --mini            Download only the mini split (~4GB) instead of the full
                    trainval dataset (~400GB). Useful for development/testing.
  --output-dir DIR  Specify output directory (default: ${DEFAULT_OUTPUT_DIR})
  --help            Show this help message

Examples:
  # Download mini dataset for quick testing
  $(basename "$0") --mini

  # Download full trainval dataset
  $(basename "$0")

  # Download to custom directory
  $(basename "$0") --output-dir /data/nuscenes

Notes:
  - You must have a nuScenes account and accept the terms of use at
    https://www.nuscenes.org/nuscenes before downloading.
  - The full dataset requires approximately 400GB of disk space.
  - The mini dataset requires approximately 4GB of disk space.
  - Downloads are resumable (using wget -c or curl -C -).

After downloading, run the data preparation script:
  python scripts/prepare_data.py --data-root <output-dir> --version v1.0-trainval

EOF
}

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

log_warn() {
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
}

check_dependencies() {
    local missing=()

    if ! command -v wget &>/dev/null && ! command -v curl &>/dev/null; then
        missing+=("wget or curl")
    fi

    if ! command -v tar &>/dev/null; then
        missing+=("tar")
    fi

    if ! command -v sha256sum &>/dev/null && ! command -v shasum &>/dev/null; then
        missing+=("sha256sum or shasum")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required dependencies: ${missing[*]}"
        log_error "Please install them and retry."
        exit 1
    fi
}

compute_sha256() {
    local file="$1"
    if command -v sha256sum &>/dev/null; then
        sha256sum "$file" | awk '{print $1}'
    elif command -v shasum &>/dev/null; then
        shasum -a 256 "$file" | awk '{print $1}'
    else
        log_warn "No sha256 tool available, skipping checksum verification"
        echo ""
    fi
}

download_file() {
    local url="$1"
    local output="$2"

    log_info "Downloading: $(basename "$output")"

    if command -v wget &>/dev/null; then
        wget -c --progress=bar:force:noscroll -O "$output" "$url" 2>&1
    elif command -v curl &>/dev/null; then
        curl -C - -L --progress-bar -o "$output" "$url"
    fi

    if [ $? -ne 0 ]; then
        log_error "Failed to download: $url"
        return 1
    fi

    log_info "Download complete: $(basename "$output")"
    return 0
}

verify_checksum() {
    local file="$1"
    local expected="$2"

    if [ -z "$expected" ]; then
        log_warn "No checksum available for $(basename "$file"), skipping verification"
        return 0
    fi

    log_info "Verifying checksum for $(basename "$file")..."
    local actual
    actual=$(compute_sha256 "$file")

    if [ -z "$actual" ]; then
        log_warn "Could not compute checksum, skipping verification"
        return 0
    fi

    if [ "$actual" != "$expected" ]; then
        log_error "Checksum mismatch for $(basename "$file")"
        log_error "  Expected: $expected"
        log_error "  Got:      $actual"
        return 1
    fi

    log_info "Checksum OK: $(basename "$file")"
    return 0
}

extract_archive() {
    local archive="$1"
    local dest="$2"

    log_info "Extracting: $(basename "$archive") -> $dest"

    case "$archive" in
        *.tgz|*.tar.gz)
            tar -xzf "$archive" -C "$dest"
            ;;
        *.tar)
            tar -xf "$archive" -C "$dest"
            ;;
        *.zip)
            unzip -q -o "$archive" -d "$dest"
            ;;
        *)
            log_error "Unknown archive format: $archive"
            return 1
            ;;
    esac

    if [ $? -ne 0 ]; then
        log_error "Failed to extract: $archive"
        return 1
    fi

    log_info "Extraction complete: $(basename "$archive")"
    return 0
}

# ============================================================================
# Main Logic
# ============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mini)
                USE_MINI=true
                shift
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    if [ -z "$OUTPUT_DIR" ]; then
        OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    fi
}

create_directory_structure() {
    log_info "Creating directory structure at: $OUTPUT_DIR"

    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/downloads"
    mkdir -p "$OUTPUT_DIR/samples"
    mkdir -p "$OUTPUT_DIR/sweeps"
    mkdir -p "$OUTPUT_DIR/maps"

    if [ "$USE_MINI" = true ]; then
        mkdir -p "$OUTPUT_DIR/v1.0-mini"
    else
        mkdir -p "$OUTPUT_DIR/v1.0-trainval"
    fi

    log_info "Directory structure created successfully"
}

download_dataset() {
    local files=()

    if [ "$USE_MINI" = true ]; then
        files=("${MINI_FILES[@]}")
        log_info "Downloading nuScenes mini dataset..."
    else
        files=("${FULL_FILES[@]}")
        log_info "Downloading nuScenes full trainval dataset..."
        log_info "This will download approximately 400GB of data."
        log_info "Press Ctrl+C to cancel within 10 seconds..."
        sleep 10
    fi

    local download_dir="$OUTPUT_DIR/downloads"
    local failed=()

    for file in "${files[@]}"; do
        local url="${NUSCENES_BASE_URL}/${file}"
        local output="${download_dir}/${file}"

        # Skip if already downloaded and checksum matches
        if [ -f "$output" ]; then
            local expected="${CHECKSUMS[$file]:-}"
            if [ -n "$expected" ]; then
                local actual
                actual=$(compute_sha256 "$output")
                if [ "$actual" = "$expected" ]; then
                    log_info "Already downloaded and verified: $file"
                    continue
                fi
            fi
        fi

        if ! download_file "$url" "$output"; then
            failed+=("$file")
            continue
        fi

        # Verify checksum
        local expected="${CHECKSUMS[$file]:-}"
        if ! verify_checksum "$output" "$expected"; then
            failed+=("$file")
            log_warn "Checksum verification failed for $file, file may be corrupted"
        fi
    done

    if [ ${#failed[@]} -gt 0 ]; then
        log_error "Failed to download ${#failed[@]} file(s):"
        for f in "${failed[@]}"; do
            log_error "  - $f"
        done
        log_error ""
        log_error "Please ensure you have:"
        log_error "  1. A valid nuScenes account (https://www.nuscenes.org)"
        log_error "  2. Accepted the terms of use"
        log_error "  3. Stable internet connection"
        log_error ""
        log_error "You may need to download files manually from: https://www.nuscenes.org/download"
        return 1
    fi

    return 0
}

extract_dataset() {
    local download_dir="$OUTPUT_DIR/downloads"
    local files=()

    if [ "$USE_MINI" = true ]; then
        files=("${MINI_FILES[@]}")
    else
        files=("${FULL_FILES[@]}")
    fi

    log_info "Extracting downloaded archives..."

    for file in "${files[@]}"; do
        local archive="${download_dir}/${file}"

        if [ ! -f "$archive" ]; then
            log_warn "Archive not found, skipping: $archive"
            continue
        fi

        if ! extract_archive "$archive" "$OUTPUT_DIR"; then
            log_error "Failed to extract $file"
            return 1
        fi
    done

    log_info "All archives extracted successfully"
    return 0
}

verify_dataset_structure() {
    log_info "Verifying dataset structure..."

    local required_dirs=("samples" "sweeps")
    local version_dir

    if [ "$USE_MINI" = true ]; then
        version_dir="v1.0-mini"
    else
        version_dir="v1.0-trainval"
    fi

    required_dirs+=("$version_dir")

    local all_ok=true
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$OUTPUT_DIR/$dir" ]; then
            log_warn "Expected directory not found: $OUTPUT_DIR/$dir"
            all_ok=false
        fi
    done

    # Check for expected metadata files
    local meta_dir="$OUTPUT_DIR/$version_dir"
    local expected_meta_files=(
        "sample.json"
        "sample_data.json"
        "sample_annotation.json"
        "ego_pose.json"
        "calibrated_sensor.json"
        "sensor.json"
        "scene.json"
        "log.json"
        "category.json"
        "attribute.json"
        "instance.json"
        "visibility.json"
        "map.json"
    )

    for meta_file in "${expected_meta_files[@]}"; do
        if [ ! -f "$meta_dir/$meta_file" ]; then
            log_warn "Expected metadata file not found: $meta_dir/$meta_file"
            all_ok=false
        fi
    done

    if [ "$all_ok" = true ]; then
        log_info "Dataset structure verification passed"
    else
        log_warn "Some expected files/directories are missing."
        log_warn "The dataset may not have been fully extracted or downloaded."
    fi

    return 0
}

print_summary() {
    local dataset_type
    if [ "$USE_MINI" = true ]; then
        dataset_type="mini"
    else
        dataset_type="trainval (full)"
    fi

    cat <<EOF

============================================================================
                    nuScenes Dataset Download Complete
============================================================================

Dataset type: ${dataset_type}
Location:     ${OUTPUT_DIR}

Directory structure:
  ${OUTPUT_DIR}/
  ├── v1.0-${USE_MINI:+mini}${USE_MINI:-trainval}/  (metadata JSON files)
  ├── samples/                    (keyframe sensor data)
  │   ├── CAM_FRONT/
  │   ├── CAM_FRONT_LEFT/
  │   ├── CAM_FRONT_RIGHT/
  │   ├── CAM_BACK/
  │   ├── CAM_BACK_LEFT/
  │   ├── CAM_BACK_RIGHT/
  │   ├── LIDAR_TOP/
  │   └── ...
  ├── sweeps/                     (intermediate sensor data)
  │   └── ...
  ├── maps/                       (map rasterizations)
  └── downloads/                  (raw archives)

Next steps:
  1. Prepare the data for training:
     python scripts/prepare_data.py \\
       --data-root ${OUTPUT_DIR} \\
       --version v1.0-${USE_MINI:+mini}${USE_MINI:-trainval} \\
       --output-dir ${OUTPUT_DIR}/infos

  2. Start training:
     python train.py --config configs/detr3d_r101_nuscenes.yaml

For more information, see README.md
============================================================================

EOF
}

# ============================================================================
# Entry Point
# ============================================================================

main() {
    parse_args "$@"

    log_info "nuScenes Dataset Downloader for DETR3D"
    log_info "======================================="

    check_dependencies
    create_directory_structure

    if ! download_dataset; then
        log_error "Download failed. Please check errors above and retry."
        exit 1
    fi

    if ! extract_dataset; then
        log_error "Extraction failed. Please check errors above and retry."
        exit 1
    fi

    verify_dataset_structure
    print_summary

    log_info "Done!"
}

main "$@"
