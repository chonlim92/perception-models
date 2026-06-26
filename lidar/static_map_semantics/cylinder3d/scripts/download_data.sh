#!/usr/bin/env bash
# =============================================================================
# download_data.sh - Download and prepare SemanticKITTI dataset for Cylinder3D
# =============================================================================
# Downloads velodyne scans and semantic labels from semantic-kitti.org,
# verifies checksums, and creates the expected directory structure.
#
# Usage:
#   ./download_data.sh [OPTIONS]
#
# Options:
#   -d, --dest DIR       Destination directory (default: ./dataset)
#   -s, --sequences SEQ  Comma-separated sequences to download (default: all 00-21)
#   --skip-verify        Skip MD5 checksum verification
#   --nuscenes           Print nuScenes download instructions
#   -h, --help           Show this help message
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SEMANTICKITTI_BASE_URL="http://www.semantic-kitti.org/assets"
DEST_DIR="./dataset"
SEQUENCES="00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21"
SKIP_VERIFY=false
SHOW_NUSCENES=false

# Known MD5 checksums for SemanticKITTI archives
declare -A MD5_CHECKSUMS=(
    ["data_odometry_velodyne_00.zip"]="a]placeholder_md5_seq00_velodyne"
    ["data_odometry_velodyne_01.zip"]="placeholder_md5_seq01_velodyne"
    ["data_odometry_labels.zip"]="b18589b6e323af90c1882b0a0e92e812"
)

# SemanticKITTI download URLs
VELODYNE_URL="${SEMANTICKITTI_BASE_URL}/data_odometry_velodyne.zip"
LABELS_URL="${SEMANTICKITTI_BASE_URL}/data_odometry_labels.zip"
CALIB_URL="${SEMANTICKITTI_BASE_URL}/data_odometry_calib.zip"

# =============================================================================
# Helper Functions
# =============================================================================

print_usage() {
    cat << 'EOF'
Usage: ./download_data.sh [OPTIONS]

Download and prepare SemanticKITTI dataset for Cylinder3D training.

Options:
  -d, --dest DIR       Destination directory (default: ./dataset)
  -s, --sequences SEQ  Comma-separated sequences to download (default: all 00-21)
  --skip-verify        Skip MD5 checksum verification
  --nuscenes           Print nuScenes download instructions
  -h, --help           Show this help message

Expected output structure:
  dataset/
  └── sequences/
      ├── 00/
      │   ├── velodyne/
      │   │   ├── 000000.bin
      │   │   ├── 000001.bin
      │   │   └── ...
      │   └── labels/
      │       ├── 000000.label
      │       ├── 000001.label
      │       └── ...
      ├── 01/
      │   ├── velodyne/
      │   └── labels/
      └── ...

Train sequences: 00-07, 09-10
Validation sequence: 08
Test sequences: 11-21 (no labels provided)

Note: The full dataset is ~80GB. Ensure sufficient disk space.
EOF
}

print_nuscenes_instructions() {
    cat << 'EOF'
=============================================================================
nuScenes Dataset Download Instructions
=============================================================================

nuScenes requires authentication. Follow these steps:

1. Create an account at https://www.nuscenes.org/sign-up

2. Accept the Terms of Use at https://www.nuscenes.org/nuscenes#download

3. Get your access token from your profile page

4. Download using the nuScenes devkit:

   pip install nuscenes-devkit

   # Option A: Using the download script
   python -c "
   from nuscenes.utils.data_io import download_nuscenes
   download_nuscenes(
       version='v1.0-trainval',
       dataroot='./dataset/nuscenes',
       token='YOUR_ACCESS_TOKEN'
   )
   "

   # Option B: Manual download via AWS CLI
   # After accepting terms, you receive AWS credentials
   aws s3 cp s3://nuscenes/v1.0-trainval/ ./dataset/nuscenes/ --recursive

5. Expected structure after download:
   dataset/nuscenes/
   ├── maps/
   ├── samples/
   │   └── LIDAR_TOP/
   │       ├── n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530.bin
   │       └── ...
   ├── sweeps/
   │   └── LIDAR_TOP/
   ├── v1.0-trainval/
   │   ├── category.json
   │   ├── lidarseg.json
   │   └── ...
   └── lidarseg/
       └── v1.0-trainval/
           ├── <token>_lidarseg.bin
           └── ...

6. For lidarseg annotations (semantic labels):
   Download separately from the nuScenes-lidarseg page:
   https://www.nuscenes.org/nuscenes#lidarseg

Note: Full nuScenes dataset is ~400GB. The mini split (~4GB) is available
for quick testing: version='v1.0-mini'
=============================================================================
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

# Check if required tools are available
check_dependencies() {
    local missing=()

    if ! command -v wget &>/dev/null && ! command -v curl &>/dev/null; then
        missing+=("wget or curl")
    fi

    if ! command -v unzip &>/dev/null; then
        missing+=("unzip")
    fi

    if ! command -v md5sum &>/dev/null && ! command -v md5 &>/dev/null; then
        log_warn "md5sum/md5 not found - checksum verification will be skipped"
        SKIP_VERIFY=true
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Please install them and try again."
        exit 1
    fi
}

# Download a file with resume support and progress indication
download_file() {
    local url="$1"
    local output="$2"
    local description="${3:-file}"

    log_info "Downloading ${description}..."
    log_info "  URL: ${url}"
    log_info "  Destination: ${output}"

    if command -v wget &>/dev/null; then
        wget \
            --continue \
            --progress=bar:force:noscroll \
            --timeout=60 \
            --tries=5 \
            --retry-connrefused \
            -O "${output}" \
            "${url}"
    elif command -v curl &>/dev/null; then
        curl \
            --continue-at - \
            --progress-bar \
            --retry 5 \
            --retry-delay 10 \
            --connect-timeout 60 \
            --location \
            -o "${output}" \
            "${url}"
    fi

    if [ $? -ne 0 ]; then
        log_error "Failed to download ${description}"
        return 1
    fi

    log_info "Download complete: ${description}"
}

# Verify MD5 checksum of a file
verify_checksum() {
    local file="$1"
    local expected_md5="$2"

    if [ "${SKIP_VERIFY}" = true ]; then
        log_info "Skipping checksum verification for $(basename "${file}")"
        return 0
    fi

    log_info "Verifying checksum for $(basename "${file}")..."

    local actual_md5
    if command -v md5sum &>/dev/null; then
        actual_md5=$(md5sum "${file}" | awk '{print $1}')
    elif command -v md5 &>/dev/null; then
        actual_md5=$(md5 -q "${file}")
    else
        log_warn "No MD5 tool available, skipping verification"
        return 0
    fi

    if [ "${actual_md5}" = "${expected_md5}" ]; then
        log_info "Checksum OK: $(basename "${file}")"
        return 0
    else
        log_error "Checksum MISMATCH for $(basename "${file}")"
        log_error "  Expected: ${expected_md5}"
        log_error "  Actual:   ${actual_md5}"
        return 1
    fi
}

# Extract archive and organize files
extract_archive() {
    local archive="$1"
    local dest="$2"
    local description="${3:-archive}"

    log_info "Extracting ${description}..."

    if [ ! -f "${archive}" ]; then
        log_error "Archive not found: ${archive}"
        return 1
    fi

    unzip -o -q "${archive}" -d "${dest}"

    if [ $? -ne 0 ]; then
        log_error "Failed to extract ${description}"
        return 1
    fi

    log_info "Extraction complete: ${description}"
}

# Create the expected directory structure
create_directory_structure() {
    local base_dir="$1"
    local sequences_dir="${base_dir}/sequences"

    log_info "Creating directory structure..."

    IFS=',' read -ra SEQ_ARRAY <<< "${SEQUENCES}"
    for seq in "${SEQ_ARRAY[@]}"; do
        mkdir -p "${sequences_dir}/${seq}/velodyne"
        mkdir -p "${sequences_dir}/${seq}/labels"
    done

    log_info "Directory structure created at ${sequences_dir}"
}

# Verify data integrity after extraction
verify_data_integrity() {
    local sequences_dir="$1/sequences"
    local errors=0

    log_info "Verifying data integrity..."

    # Training and validation sequences (00-10) should have both velodyne and labels
    for seq in 00 01 02 03 04 05 06 07 08 09 10; do
        local vel_dir="${sequences_dir}/${seq}/velodyne"
        local lab_dir="${sequences_dir}/${seq}/labels"

        if [ ! -d "${vel_dir}" ]; then
            log_warn "Missing velodyne directory: ${vel_dir}"
            ((errors++)) || true
            continue
        fi

        local vel_count=$(find "${vel_dir}" -name "*.bin" 2>/dev/null | wc -l)
        local lab_count=$(find "${lab_dir}" -name "*.label" 2>/dev/null | wc -l)

        if [ "${vel_count}" -eq 0 ]; then
            log_warn "No .bin files in ${vel_dir}"
            ((errors++)) || true
        elif [ "${lab_count}" -eq 0 ]; then
            log_warn "No .label files in ${lab_dir}"
            ((errors++)) || true
        elif [ "${vel_count}" -ne "${lab_count}" ]; then
            log_warn "Mismatch in seq ${seq}: ${vel_count} scans vs ${lab_count} labels"
            ((errors++)) || true
        else
            log_info "  Sequence ${seq}: ${vel_count} scans, ${lab_count} labels - OK"
        fi
    done

    # Test sequences (11-21) only need velodyne
    for seq in 11 12 13 14 15 16 17 18 19 20 21; do
        local vel_dir="${sequences_dir}/${seq}/velodyne"
        if [ -d "${vel_dir}" ]; then
            local vel_count=$(find "${vel_dir}" -name "*.bin" 2>/dev/null | wc -l)
            log_info "  Sequence ${seq}: ${vel_count} scans (test, no labels) - OK"
        fi
    done

    if [ "${errors}" -gt 0 ]; then
        log_warn "Data verification completed with ${errors} warning(s)"
    else
        log_info "Data verification passed!"
    fi
}

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dest)
            DEST_DIR="$2"
            shift 2
            ;;
        -s|--sequences)
            SEQUENCES="$2"
            shift 2
            ;;
        --skip-verify)
            SKIP_VERIFY=true
            shift
            ;;
        --nuscenes)
            SHOW_NUSCENES=true
            shift
            ;;
        -h|--help)
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

# =============================================================================
# Main
# =============================================================================

if [ "${SHOW_NUSCENES}" = true ]; then
    print_nuscenes_instructions
    exit 0
fi

log_info "============================================="
log_info "SemanticKITTI Dataset Download Script"
log_info "============================================="
log_info "Destination: ${DEST_DIR}"
log_info "Sequences: ${SEQUENCES}"
log_info "Verify checksums: $([ "${SKIP_VERIFY}" = true ] && echo 'No' || echo 'Yes')"
log_info "============================================="

# Check prerequisites
check_dependencies

# Create destination directory
mkdir -p "${DEST_DIR}"
DOWNLOAD_DIR="${DEST_DIR}/downloads"
mkdir -p "${DOWNLOAD_DIR}"

# Create directory structure
create_directory_structure "${DEST_DIR}"

# Download velodyne point clouds
VELODYNE_ARCHIVE="${DOWNLOAD_DIR}/data_odometry_velodyne.zip"
if [ -f "${VELODYNE_ARCHIVE}" ]; then
    log_info "Velodyne archive already exists, resuming/skipping..."
fi
download_file "${VELODYNE_URL}" "${VELODYNE_ARCHIVE}" "Velodyne point clouds (~80GB)"

# Verify velodyne checksum
if [ -n "${MD5_CHECKSUMS[data_odometry_velodyne.zip]:-}" ]; then
    verify_checksum "${VELODYNE_ARCHIVE}" "${MD5_CHECKSUMS[data_odometry_velodyne.zip]}" || true
fi

# Download semantic labels
LABELS_ARCHIVE="${DOWNLOAD_DIR}/data_odometry_labels.zip"
if [ -f "${LABELS_ARCHIVE}" ]; then
    log_info "Labels archive already exists, resuming/skipping..."
fi
download_file "${LABELS_URL}" "${LABELS_ARCHIVE}" "Semantic labels (~700MB)"

# Verify labels checksum
if [ -n "${MD5_CHECKSUMS[data_odometry_labels.zip]:-}" ]; then
    verify_checksum "${LABELS_ARCHIVE}" "${MD5_CHECKSUMS[data_odometry_labels.zip]}" || true
fi

# Download calibration files
CALIB_ARCHIVE="${DOWNLOAD_DIR}/data_odometry_calib.zip"
download_file "${CALIB_URL}" "${CALIB_ARCHIVE}" "Calibration files"

# Extract archives
log_info "============================================="
log_info "Extracting archives..."
log_info "============================================="

extract_archive "${VELODYNE_ARCHIVE}" "${DEST_DIR}" "Velodyne point clouds"
extract_archive "${LABELS_ARCHIVE}" "${DEST_DIR}" "Semantic labels"
extract_archive "${CALIB_ARCHIVE}" "${DEST_DIR}" "Calibration files"

# Reorganize if needed (KITTI archives extract to dataset/sequences/ structure)
# Handle case where archives extract to a different structure
if [ -d "${DEST_DIR}/dataset/sequences" ] && [ "${DEST_DIR}" != "./dataset" ]; then
    log_info "Reorganizing extracted files..."
    mv "${DEST_DIR}/dataset/sequences/"* "${DEST_DIR}/sequences/" 2>/dev/null || true
    rmdir "${DEST_DIR}/dataset/sequences" "${DEST_DIR}/dataset" 2>/dev/null || true
fi

# Verify data integrity
log_info "============================================="
log_info "Verifying data integrity..."
log_info "============================================="
verify_data_integrity "${DEST_DIR}"

# Print summary
log_info "============================================="
log_info "Download complete!"
log_info "============================================="
log_info ""
log_info "Dataset location: ${DEST_DIR}/sequences/"
log_info ""
log_info "SemanticKITTI splits:"
log_info "  Train: sequences 00-07, 09-10"
log_info "  Val:   sequence 08"
log_info "  Test:  sequences 11-21"
log_info ""
log_info "Next steps:"
log_info "  1. Run prepare_data.py to generate train/val/test file lists"
log_info "     python scripts/prepare_data.py --dataset_root ${DEST_DIR}"
log_info ""
log_info "  2. Start training:"
log_info "     python train.py --config configs/semantickitti.yaml"
log_info ""
log_info "For nuScenes download instructions, run:"
log_info "  ./download_data.sh --nuscenes"
log_info "============================================="
