#!/usr/bin/env bash
# =============================================================================
# download_data.sh - Download nuScenes dataset for CRAFT model training/evaluation
#
# This script downloads the nuScenes dataset from the official website,
# extracts archives, verifies directory structure, and creates necessary symlinks.
#
# Usage:
#   ./download_data.sh --version mini --output /data/nuscenes --token YOUR_TOKEN
#   ./download_data.sh --version trainval --output /data/nuscenes --token YOUR_TOKEN
#   ./download_data.sh --version test --output /data/nuscenes --token YOUR_TOKEN
#
# Requirements:
#   - wget or curl
#   - tar
#   - md5sum (optional, for verification)
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

NUSCENES_BASE_URL="https://www.nuscenes.org/data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
VERSION=""
OUTPUT_DIR=""
TOKEN=""
SYMLINK_DIR=""
SKIP_EXTRACT=false
SKIP_VERIFY=false
NUM_RETRIES=3
DOWNLOAD_TOOL=""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Download and prepare the nuScenes dataset for CRAFT model.

Required Options:
  --version VERSION    Dataset version to download: mini, trainval, or test
  --output DIR         Output directory for downloaded data
  --token TOKEN        nuScenes access token (from https://www.nuscenes.org)

Optional:
  --symlink DIR        Create a symlink from this path to the output directory
  --skip-extract       Skip extraction (files already extracted)
  --skip-verify        Skip directory structure verification
  --retries N          Number of download retry attempts (default: 3)
  --tool TOOL          Force download tool: wget or curl (auto-detected if omitted)
  -h, --help           Show this help message

Examples:
  # Download mini split for quick testing
  $(basename "$0") --version mini --output /data/nuscenes --token abc123

  # Download full trainval split
  $(basename "$0") --version trainval --output /data/nuscenes --token abc123

  # Download and create symlink for project
  $(basename "$0") --version trainval --output /data/nuscenes --token abc123 \\
      --symlink ${PROJECT_ROOT}/data/nuscenes
EOF
}

detect_download_tool() {
    if [ -n "$DOWNLOAD_TOOL" ]; then
        if ! command -v "$DOWNLOAD_TOOL" &>/dev/null; then
            log_error "Specified download tool '$DOWNLOAD_TOOL' not found"
            exit 1
        fi
        return
    fi

    if command -v wget &>/dev/null; then
        DOWNLOAD_TOOL="wget"
    elif command -v curl &>/dev/null; then
        DOWNLOAD_TOOL="curl"
    else
        log_error "Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    log_info "Using download tool: $DOWNLOAD_TOOL"
}

# Download a file with resume support and progress bar
# Args: $1=URL, $2=output_path, $3=auth_token
download_file() {
    local url="$1"
    local output="$2"
    local token="$3"
    local attempt=1

    while [ $attempt -le $NUM_RETRIES ]; do
        log_info "Downloading $(basename "$output") (attempt $attempt/$NUM_RETRIES)..."

        if [ "$DOWNLOAD_TOOL" = "wget" ]; then
            # wget with resume (-c), progress bar, and authentication
            if wget -c \
                --header="Authorization: Bearer ${token}" \
                --progress=bar:force:noscroll \
                --tries=3 \
                --timeout=60 \
                --waitretry=10 \
                -O "$output" \
                "$url" 2>&1; then
                log_success "Downloaded: $(basename "$output")"
                return 0
            fi
        elif [ "$DOWNLOAD_TOOL" = "curl" ]; then
            # curl with resume (-C -), progress bar, and authentication
            if curl -C - \
                -H "Authorization: Bearer ${token}" \
                --progress-bar \
                --retry 3 \
                --retry-delay 10 \
                --connect-timeout 60 \
                -L \
                -o "$output" \
                "$url" 2>&1; then
                log_success "Downloaded: $(basename "$output")"
                return 0
            fi
        fi

        log_warn "Download attempt $attempt failed. Retrying..."
        attempt=$((attempt + 1))
        sleep $((attempt * 5))
    done

    log_error "Failed to download $(basename "$output") after $NUM_RETRIES attempts"
    return 1
}

# Get file list based on dataset version
get_file_list() {
    local version="$1"
    local -n files_ref=$2

    case "$version" in
        mini)
            files_ref=(
                "v1.0-mini.tgz"
            )
            ;;
        trainval)
            files_ref=(
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
            ;;
        test)
            files_ref=(
                "v1.0-test_meta.tgz"
                "v1.0-test01_blobs.tgz"
                "v1.0-test02_blobs.tgz"
                "v1.0-test03_blobs.tgz"
                "v1.0-test04_blobs.tgz"
                "v1.0-test05_blobs.tgz"
                "v1.0-test06_blobs.tgz"
                "v1.0-test07_blobs.tgz"
                "v1.0-test08_blobs.tgz"
            )
            ;;
        *)
            log_error "Unknown version: $version"
            exit 1
            ;;
    esac
}

# Extract a tar.gz archive
extract_archive() {
    local archive="$1"
    local dest="$2"

    log_info "Extracting $(basename "$archive")..."

    if [ ! -f "$archive" ]; then
        log_error "Archive not found: $archive"
        return 1
    fi

    # Get archive size for progress indication
    local size
    size=$(du -h "$archive" | cut -f1)
    log_info "  Archive size: $size"

    # Extract with verbose output piped to a counter
    local count=0
    tar -xzf "$archive" -C "$dest" &
    local tar_pid=$!

    # Show a simple progress indicator
    while kill -0 "$tar_pid" 2>/dev/null; do
        printf "\r  Extracting... [%d seconds elapsed]" $count
        sleep 1
        count=$((count + 1))
    done

    wait "$tar_pid"
    local exit_code=$?
    printf "\r  Extraction complete (%d seconds)          \n" $count

    if [ $exit_code -ne 0 ]; then
        log_error "Failed to extract $(basename "$archive")"
        return 1
    fi

    log_success "Extracted: $(basename "$archive")"
    return 0
}

# Verify the expected directory structure after extraction
verify_structure() {
    local base_dir="$1"
    local version="$2"
    local errors=0

    log_info "Verifying directory structure..."

    # Common directories that should exist
    local expected_dirs=()

    case "$version" in
        mini)
            expected_dirs=(
                "v1.0-mini"
                "samples"
                "samples/CAM_FRONT"
                "samples/CAM_FRONT_LEFT"
                "samples/CAM_FRONT_RIGHT"
                "samples/CAM_BACK"
                "samples/CAM_BACK_LEFT"
                "samples/CAM_BACK_RIGHT"
                "samples/RADAR_FRONT"
                "samples/RADAR_FRONT_LEFT"
                "samples/RADAR_FRONT_RIGHT"
                "samples/RADAR_BACK_LEFT"
                "samples/RADAR_BACK_RIGHT"
                "sweeps"
            )
            ;;
        trainval)
            expected_dirs=(
                "v1.0-trainval"
                "samples"
                "samples/CAM_FRONT"
                "samples/CAM_FRONT_LEFT"
                "samples/CAM_FRONT_RIGHT"
                "samples/CAM_BACK"
                "samples/CAM_BACK_LEFT"
                "samples/CAM_BACK_RIGHT"
                "samples/RADAR_FRONT"
                "samples/RADAR_FRONT_LEFT"
                "samples/RADAR_FRONT_RIGHT"
                "samples/RADAR_BACK_LEFT"
                "samples/RADAR_BACK_RIGHT"
                "sweeps"
                "sweeps/RADAR_FRONT"
                "sweeps/RADAR_FRONT_LEFT"
                "sweeps/RADAR_FRONT_RIGHT"
                "sweeps/RADAR_BACK_LEFT"
                "sweeps/RADAR_BACK_RIGHT"
            )
            ;;
        test)
            expected_dirs=(
                "v1.0-test"
                "samples"
                "samples/CAM_FRONT"
                "samples/RADAR_FRONT"
                "sweeps"
            )
            ;;
    esac

    for dir in "${expected_dirs[@]}"; do
        if [ -d "${base_dir}/${dir}" ]; then
            log_success "  Found: ${dir}/"
        else
            log_warn "  Missing: ${dir}/"
            errors=$((errors + 1))
        fi
    done

    # Check for JSON metadata files
    local meta_dir=""
    case "$version" in
        mini) meta_dir="${base_dir}/v1.0-mini" ;;
        trainval) meta_dir="${base_dir}/v1.0-trainval" ;;
        test) meta_dir="${base_dir}/v1.0-test" ;;
    esac

    local expected_json_files=(
        "scene.json"
        "sample.json"
        "sample_data.json"
        "ego_pose.json"
        "calibrated_sensor.json"
        "sensor.json"
        "log.json"
        "map.json"
    )

    if [ "$version" != "test" ]; then
        expected_json_files+=(
            "sample_annotation.json"
            "instance.json"
            "category.json"
            "attribute.json"
            "visibility.json"
        )
    fi

    log_info "Checking metadata files in ${meta_dir}..."
    for json_file in "${expected_json_files[@]}"; do
        if [ -f "${meta_dir}/${json_file}" ]; then
            local file_size
            file_size=$(du -h "${meta_dir}/${json_file}" | cut -f1)
            log_success "  Found: ${json_file} (${file_size})"
        else
            log_warn "  Missing: ${json_file}"
            errors=$((errors + 1))
        fi
    done

    if [ $errors -gt 0 ]; then
        log_warn "Verification completed with $errors warnings"
        return 1
    else
        log_success "All expected files and directories found!"
        return 0
    fi
}

# Create symlinks for project use
create_symlinks() {
    local source_dir="$1"
    local link_path="$2"

    if [ -z "$link_path" ]; then
        return 0
    fi

    log_info "Creating symlink: ${link_path} -> ${source_dir}"

    # Create parent directory if needed
    local link_parent
    link_parent="$(dirname "$link_path")"
    mkdir -p "$link_parent"

    # Remove existing symlink if present
    if [ -L "$link_path" ]; then
        rm "$link_path"
    elif [ -d "$link_path" ]; then
        log_warn "Directory already exists at symlink path: $link_path"
        log_warn "Skipping symlink creation. Remove it manually if you want a symlink."
        return 0
    fi

    ln -sf "$source_dir" "$link_path"
    log_success "Symlink created: ${link_path} -> ${source_dir}"
}

# Print summary of downloaded data
print_summary() {
    local base_dir="$1"
    local version="$2"

    echo ""
    echo "============================================================================="
    echo "                        DOWNLOAD SUMMARY"
    echo "============================================================================="
    echo ""
    echo "  Version:     ${version}"
    echo "  Location:    ${base_dir}"
    echo ""

    # Count files by type
    local num_cam_samples=0
    local num_radar_samples=0
    local num_radar_sweeps=0

    if [ -d "${base_dir}/samples" ]; then
        # Count camera images
        for cam_dir in CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT; do
            if [ -d "${base_dir}/samples/${cam_dir}" ]; then
                local count
                count=$(find "${base_dir}/samples/${cam_dir}" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
                num_cam_samples=$((num_cam_samples + count))
            fi
        done

        # Count radar point clouds
        for radar_dir in RADAR_FRONT RADAR_FRONT_LEFT RADAR_FRONT_RIGHT RADAR_BACK_LEFT RADAR_BACK_RIGHT; do
            if [ -d "${base_dir}/samples/${radar_dir}" ]; then
                local count
                count=$(find "${base_dir}/samples/${radar_dir}" -name "*.pcd" 2>/dev/null | wc -l)
                num_radar_samples=$((num_radar_samples + count))
            fi
        done
    fi

    if [ -d "${base_dir}/sweeps" ]; then
        for radar_dir in RADAR_FRONT RADAR_FRONT_LEFT RADAR_FRONT_RIGHT RADAR_BACK_LEFT RADAR_BACK_RIGHT; do
            if [ -d "${base_dir}/sweeps/${radar_dir}" ]; then
                local count
                count=$(find "${base_dir}/sweeps/${radar_dir}" -name "*.pcd" 2>/dev/null | wc -l)
                num_radar_sweeps=$((num_radar_sweeps + count))
            fi
        done
    fi

    echo "  Data Statistics:"
    echo "  ----------------"
    echo "  Camera samples:     ${num_cam_samples}"
    echo "  Radar samples:      ${num_radar_samples}"
    echo "  Radar sweeps:       ${num_radar_sweeps}"
    echo ""

    # Total disk usage
    local total_size
    total_size=$(du -sh "$base_dir" 2>/dev/null | cut -f1)
    echo "  Total disk usage:   ${total_size}"
    echo ""

    # Scene count from metadata
    local meta_dir=""
    case "$version" in
        mini) meta_dir="${base_dir}/v1.0-mini" ;;
        trainval) meta_dir="${base_dir}/v1.0-trainval" ;;
        test) meta_dir="${base_dir}/v1.0-test" ;;
    esac

    if [ -f "${meta_dir}/scene.json" ]; then
        local num_scenes
        num_scenes=$(python3 -c "import json; print(len(json.load(open('${meta_dir}/scene.json'))))" 2>/dev/null || echo "N/A")
        echo "  Number of scenes:   ${num_scenes}"
    fi

    if [ -f "${meta_dir}/sample.json" ]; then
        local num_samples
        num_samples=$(python3 -c "import json; print(len(json.load(open('${meta_dir}/sample.json'))))" 2>/dev/null || echo "N/A")
        echo "  Number of samples:  ${num_samples}"
    fi

    echo ""
    echo "============================================================================="
    echo ""
}

# =============================================================================
# Main Script
# =============================================================================

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --token)
            TOKEN="$2"
            shift 2
            ;;
        --symlink)
            SYMLINK_DIR="$2"
            shift 2
            ;;
        --skip-extract)
            SKIP_EXTRACT=true
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY=true
            shift
            ;;
        --retries)
            NUM_RETRIES="$2"
            shift 2
            ;;
        --tool)
            DOWNLOAD_TOOL="$2"
            shift 2
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

# Validate required arguments
if [ -z "$VERSION" ]; then
    log_error "Missing required argument: --version"
    print_usage
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    log_error "Missing required argument: --output"
    print_usage
    exit 1
fi

if [ -z "$TOKEN" ]; then
    log_error "Missing required argument: --token"
    print_usage
    exit 1
fi

# Validate version
if [[ ! "$VERSION" =~ ^(mini|trainval|test)$ ]]; then
    log_error "Invalid version: $VERSION. Must be one of: mini, trainval, test"
    exit 1
fi

# Detect download tool
detect_download_tool

# Create output directory
mkdir -p "$OUTPUT_DIR"
DOWNLOAD_DIR="${OUTPUT_DIR}/.downloads"
mkdir -p "$DOWNLOAD_DIR"

log_info "============================================="
log_info "  nuScenes Dataset Downloader for CRAFT"
log_info "============================================="
log_info ""
log_info "Version:      ${VERSION}"
log_info "Output:       ${OUTPUT_DIR}"
log_info "Download dir: ${DOWNLOAD_DIR}"
log_info "Tool:         ${DOWNLOAD_TOOL}"
log_info ""

# Get list of files to download
declare -a FILE_LIST
get_file_list "$VERSION" FILE_LIST

log_info "Files to download: ${#FILE_LIST[@]}"
for f in "${FILE_LIST[@]}"; do
    echo "  - $f"
done
echo ""

# Download phase
download_failures=0
for filename in "${FILE_LIST[@]}"; do
    url="${NUSCENES_BASE_URL}/${filename}"
    output_path="${DOWNLOAD_DIR}/${filename}"

    # Skip if already downloaded and has non-zero size
    if [ -f "$output_path" ] && [ -s "$output_path" ]; then
        log_info "File already exists, checking for resume: ${filename}"
    fi

    if ! download_file "$url" "$output_path" "$TOKEN"; then
        download_failures=$((download_failures + 1))
        log_error "Failed to download: ${filename}"
    fi
done

if [ $download_failures -gt 0 ]; then
    log_error "$download_failures file(s) failed to download"
    log_error "You may re-run this script to resume partial downloads"
    exit 1
fi

log_success "All files downloaded successfully!"

# Extraction phase
if [ "$SKIP_EXTRACT" = false ]; then
    log_info ""
    log_info "Starting extraction..."

    extract_failures=0
    for filename in "${FILE_LIST[@]}"; do
        archive_path="${DOWNLOAD_DIR}/${filename}"

        if ! extract_archive "$archive_path" "$OUTPUT_DIR"; then
            extract_failures=$((extract_failures + 1))
        fi
    done

    if [ $extract_failures -gt 0 ]; then
        log_error "$extract_failures archive(s) failed to extract"
        exit 1
    fi

    log_success "All archives extracted successfully!"
else
    log_info "Skipping extraction (--skip-extract)"
fi

# Verification phase
if [ "$SKIP_VERIFY" = false ]; then
    log_info ""
    verify_structure "$OUTPUT_DIR" "$VERSION"
else
    log_info "Skipping verification (--skip-verify)"
fi

# Create symlinks
if [ -n "$SYMLINK_DIR" ]; then
    create_symlinks "$OUTPUT_DIR" "$SYMLINK_DIR"
fi

# Print summary
print_summary "$OUTPUT_DIR" "$VERSION"

log_success "Done! Dataset is ready for use with CRAFT."
log_info "Next step: Run prepare_data.py to generate info files for training."
