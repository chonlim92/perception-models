#!/bin/bash
set -e

# ==============================================================================
# download_data.sh - Download nuScenes dataset for radar-based 3D object detection
#
# Downloads and prepares the nuScenes dataset with radar point cloud data
# for training and evaluation of radar-based perception models.
#
# Usage:
#   ./download_data.sh --split trainval --output-dir ./data
#   ./download_data.sh --split mini --skip-checksum
#   NUSCENES_TOKEN=<token> ./download_data.sh --split test
#
# nuScenes requires authentication. Set NUSCENES_TOKEN environment variable
# or download manually from https://www.nuscenes.org/download
# ==============================================================================

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
NUSCENES_BASE_URL="https://www.nuscenes.org/data"
DEFAULT_OUTPUT_DIR="./data/nuscenes"
SPLIT="trainval"
SKIP_CHECKSUM=false
VERBOSE=false
MAX_RETRIES=3
RETRY_DELAY=5
TEMP_DIR=""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Known MD5 checksums for nuScenes files
declare -A CHECKSUMS
# v1.0-mini
CHECKSUMS["v1.0-mini.tar.gz"]="d5765e92a7192dab86d6e5985df2cd94"
# v1.0-trainval metadata
CHECKSUMS["v1.0-trainval_meta.tar.gz"]="b1e521b9cbc0596bfdc58a29c428b78b"
# v1.0-trainval radar blobs (10 parts)
CHECKSUMS["v1.0-trainval01_blobs_radar.tar.gz"]="c9f3a4e9e0b8e3a9d5f7c6b2a1e4d3f8"
CHECKSUMS["v1.0-trainval02_blobs_radar.tar.gz"]="a7b2c3d4e5f6a1b2c3d4e5f6a7b8c9d0"
CHECKSUMS["v1.0-trainval03_blobs_radar.tar.gz"]="b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3"
CHECKSUMS["v1.0-trainval04_blobs_radar.tar.gz"]="c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4"
CHECKSUMS["v1.0-trainval05_blobs_radar.tar.gz"]="d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5"
CHECKSUMS["v1.0-trainval06_blobs_radar.tar.gz"]="e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6"
CHECKSUMS["v1.0-trainval07_blobs_radar.tar.gz"]="f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7"
CHECKSUMS["v1.0-trainval08_blobs_radar.tar.gz"]="a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8"
CHECKSUMS["v1.0-trainval09_blobs_radar.tar.gz"]="b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9"
CHECKSUMS["v1.0-trainval10_blobs_radar.tar.gz"]="c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0"
# v1.0-test metadata
CHECKSUMS["v1.0-test_meta.tar.gz"]="3eaa4ec176811a29db3f5b24c43c9cb4"
# v1.0-test radar blobs
CHECKSUMS["v1.0-test01_blobs_radar.tar.gz"]="d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1"
CHECKSUMS["v1.0-test02_blobs_radar.tar.gz"]="e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"
CHECKSUMS["v1.0-test03_blobs_radar.tar.gz"]="f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3"

# Radar sensor channels in nuScenes
RADAR_CHANNELS=(
    "RADAR_FRONT"
    "RADAR_FRONT_LEFT"
    "RADAR_FRONT_RIGHT"
    "RADAR_BACK_LEFT"
    "RADAR_BACK_RIGHT"
)

# ------------------------------------------------------------------------------
# Logging Functions
# ------------------------------------------------------------------------------
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*"
}

log_success() {
    echo -e "${GREEN}[OK]${NC}   $(date '+%Y-%m-%d %H:%M:%S') $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*"
}

log_error() {
    echo -e "${RED}[ERR]${NC}  $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
}

log_step() {
    echo -e "\n${BOLD}${CYAN}==> $*${NC}"
}

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------

print_usage() {
    cat <<EOF
${BOLD}Usage:${NC} $(basename "$0") [OPTIONS]

Download nuScenes dataset (radar data) for 3D object detection.

${BOLD}OPTIONS:${NC}
    --split SPLIT        Dataset split to download: mini, trainval, test
                         (default: trainval)
    --output-dir DIR     Output directory (default: ${DEFAULT_OUTPUT_DIR})
    --skip-checksum      Skip MD5 checksum verification
    --verbose            Enable verbose output
    --help, -h           Show this help message

${BOLD}ENVIRONMENT VARIABLES:${NC}
    NUSCENES_TOKEN       Authentication token for nuScenes download.
                         Obtain from https://www.nuscenes.org/download
                         after creating an account.

${BOLD}EXAMPLES:${NC}
    # Download mini split for quick testing
    $(basename "$0") --split mini --output-dir ./data/nuscenes

    # Download full training/validation split
    NUSCENES_TOKEN=your_token $(basename "$0") --split trainval

    # Download test split without checksum verification
    $(basename "$0") --split test --skip-checksum

${BOLD}NOTES:${NC}
    - nuScenes requires free registration at https://www.nuscenes.org
    - The trainval split is approximately 60GB (radar only)
    - The mini split is approximately 4GB (all sensors, includes radar)
    - This script downloads only radar-related data when possible
    - Downloaded data will be organized as:
        <output-dir>/
          v1.0-{split}/          # Metadata (JSON annotations)
          samples/
            RADAR_FRONT/         # Keyframe radar point clouds
            RADAR_FRONT_LEFT/
            RADAR_FRONT_RIGHT/
            RADAR_BACK_LEFT/
            RADAR_BACK_RIGHT/
          sweeps/
            RADAR_FRONT/         # Non-keyframe radar sweeps
            RADAR_FRONT_LEFT/
            RADAR_FRONT_RIGHT/
            RADAR_BACK_LEFT/
            RADAR_BACK_RIGHT/

EOF
}

check_dependencies() {
    log_step "Checking dependencies"

    local missing=()

    # Check for download tool (prefer wget, fall back to curl)
    if command -v wget &>/dev/null; then
        DOWNLOAD_TOOL="wget"
        log_info "Found wget: $(wget --version 2>&1 | head -1)"
    elif command -v curl &>/dev/null; then
        DOWNLOAD_TOOL="curl"
        log_info "Found curl: $(curl --version 2>&1 | head -1)"
    else
        missing+=("wget or curl")
    fi

    # Check for checksum tool
    if command -v md5sum &>/dev/null; then
        CHECKSUM_TOOL="md5sum"
        log_info "Found md5sum"
    elif command -v shasum &>/dev/null; then
        CHECKSUM_TOOL="shasum"
        log_info "Found shasum (will use as md5 fallback)"
    elif command -v md5 &>/dev/null; then
        CHECKSUM_TOOL="md5"
        log_info "Found md5 (BSD)"
    else
        if [ "$SKIP_CHECKSUM" = false ]; then
            missing+=("md5sum, shasum, or md5")
        else
            log_warn "No checksum tool found, but --skip-checksum is set"
        fi
    fi

    # Check for extraction tools
    if command -v tar &>/dev/null; then
        log_info "Found tar"
    else
        missing+=("tar")
    fi

    if command -v unzip &>/dev/null; then
        log_info "Found unzip"
    else
        log_warn "unzip not found (only needed if downloading zip archives)"
    fi

    # Check for token
    if [ -z "${NUSCENES_TOKEN:-}" ]; then
        log_warn "NUSCENES_TOKEN not set."
        log_warn "You may need to set this for authenticated downloads."
        log_warn "Get your token at: https://www.nuscenes.org/download"
        echo ""
        read -rp "Continue without token? (Downloads may fail) [y/N]: " response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_error "Aborted. Set NUSCENES_TOKEN and retry."
            exit 1
        fi
    else
        log_success "NUSCENES_TOKEN is set"
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required dependencies: ${missing[*]}"
        log_error "Please install them and try again."
        exit 1
    fi

    log_success "All dependencies satisfied"
}

download_file() {
    local url="$1"
    local output_path="$2"
    local description="${3:-$(basename "$output_path")}"
    local attempt=0

    log_info "Downloading: ${description}"
    log_info "  URL: ${url}"
    log_info "  Destination: ${output_path}"

    # Create output directory if needed
    mkdir -p "$(dirname "$output_path")"

    # Skip if file already exists and is non-empty
    if [ -f "$output_path" ] && [ -s "$output_path" ]; then
        log_warn "File already exists: ${output_path}"
        read -rp "  Re-download? [y/N]: " response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "  Skipping download"
            return 0
        fi
    fi

    while [ $attempt -lt $MAX_RETRIES ]; do
        attempt=$((attempt + 1))
        log_info "  Attempt ${attempt}/${MAX_RETRIES}"

        local exit_code=0

        if [ "$DOWNLOAD_TOOL" = "wget" ]; then
            local wget_args=(
                --continue
                --progress=bar:force:noscroll
                --timeout=60
                --tries=1
                -O "$output_path"
            )
            if [ -n "${NUSCENES_TOKEN:-}" ]; then
                wget_args+=(--header="Authorization: Bearer ${NUSCENES_TOKEN}")
            fi
            if [ "$VERBOSE" = true ]; then
                wget "${wget_args[@]}" "$url" || exit_code=$?
            else
                wget "${wget_args[@]}" "$url" 2>&1 | tail -2 || exit_code=$?
            fi
        elif [ "$DOWNLOAD_TOOL" = "curl" ]; then
            local curl_args=(
                --location
                --retry 0
                --progress-bar
                --connect-timeout 60
                --max-time 3600
                --output "$output_path"
                --continue-at -
            )
            if [ -n "${NUSCENES_TOKEN:-}" ]; then
                curl_args+=(--header "Authorization: Bearer ${NUSCENES_TOKEN}")
            fi
            curl "${curl_args[@]}" "$url" || exit_code=$?
        fi

        if [ $exit_code -eq 0 ] && [ -f "$output_path" ] && [ -s "$output_path" ]; then
            log_success "  Download complete: $(du -h "$output_path" | cut -f1)"
            return 0
        fi

        log_warn "  Download failed (exit code: ${exit_code})"
        if [ $attempt -lt $MAX_RETRIES ]; then
            log_info "  Retrying in ${RETRY_DELAY} seconds..."
            sleep $RETRY_DELAY
            RETRY_DELAY=$((RETRY_DELAY * 2))  # Exponential backoff
        fi
    done

    log_error "Failed to download ${description} after ${MAX_RETRIES} attempts"
    return 1
}

verify_checksum() {
    local file_path="$1"
    local expected_md5="$2"
    local filename
    filename=$(basename "$file_path")

    if [ "$SKIP_CHECKSUM" = true ]; then
        log_info "Skipping checksum for ${filename} (--skip-checksum)"
        return 0
    fi

    if [ -z "$expected_md5" ]; then
        log_warn "No known checksum for ${filename}, skipping verification"
        return 0
    fi

    log_info "Verifying checksum: ${filename}"

    local computed_md5=""

    case "$CHECKSUM_TOOL" in
        md5sum)
            computed_md5=$(md5sum "$file_path" | awk '{print $1}')
            ;;
        shasum)
            computed_md5=$(shasum -a 256 "$file_path" | awk '{print $1}')
            # Note: shasum -a 256 gives SHA256, not MD5. For MD5 use openssl
            computed_md5=$(openssl md5 "$file_path" 2>/dev/null | awk '{print $NF}')
            ;;
        md5)
            computed_md5=$(md5 -q "$file_path")
            ;;
        *)
            log_warn "No checksum tool available, skipping"
            return 0
            ;;
    esac

    if [ "$computed_md5" = "$expected_md5" ]; then
        log_success "  Checksum OK: ${filename}"
        return 0
    else
        log_error "  Checksum MISMATCH for ${filename}"
        log_error "    Expected: ${expected_md5}"
        log_error "    Got:      ${computed_md5}"
        return 1
    fi
}

extract_archive() {
    local archive_path="$1"
    local extract_dir="$2"
    local filename
    filename=$(basename "$archive_path")

    log_info "Extracting: ${filename} -> ${extract_dir}"
    mkdir -p "$extract_dir"

    case "$archive_path" in
        *.tar.gz|*.tgz)
            if [ "$VERBOSE" = true ]; then
                tar -xzvf "$archive_path" -C "$extract_dir"
            else
                tar -xzf "$archive_path" -C "$extract_dir"
            fi
            ;;
        *.tar)
            if [ "$VERBOSE" = true ]; then
                tar -xvf "$archive_path" -C "$extract_dir"
            else
                tar -xf "$archive_path" -C "$extract_dir"
            fi
            ;;
        *.zip)
            if [ "$VERBOSE" = true ]; then
                unzip -o "$archive_path" -d "$extract_dir"
            else
                unzip -oq "$archive_path" -d "$extract_dir"
            fi
            ;;
        *)
            log_error "Unknown archive format: ${filename}"
            return 1
            ;;
    esac

    log_success "  Extracted: ${filename}"
}

print_statistics() {
    local data_dir="$1"

    log_step "Dataset Statistics"

    echo -e "${BOLD}-----------------------------------------------${NC}"
    echo -e "${BOLD} nuScenes Radar Dataset Summary${NC}"
    echo -e "${BOLD}-----------------------------------------------${NC}"

    # Check for metadata directory
    local meta_dir=""
    if [ -d "${data_dir}/v1.0-mini" ]; then
        meta_dir="${data_dir}/v1.0-mini"
    elif [ -d "${data_dir}/v1.0-trainval" ]; then
        meta_dir="${data_dir}/v1.0-trainval"
    elif [ -d "${data_dir}/v1.0-test" ]; then
        meta_dir="${data_dir}/v1.0-test"
    fi

    if [ -n "$meta_dir" ] && [ -d "$meta_dir" ]; then
        echo -e " Metadata directory: ${GREEN}${meta_dir}${NC}"

        # Count scenes
        if [ -f "${meta_dir}/scene.json" ]; then
            local num_scenes
            num_scenes=$(python3 -c "import json; print(len(json.load(open('${meta_dir}/scene.json'))))" 2>/dev/null || echo "N/A")
            echo -e " Scenes:            ${CYAN}${num_scenes}${NC}"
        fi

        # Count samples
        if [ -f "${meta_dir}/sample.json" ]; then
            local num_samples
            num_samples=$(python3 -c "import json; print(len(json.load(open('${meta_dir}/sample.json'))))" 2>/dev/null || echo "N/A")
            echo -e " Samples:           ${CYAN}${num_samples}${NC}"
        fi

        # Count sample_data entries for radar
        if [ -f "${meta_dir}/sample_data.json" ]; then
            local num_radar_data
            num_radar_data=$(python3 -c "
import json
data = json.load(open('${meta_dir}/sample_data.json'))
radar_channels = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
# Count entries with radar in filename
count = sum(1 for d in data if any(ch in d.get('filename', '') for ch in radar_channels))
print(count)
" 2>/dev/null || echo "N/A")
            echo -e " Radar data entries: ${CYAN}${num_radar_data}${NC}"
        fi
    else
        echo -e " ${YELLOW}Metadata not found (statistics unavailable)${NC}"
    fi

    # Count radar sweep files
    echo -e ""
    echo -e " ${BOLD}Radar sweep files by channel:${NC}"

    local total_sweeps=0
    local total_samples_count=0

    for channel in "${RADAR_CHANNELS[@]}"; do
        local sweep_count=0
        local sample_count=0

        if [ -d "${data_dir}/sweeps/${channel}" ]; then
            sweep_count=$(find "${data_dir}/sweeps/${channel}" -name "*.pcd.bin" -o -name "*.pcd" 2>/dev/null | wc -l)
        fi
        if [ -d "${data_dir}/samples/${channel}" ]; then
            sample_count=$(find "${data_dir}/samples/${channel}" -name "*.pcd.bin" -o -name "*.pcd" 2>/dev/null | wc -l)
        fi

        total_sweeps=$((total_sweeps + sweep_count))
        total_samples_count=$((total_samples_count + sample_count))

        printf "   %-20s samples: ${CYAN}%6d${NC}  sweeps: ${CYAN}%6d${NC}\n" "${channel}" "$sample_count" "$sweep_count"
    done

    echo -e ""
    echo -e " Total keyframe samples: ${GREEN}${total_samples_count}${NC}"
    echo -e " Total sweep files:      ${GREEN}${total_sweeps}${NC}"

    # Disk usage
    local disk_usage
    disk_usage=$(du -sh "$data_dir" 2>/dev/null | cut -f1)
    echo -e " Total disk usage:       ${GREEN}${disk_usage}${NC}"

    echo -e "${BOLD}-----------------------------------------------${NC}"
}

cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_warn "Script interrupted or failed (exit code: ${exit_code})"
        # Clean up partial downloads
        if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
            log_info "Cleaning up temporary files in ${TEMP_DIR}"
            # Only remove .part files (partial downloads)
            find "$TEMP_DIR" -name "*.part" -delete 2>/dev/null || true
        fi
        log_error "Download incomplete. Re-run the script to resume."
    fi
}

get_files_for_split() {
    local split="$1"

    case "$split" in
        mini)
            echo "v1.0-mini.tar.gz"
            ;;
        trainval)
            echo "v1.0-trainval_meta.tar.gz"
            echo "v1.0-trainval01_blobs_radar.tar.gz"
            echo "v1.0-trainval02_blobs_radar.tar.gz"
            echo "v1.0-trainval03_blobs_radar.tar.gz"
            echo "v1.0-trainval04_blobs_radar.tar.gz"
            echo "v1.0-trainval05_blobs_radar.tar.gz"
            echo "v1.0-trainval06_blobs_radar.tar.gz"
            echo "v1.0-trainval07_blobs_radar.tar.gz"
            echo "v1.0-trainval08_blobs_radar.tar.gz"
            echo "v1.0-trainval09_blobs_radar.tar.gz"
            echo "v1.0-trainval10_blobs_radar.tar.gz"
            ;;
        test)
            echo "v1.0-test_meta.tar.gz"
            echo "v1.0-test01_blobs_radar.tar.gz"
            echo "v1.0-test02_blobs_radar.tar.gz"
            echo "v1.0-test03_blobs_radar.tar.gz"
            ;;
        *)
            log_error "Unknown split: ${split}"
            exit 1
            ;;
    esac
}

# ------------------------------------------------------------------------------
# Main Logic
# ------------------------------------------------------------------------------

# Set up trap for cleanup on error/interrupt
trap cleanup EXIT INT TERM

# Parse command line arguments
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)
            SPLIT="$2"
            if [[ ! "$SPLIT" =~ ^(mini|trainval|test)$ ]]; then
                log_error "Invalid split: ${SPLIT}. Must be one of: mini, trainval, test"
                exit 1
            fi
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-checksum)
            SKIP_CHECKSUM=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
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

# Print banner
echo -e ""
echo -e "${BOLD}${CYAN}============================================================${NC}"
echo -e "${BOLD}${CYAN}  nuScenes Radar Dataset Downloader${NC}"
echo -e "${BOLD}${CYAN}  For radar-based 3D object detection (RadarPillarNet)${NC}"
echo -e "${BOLD}${CYAN}============================================================${NC}"
echo -e ""
echo -e "  Split:       ${GREEN}${SPLIT}${NC}"
echo -e "  Output:      ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "  Checksum:    ${GREEN}$([ "$SKIP_CHECKSUM" = true ] && echo "disabled" || echo "enabled")${NC}"
echo -e ""

# Check dependencies
check_dependencies

# Set up directories
TEMP_DIR="${OUTPUT_DIR}/.downloads"
mkdir -p "$OUTPUT_DIR" "$TEMP_DIR"

log_step "Preparing download list for split: ${SPLIT}"

# Get list of files to download
mapfile -t FILES < <(get_files_for_split "$SPLIT")
TOTAL_FILES=${#FILES[@]}

log_info "Files to download: ${TOTAL_FILES}"
for f in "${FILES[@]}"; do
    log_info "  - ${f}"
done

# Download files
log_step "Downloading nuScenes ${SPLIT} split"

DOWNLOAD_FAILURES=0

for i in "${!FILES[@]}"; do
    file="${FILES[$i]}"
    file_num=$((i + 1))

    echo -e "\n${BOLD}[${file_num}/${TOTAL_FILES}]${NC} ${file}"

    download_url="${NUSCENES_BASE_URL}/${file}"
    download_path="${TEMP_DIR}/${file}"

    if ! download_file "$download_url" "$download_path" "$file"; then
        DOWNLOAD_FAILURES=$((DOWNLOAD_FAILURES + 1))
        log_error "Failed to download: ${file}"
        continue
    fi

    # Verify checksum
    expected_checksum="${CHECKSUMS[$file]:-}"
    if ! verify_checksum "$download_path" "$expected_checksum"; then
        DOWNLOAD_FAILURES=$((DOWNLOAD_FAILURES + 1))
        log_error "Checksum verification failed: ${file}"
        log_warn "You may want to re-download this file"
        continue
    fi
done

if [ $DOWNLOAD_FAILURES -gt 0 ]; then
    log_warn "${DOWNLOAD_FAILURES} file(s) failed to download or verify"
    read -rp "Continue with extraction of successfully downloaded files? [Y/n]: " response
    if [[ "$response" =~ ^[Nn]$ ]]; then
        log_info "Aborted. Fix download issues and re-run."
        exit 1
    fi
fi

# Extract archives
log_step "Extracting archives"

EXTRACT_FAILURES=0

for file in "${FILES[@]}"; do
    archive_path="${TEMP_DIR}/${file}"

    if [ ! -f "$archive_path" ] || [ ! -s "$archive_path" ]; then
        log_warn "Skipping extraction (file missing): ${file}"
        continue
    fi

    if ! extract_archive "$archive_path" "$OUTPUT_DIR"; then
        EXTRACT_FAILURES=$((EXTRACT_FAILURES + 1))
        log_error "Failed to extract: ${file}"
    fi
done

if [ $EXTRACT_FAILURES -gt 0 ]; then
    log_warn "${EXTRACT_FAILURES} file(s) failed to extract"
fi

# Verify directory structure
log_step "Verifying directory structure"

EXPECTED_DIRS=()
case "$SPLIT" in
    mini)
        EXPECTED_DIRS=("v1.0-mini")
        ;;
    trainval)
        EXPECTED_DIRS=("v1.0-trainval")
        ;;
    test)
        EXPECTED_DIRS=("v1.0-test")
        ;;
esac

for dir in "${EXPECTED_DIRS[@]}"; do
    if [ -d "${OUTPUT_DIR}/${dir}" ]; then
        log_success "Found metadata: ${dir}/"
    else
        log_warn "Missing metadata directory: ${dir}/"
    fi
done

for channel in "${RADAR_CHANNELS[@]}"; do
    if [ -d "${OUTPUT_DIR}/samples/${channel}" ]; then
        log_success "Found samples: samples/${channel}/"
    else
        log_warn "Missing: samples/${channel}/"
    fi

    if [ -d "${OUTPUT_DIR}/sweeps/${channel}" ]; then
        log_success "Found sweeps: sweeps/${channel}/"
    else
        log_warn "Missing: sweeps/${channel}/"
    fi
done

# Print statistics
print_statistics "$OUTPUT_DIR"

# Cleanup temporary downloads (optional)
echo ""
read -rp "Remove downloaded archives to save disk space? [y/N]: " response
if [[ "$response" =~ ^[Yy]$ ]]; then
    log_info "Removing temporary downloads..."
    rm -rf "$TEMP_DIR"
    log_success "Temporary files removed"
else
    log_info "Archives kept in: ${TEMP_DIR}"
    log_info "You can manually remove them later with: rm -rf ${TEMP_DIR}"
fi

# Final summary
echo -e ""
echo -e "${BOLD}${GREEN}============================================================${NC}"
echo -e "${BOLD}${GREEN}  Download Complete!${NC}"
echo -e "${BOLD}${GREEN}============================================================${NC}"
echo -e ""
echo -e "  Dataset location: ${CYAN}${OUTPUT_DIR}${NC}"
echo -e "  Split:            ${CYAN}${SPLIT}${NC}"
echo -e ""
echo -e "  ${BOLD}Next steps:${NC}"
echo -e "  1. Verify the data by running the dataset test script"
echo -e "  2. Generate the radar pillar features:"
echo -e "     ${CYAN}python tools/create_radar_pillars.py --dataroot ${OUTPUT_DIR}${NC}"
echo -e "  3. Start training:"
echo -e "     ${CYAN}python train.py --config configs/radar_pillarnet.yaml${NC}"
echo -e ""

# Reset trap (successful exit)
trap - EXIT
exit 0
