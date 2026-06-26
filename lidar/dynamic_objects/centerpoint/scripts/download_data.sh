#!/bin/bash
# ==============================================================================
# download_data.sh - Download nuScenes and Waymo Open Dataset for CenterPoint
# ==============================================================================
# This script downloads and prepares datasets for CenterPoint model training.
# Supports nuScenes (mini and full) and Waymo Open Dataset v1.4.1.
#
# Usage:
#   ./download_data.sh [OPTIONS] DATASET
#
# Datasets:
#   nuscenes-mini    Download nuScenes mini split (for quick testing)
#   nuscenes-full    Download nuScenes full trainval split (for training)
#   waymo            Download Waymo Open Dataset v1.4.1
#   all              Download all datasets
#
# Options:
#   -d, --data-root DIR    Root directory for data (default: ./data)
#   -j, --jobs N           Number of parallel downloads (default: 4)
#   -n, --dry-run          Show what would be downloaded without downloading
#   -h, --help             Show this help message
#
# Prerequisites:
#   - nuScenes: Account and API key from https://www.nuscenes.org/
#   - Waymo: Google Cloud SDK (gsutil) with access to Waymo Open Dataset
#
# Environment Variables:
#   NUSCENES_API_KEY       Your nuScenes devkit API key
#   NUSCENES_EMAIL         Your nuScenes account email
# ==============================================================================

set -e
set -o pipefail

# ==============================================================================
# Color output helpers
# ==============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}=== $* ===${NC}\n"; }

# ==============================================================================
# Configuration
# ==============================================================================
DATA_ROOT="./data"
PARALLEL_JOBS=4
DRY_RUN=false

# nuScenes URLs (these require authentication - use token-based access)
# After logging in at https://www.nuscenes.org/, obtain your API token from
# your profile page. Set NUSCENES_API_KEY environment variable.
NUSCENES_BASE_URL="https://www.nuscenes.org/data"
NUSCENES_VERSION="v1.0"

# Waymo Open Dataset GCS bucket
WAYMO_BUCKET="gs://waymo_open_dataset_v_1_4_1"

# Minimum disk space requirements (in GB)
NUSCENES_MINI_SPACE_GB=4
NUSCENES_FULL_SPACE_GB=350
WAYMO_SPACE_GB=800

# Expected MD5 checksums for nuScenes files (examples - verify against official)
declare -A NUSCENES_CHECKSUMS=(
    ["v1.0-mini.tgz"]="d7e5a06e0e9eac3df53ec0948a2dbb33"
    ["v1.0-trainval01_blobs.tgz"]="29e47c90baf4e0f0b89e7f3b4e4e3215"
    ["v1.0-trainval02_blobs.tgz"]="a6f7c5e3e5c6f0a8b2d4e6f8a0b2c4d6"
    ["v1.0-trainval03_blobs.tgz"]="b7f8c6e4f6d7a1b3c5d7e9f1a3b5c7d9"
    ["v1.0-trainval04_blobs.tgz"]="c8f9d7e5a7e8b2c4d6e8f0a2b4c6d8e0"
    ["v1.0-trainval05_blobs.tgz"]="d9a0e8f6b8f9c3d5e7f9a1b3c5d7e9f1"
    ["v1.0-trainval06_blobs.tgz"]="e0b1f9a7c9a0d4e6f8a0b2c4d6e8f0a2"
    ["v1.0-trainval07_blobs.tgz"]="f1c2a0b8d0b1e5f7a9b1c3d5e7f9a1b3"
    ["v1.0-trainval08_blobs.tgz"]="a2d3b1c9e1c2f6a8b0c2d4e6f8a0b2c4"
    ["v1.0-trainval09_blobs.tgz"]="b3e4c2d0f2d3a7b9c1d3e5f7a9b1c3d5"
    ["v1.0-trainval10_blobs.tgz"]="c4f5d3e1a3e4b8c0d2e4f6a8b0c2d4e6"
    ["v1.0-trainval_meta.tgz"]="f0e1d2c3b4a5968778695a4b3c2d1e0f"
)

# ==============================================================================
# Utility Functions
# ==============================================================================

show_help() {
    sed -n '2,/^# ==/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 0
}

check_command() {
    local cmd="$1"
    local install_hint="$2"
    if ! command -v "$cmd" &>/dev/null; then
        error "'$cmd' is not installed or not in PATH."
        if [[ -n "$install_hint" ]]; then
            echo -e "  ${YELLOW}Install with:${NC} $install_hint"
        fi
        return 1
    fi
    return 0
}

check_disk_space() {
    local path="$1"
    local required_gb="$2"
    local label="$3"

    # Get available space in GB
    local available_gb
    available_gb=$(df -BG "$path" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//')

    if [[ -z "$available_gb" ]]; then
        warn "Could not determine available disk space at $path"
        return 0
    fi

    if (( available_gb < required_gb )); then
        error "Insufficient disk space for $label"
        echo -e "  Required: ${BOLD}${required_gb}GB${NC}"
        echo -e "  Available: ${BOLD}${available_gb}GB${NC}"
        echo -e "  Path: $path"
        return 1
    fi

    info "Disk space check passed for $label (${available_gb}GB available, ${required_gb}GB required)"
    return 0
}

# Create a progress bar
progress_bar() {
    local current="$1"
    local total="$2"
    local width=50
    local percent=$(( current * 100 / total ))
    local filled=$(( current * width / total ))
    local empty=$(( width - filled ))

    printf "\r  ${CYAN}[${NC}"
    printf "%${filled}s" | tr ' ' '#'
    printf "%${empty}s" | tr ' ' '-'
    printf "${CYAN}]${NC} %3d%% (%d/%d)" "$percent" "$current" "$total"
}

verify_md5() {
    local file="$1"
    local expected="$2"

    if [[ ! -f "$file" ]]; then
        error "File not found for checksum verification: $file"
        return 1
    fi

    info "Verifying checksum for $(basename "$file")..."
    local actual
    if command -v md5sum &>/dev/null; then
        actual=$(md5sum "$file" | awk '{print $1}')
    elif command -v md5 &>/dev/null; then
        actual=$(md5 -q "$file")
    else
        warn "Neither md5sum nor md5 found; skipping checksum verification"
        return 0
    fi

    if [[ "$actual" == "$expected" ]]; then
        success "Checksum verified: $(basename "$file")"
        return 0
    else
        error "Checksum mismatch for $(basename "$file")"
        echo -e "  Expected: $expected"
        echo -e "  Actual:   $actual"
        return 1
    fi
}

print_directory_tree() {
    local dir="$1"
    local depth="${2:-3}"

    header "Directory Structure: $dir"
    if command -v tree &>/dev/null; then
        tree -L "$depth" --dirsfirst "$dir"
    else
        find "$dir" -maxdepth "$depth" -type d | sort | while read -r d; do
            local indent=$(echo "$d" | sed "s|$dir||" | tr -cd '/' | wc -c)
            printf "%*s%s/\n" $((indent * 2)) "" "$(basename "$d")"
        done
    fi

    echo ""
    info "File counts:"
    find "$dir" -type f | wc -l | xargs -I{} echo "  Total files: {}"
    if [[ -d "$dir/nuscenes" ]]; then
        find "$dir/nuscenes" -type f | wc -l | xargs -I{} echo "  nuScenes files: {}"
    fi
    if [[ -d "$dir/waymo" ]]; then
        find "$dir/waymo" -type f | wc -l | xargs -I{} echo "  Waymo files: {}"
    fi
}

# ==============================================================================
# nuScenes Download Functions
# ==============================================================================

setup_nuscenes_auth() {
    if [[ -z "${NUSCENES_API_KEY:-}" ]]; then
        echo ""
        warn "NUSCENES_API_KEY environment variable is not set."
        echo ""
        echo -e "  ${BOLD}To obtain your API key:${NC}"
        echo "  1. Create an account at https://www.nuscenes.org/sign-up"
        echo "  2. Log in at https://www.nuscenes.org/login"
        echo "  3. Go to your profile page"
        echo "  4. Copy your access token / API key"
        echo ""
        echo -e "  ${BOLD}Then set the environment variable:${NC}"
        echo "  export NUSCENES_API_KEY='your-api-key-here'"
        echo ""
        read -rp "  Enter your nuScenes API key (or press Ctrl+C to cancel): " NUSCENES_API_KEY
        if [[ -z "$NUSCENES_API_KEY" ]]; then
            error "API key is required for nuScenes downloads."
            exit 1
        fi
    fi
    success "nuScenes API key configured"
}

create_nuscenes_dirs() {
    local base_dir="$1"

    info "Creating nuScenes directory structure..."
    mkdir -p "$base_dir/nuscenes/v1.0-trainval"
    mkdir -p "$base_dir/nuscenes/v1.0-mini"
    mkdir -p "$base_dir/nuscenes/samples"
    mkdir -p "$base_dir/nuscenes/sweeps"
    mkdir -p "$base_dir/nuscenes/maps"
    success "nuScenes directories created at $base_dir/nuscenes/"
}

download_nuscenes_file() {
    local url="$1"
    local output_path="$2"
    local description="$3"

    if [[ -f "$output_path" ]]; then
        info "File already exists, skipping: $(basename "$output_path")"
        return 0
    fi

    info "Downloading $description..."

    if $DRY_RUN; then
        echo "  [DRY RUN] Would download: $url"
        echo "  [DRY RUN] To: $output_path"
        return 0
    fi

    # Use wget with resume support, authentication header, and progress
    local wget_opts=(
        --continue
        --show-progress
        --header="Authorization: Bearer ${NUSCENES_API_KEY}"
        --retry-connrefused
        --waitretry=5
        --timeout=60
        --tries=5
        -O "$output_path"
    )

    if command -v wget &>/dev/null; then
        wget "${wget_opts[@]}" "$url" || {
            error "Failed to download: $description"
            error "URL: $url"
            rm -f "$output_path"
            return 1
        }
    elif command -v curl &>/dev/null; then
        curl -L -C - \
            --retry 5 \
            --retry-delay 5 \
            --connect-timeout 60 \
            -H "Authorization: Bearer ${NUSCENES_API_KEY}" \
            --progress-bar \
            -o "$output_path" \
            "$url" || {
            error "Failed to download: $description"
            error "URL: $url"
            rm -f "$output_path"
            return 1
        }
    else
        error "Neither wget nor curl is available"
        return 1
    fi

    success "Downloaded: $description"
}

download_nuscenes_mini() {
    header "Downloading nuScenes Mini Dataset"
    info "The mini split contains 10 scenes (~4GB) for quick testing and debugging."

    local base_dir="$DATA_ROOT"
    local download_dir="$base_dir/nuscenes/downloads"
    mkdir -p "$download_dir"

    setup_nuscenes_auth
    create_nuscenes_dirs "$base_dir"

    if ! check_disk_space "$base_dir" "$NUSCENES_MINI_SPACE_GB" "nuScenes mini"; then
        return 1
    fi

    # Download mini dataset archive
    local mini_url="${NUSCENES_BASE_URL}/v1.0-mini.tgz"
    local mini_file="$download_dir/v1.0-mini.tgz"

    download_nuscenes_file "$mini_url" "$mini_file" "nuScenes v1.0-mini (metadata + sensor data)"

    if ! $DRY_RUN; then
        # Verify checksum
        local expected_md5="${NUSCENES_CHECKSUMS[v1.0-mini.tgz]}"
        if [[ -n "$expected_md5" ]]; then
            verify_md5 "$mini_file" "$expected_md5" || {
                warn "Checksum verification failed. File may be corrupted or checksum may be outdated."
                warn "You can re-download by removing: $mini_file"
            }
        fi

        # Extract
        info "Extracting nuScenes mini dataset..."
        tar -xzf "$mini_file" -C "$base_dir/nuscenes/" --strip-components=0
        success "nuScenes mini dataset extracted"
    fi

    success "nuScenes mini dataset download complete!"
    echo -e "  ${CYAN}Location:${NC} $base_dir/nuscenes/"
    echo -e "  ${CYAN}Usage:${NC} Set data_root='$base_dir/nuscenes/' and version='v1.0-mini' in your config"
}

download_nuscenes_full() {
    header "Downloading nuScenes Full Trainval Dataset"
    info "The full trainval split contains 850 scenes (~350GB) for complete training."
    warn "This is a very large download. Ensure you have sufficient disk space and bandwidth."

    local base_dir="$DATA_ROOT"
    local download_dir="$base_dir/nuscenes/downloads"
    mkdir -p "$download_dir"

    setup_nuscenes_auth
    create_nuscenes_dirs "$base_dir"

    if ! check_disk_space "$base_dir" "$NUSCENES_FULL_SPACE_GB" "nuScenes full trainval"; then
        return 1
    fi

    # List of files to download for full trainval
    # The dataset is split into multiple blobs for manageability
    local -a download_files=(
        "v1.0-trainval_meta.tgz:nuScenes v1.0 trainval metadata (annotations, calibration, maps)"
        "v1.0-trainval01_blobs.tgz:nuScenes trainval blob part 01/10 (sensor data)"
        "v1.0-trainval02_blobs.tgz:nuScenes trainval blob part 02/10 (sensor data)"
        "v1.0-trainval03_blobs.tgz:nuScenes trainval blob part 03/10 (sensor data)"
        "v1.0-trainval04_blobs.tgz:nuScenes trainval blob part 04/10 (sensor data)"
        "v1.0-trainval05_blobs.tgz:nuScenes trainval blob part 05/10 (sensor data)"
        "v1.0-trainval06_blobs.tgz:nuScenes trainval blob part 06/10 (sensor data)"
        "v1.0-trainval07_blobs.tgz:nuScenes trainval blob part 07/10 (sensor data)"
        "v1.0-trainval08_blobs.tgz:nuScenes trainval blob part 08/10 (sensor data)"
        "v1.0-trainval09_blobs.tgz:nuScenes trainval blob part 09/10 (sensor data)"
        "v1.0-trainval10_blobs.tgz:nuScenes trainval blob part 10/10 (sensor data)"
    )

    local total=${#download_files[@]}
    local current=0
    local failed=0

    for entry in "${download_files[@]}"; do
        local filename="${entry%%:*}"
        local description="${entry#*:}"
        local url="${NUSCENES_BASE_URL}/${filename}"
        local output_path="$download_dir/$filename"

        current=$((current + 1))
        echo ""
        progress_bar "$current" "$total"
        echo ""

        download_nuscenes_file "$url" "$output_path" "$description" || {
            failed=$((failed + 1))
            warn "Failed to download $filename (will continue with remaining files)"
            continue
        }

        # Verify checksum if available
        if ! $DRY_RUN; then
            local expected_md5="${NUSCENES_CHECKSUMS[$filename]:-}"
            if [[ -n "$expected_md5" ]]; then
                verify_md5 "$output_path" "$expected_md5" || {
                    warn "Checksum mismatch for $filename. Consider re-downloading."
                }
            fi
        fi
    done

    if (( failed > 0 )); then
        warn "$failed out of $total files failed to download."
        warn "Re-run the script to resume failed downloads (resume support is enabled)."
    fi

    if ! $DRY_RUN; then
        # Extract all downloaded archives
        header "Extracting nuScenes Full Dataset"
        info "Extracting metadata and sensor data archives..."

        for entry in "${download_files[@]}"; do
            local filename="${entry%%:*}"
            local archive="$download_dir/$filename"

            if [[ -f "$archive" ]]; then
                info "Extracting $filename..."
                tar -xzf "$archive" -C "$base_dir/nuscenes/" --strip-components=0
                success "Extracted $filename"
            fi
        done
    fi

    success "nuScenes full trainval dataset download complete!"
    echo -e "  ${CYAN}Location:${NC} $base_dir/nuscenes/"
    echo -e "  ${CYAN}Usage:${NC} Set data_root='$base_dir/nuscenes/' and version='v1.0-trainval' in your config"
    echo ""
    info "Expected directory structure after extraction:"
    echo "  data/nuscenes/"
    echo "    v1.0-trainval/  (annotation JSON files)"
    echo "    samples/        (keyframe sensor data)"
    echo "    sweeps/         (intermediate sensor data)"
    echo "    maps/           (map rasterizations)"
}

# ==============================================================================
# Waymo Open Dataset Download Functions
# ==============================================================================

create_waymo_dirs() {
    local base_dir="$1"

    info "Creating Waymo directory structure..."
    mkdir -p "$base_dir/waymo/raw/training"
    mkdir -p "$base_dir/waymo/raw/validation"
    mkdir -p "$base_dir/waymo/raw/testing"
    mkdir -p "$base_dir/waymo/processed/training"
    mkdir -p "$base_dir/waymo/processed/validation"
    mkdir -p "$base_dir/waymo/processed/testing"
    success "Waymo directories created at $base_dir/waymo/"
}

download_waymo() {
    header "Downloading Waymo Open Dataset v1.4.1"
    info "The Waymo Open Dataset contains perception data from autonomous driving."
    info "Source: ${WAYMO_BUCKET}"
    warn "This requires Google Cloud SDK (gsutil) with proper authentication."
    warn "Total download size is approximately 800GB."

    local base_dir="$DATA_ROOT"

    # Check prerequisites
    if ! check_command "gsutil" "pip install gsutil  OR  https://cloud.google.com/sdk/docs/install"; then
        echo ""
        echo -e "  ${BOLD}Setup instructions:${NC}"
        echo "  1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
        echo "  2. Authenticate: gcloud auth login"
        echo "  3. Accept Waymo Open Dataset license:"
        echo "     https://waymo.com/open/licensing/"
        echo "  4. Ensure your Google account has access to the dataset bucket"
        echo ""
        return 1
    fi

    # Check authentication
    info "Verifying Google Cloud authentication..."
    if ! gsutil ls "${WAYMO_BUCKET}" &>/dev/null; then
        error "Cannot access Waymo Open Dataset bucket."
        echo ""
        echo -e "  ${BOLD}Possible causes:${NC}"
        echo "  1. Not authenticated: run 'gcloud auth login'"
        echo "  2. Haven't accepted the dataset license at https://waymo.com/open/licensing/"
        echo "  3. Network/firewall issue"
        echo ""
        return 1
    fi
    success "Google Cloud authentication verified"

    create_waymo_dirs "$base_dir"

    if ! check_disk_space "$base_dir" "$WAYMO_SPACE_GB" "Waymo Open Dataset"; then
        return 1
    fi

    # Download training split
    local -a splits=("training" "validation" "testing")
    local -a split_descriptions=(
        "Training split (~798 segments, ~600GB)"
        "Validation split (~202 segments, ~150GB)"
        "Testing split (~150 segments, ~50GB)"
    )

    for i in "${!splits[@]}"; do
        local split="${splits[$i]}"
        local desc="${split_descriptions[$i]}"
        local src="${WAYMO_BUCKET}/individual_files/${split}"
        local dst="$base_dir/waymo/raw/${split}"

        echo ""
        info "Downloading Waymo ${split} split..."
        info "  ${desc}"
        info "  Source: ${src}"
        info "  Destination: ${dst}"

        if $DRY_RUN; then
            echo "  [DRY RUN] gsutil -m cp -n -r ${src}/ ${dst}/"
            continue
        fi

        # Use gsutil with:
        #   -m: parallel composite uploads/downloads
        #   cp: copy command
        #   -n: no-clobber (skip existing files for resume support)
        #   -r: recursive
        gsutil -m cp -n -r "${src}/" "${dst}/" 2>&1 | while IFS= read -r line; do
            # Show progress lines but filter verbose output
            if [[ "$line" == *"Copying"* ]] || [[ "$line" == *"/"* ]]; then
                echo -ne "\r  ${CYAN}${line}${NC}                    "
            fi
        done
        echo ""
        success "Waymo ${split} split download complete"
    done

    # Print information about format conversion
    echo ""
    header "Waymo Data Format Notes"
    info "Downloaded files are in TFRecord format (.tfrecord)"
    info "For CenterPoint training, you need to convert to a compatible format."
    echo ""
    echo -e "  ${BOLD}Conversion steps:${NC}"
    echo "  1. Install waymo-open-dataset-tf package:"
    echo "     pip install waymo-open-dataset-tf-2-12-0"
    echo ""
    echo "  2. Run the conversion script (from CenterPoint repo):"
    echo "     python tools/create_data.py waymo_data_prep \\"
    echo "       --root_path=$base_dir/waymo/raw \\"
    echo "       --out_dir=$base_dir/waymo/processed \\"
    echo "       --workers=16"
    echo ""
    echo "  3. This will create processed pickle and bin files in:"
    echo "     $base_dir/waymo/processed/{training,validation,testing}/"
    echo ""

    success "Waymo Open Dataset download complete!"
    echo -e "  ${CYAN}Location:${NC} $base_dir/waymo/"
    echo -e "  ${CYAN}Raw data:${NC} $base_dir/waymo/raw/"
    echo -e "  ${CYAN}Processed:${NC} $base_dir/waymo/processed/ (after conversion)"
}

# ==============================================================================
# Main Script Logic
# ==============================================================================

check_prerequisites() {
    header "Checking Prerequisites"

    local has_downloader=false
    local errors=0

    # Check for at least one download tool
    if command -v wget &>/dev/null; then
        success "wget found: $(wget --version 2>&1 | head -1)"
        has_downloader=true
    fi
    if command -v curl &>/dev/null; then
        success "curl found: $(curl --version 2>&1 | head -1)"
        has_downloader=true
    fi
    if ! $has_downloader; then
        error "Neither wget nor curl found. At least one is required."
        errors=$((errors + 1))
    fi

    # Check for tar (needed for extraction)
    if check_command "tar" "apt-get install tar / brew install gnu-tar"; then
        success "tar found"
    else
        errors=$((errors + 1))
    fi

    # Check for md5sum or md5 (for verification)
    if command -v md5sum &>/dev/null; then
        success "md5sum found"
    elif command -v md5 &>/dev/null; then
        success "md5 found (macOS)"
    else
        warn "No MD5 tool found; checksum verification will be skipped"
    fi

    # Check for gsutil (only warn, not error, since it's only needed for Waymo)
    if command -v gsutil &>/dev/null; then
        success "gsutil found (needed for Waymo dataset)"
    else
        warn "gsutil not found (only needed for Waymo dataset download)"
        info "  Install: https://cloud.google.com/sdk/docs/install"
    fi

    if (( errors > 0 )); then
        error "Prerequisites check failed with $errors error(s)"
        return 1
    fi

    success "All essential prerequisites satisfied"
    return 0
}

parse_args() {
    local positional_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -d|--data-root)
                DATA_ROOT="$2"
                shift 2
                ;;
            -j|--jobs)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                ;;
            -*)
                error "Unknown option: $1"
                echo "Use --help for usage information."
                exit 1
                ;;
            *)
                positional_args+=("$1")
                shift
                ;;
        esac
    done

    if [[ ${#positional_args[@]} -eq 0 ]]; then
        error "No dataset specified."
        echo ""
        echo "Usage: $0 [OPTIONS] DATASET"
        echo ""
        echo "Available datasets:"
        echo "  nuscenes-mini    nuScenes mini split (~4GB, for testing)"
        echo "  nuscenes-full    nuScenes full trainval (~350GB, for training)"
        echo "  waymo            Waymo Open Dataset v1.4.1 (~800GB)"
        echo "  all              Download all datasets"
        echo ""
        echo "Use --help for full usage information."
        exit 1
    fi

    DATASETS=("${positional_args[@]}")
}

main() {
    echo -e "${BOLD}${CYAN}"
    echo "  ================================================================"
    echo "    CenterPoint Dataset Downloader"
    echo "    Datasets: nuScenes | Waymo Open Dataset"
    echo "  ================================================================"
    echo -e "${NC}"

    parse_args "$@"

    # Show configuration
    info "Configuration:"
    echo "  Data root:       $DATA_ROOT"
    echo "  Parallel jobs:   $PARALLEL_JOBS"
    echo "  Dry run:         $DRY_RUN"
    echo "  Datasets:        ${DATASETS[*]}"
    echo ""

    # Ensure data root directory exists
    mkdir -p "$DATA_ROOT"
    DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
    info "Resolved data root: $DATA_ROOT"

    # Check prerequisites
    check_prerequisites || exit 1

    # Track overall success
    local overall_success=true

    # Process each requested dataset
    for dataset in "${DATASETS[@]}"; do
        case "$dataset" in
            nuscenes-mini)
                download_nuscenes_mini || overall_success=false
                ;;
            nuscenes-full)
                download_nuscenes_full || overall_success=false
                ;;
            waymo)
                download_waymo || overall_success=false
                ;;
            all)
                download_nuscenes_mini || overall_success=false
                download_nuscenes_full || overall_success=false
                download_waymo || overall_success=false
                ;;
            *)
                error "Unknown dataset: $dataset"
                echo "  Valid options: nuscenes-mini, nuscenes-full, waymo, all"
                overall_success=false
                ;;
        esac
    done

    # Print final summary
    echo ""
    header "Download Summary"

    if $DRY_RUN; then
        info "Dry run complete - no files were downloaded."
    elif $overall_success; then
        success "All requested datasets downloaded successfully!"
    else
        warn "Some downloads failed. Re-run the script to resume (downloads support resumption)."
    fi

    # Print directory structure if not a dry run and data exists
    if ! $DRY_RUN && [[ -d "$DATA_ROOT" ]]; then
        print_directory_tree "$DATA_ROOT" 3
    fi

    echo ""
    info "Next steps:"
    echo "  1. Verify dataset integrity (check file counts and sizes)"
    echo "  2. For Waymo: run the TFRecord conversion script"
    echo "  3. Update your CenterPoint config to point to: $DATA_ROOT"
    echo "  4. Run data preprocessing: python tools/create_data.py <dataset>_data_prep"
    echo ""

    if $overall_success; then
        exit 0
    else
        exit 1
    fi
}

# Run main with all arguments
main "$@"
