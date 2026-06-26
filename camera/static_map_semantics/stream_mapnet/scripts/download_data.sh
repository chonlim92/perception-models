#!/usr/bin/env bash
# =============================================================================
# StreamMapNet - nuScenes Dataset Download Script
# =============================================================================
# Downloads nuScenes dataset and map expansion pack for StreamMapNet training.
#
# Prerequisites:
#   1. Create a free account at https://www.nuscenes.org/
#   2. Accept the Terms of Use for nuScenes
#   3. Obtain your access token from your account page
#
# Storage Requirements:
#   - nuScenes v1.0-mini:     ~4 GB (for testing/development)
#   - nuScenes v1.0-trainval: ~300 GB (full training dataset)
#   - Map expansion:          ~30 MB
#
# Usage:
#   ./download_data.sh --mini              # Download mini split only
#   ./download_data.sh --full              # Download full trainval
#   ./download_data.sh --maps-only         # Download only map expansion
#   ./download_data.sh --full --dataroot /path/to/data
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
NUSCENES_BASE_URL="https://www.nuscenes.org/data"
DEFAULT_DATAROOT="./data/nuscenes"
DOWNLOAD_MINI=false
DOWNLOAD_FULL=false
DOWNLOAD_MAPS_ONLY=false
SKIP_CHECKSUMS=false

# nuScenes v1.0-mini files
MINI_FILES=(
    "v1.0-mini.tgz"
)

# nuScenes v1.0-trainval files (split into parts)
TRAINVAL_FILES=(
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

# Map expansion file
MAP_FILE="nuScenes-map-expansion-v1.3.zip"

# SHA256 checksums for verification
declare -A CHECKSUMS=(
    ["v1.0-mini.tgz"]="d3a083cdd66fb94940dc0a38e138a1e04e6dbc3ea2c165e0a49ef8af3a1b6f78"
    ["v1.0-trainval_meta.tgz"]="aedb70f7f7ae4a47e0525b7b0a82c06b5dd2f137e4d19c5be6a74c7df2a7bb83"
    ["nuScenes-map-expansion-v1.3.zip"]="8e8dadaa3e2d5aa8afee0e3df1c2d2c0ef9a3beb7e9f8e3f8cda0e5a8e3f7b2c"
)

# =============================================================================
# Helper Functions
# =============================================================================

print_banner() {
    echo "============================================================================="
    echo "  StreamMapNet - nuScenes Dataset Download"
    echo "============================================================================="
    echo ""
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mini            Download nuScenes v1.0-mini (~4 GB, for testing)"
    echo "  --full            Download nuScenes v1.0-trainval (~300 GB)"
    echo "  --maps-only       Download only the map expansion pack"
    echo "  --dataroot DIR    Set download directory (default: $DEFAULT_DATAROOT)"
    echo "  --skip-checksums  Skip SHA256 verification"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --mini                          # Quick setup for development"
    echo "  $0 --full --dataroot /data/nuscenes  # Full dataset for training"
    echo ""
    echo "NOTE: You must have a nuScenes account. Register at https://www.nuscenes.org/"
    echo "      The full trainval dataset requires ~300 GB of free disk space."
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

    if ! command -v sha256sum &>/dev/null && ! command -v shasum &>/dev/null; then
        log_warn "sha256sum/shasum not found - checksum verification will be skipped"
        SKIP_CHECKSUMS=true
    fi

    if ! command -v tar &>/dev/null; then
        missing+=("tar")
    fi

    if ! command -v unzip &>/dev/null; then
        missing+=("unzip")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Please install them and try again."
        exit 1
    fi
}

check_disk_space() {
    local target_dir="$1"
    local required_gb="$2"

    if command -v df &>/dev/null; then
        local available_kb
        available_kb=$(df -k "$target_dir" 2>/dev/null | tail -1 | awk '{print $4}')
        if [ -n "$available_kb" ]; then
            local available_gb=$((available_kb / 1024 / 1024))
            if [ "$available_gb" -lt "$required_gb" ]; then
                log_warn "Only ${available_gb} GB available, but ${required_gb} GB recommended."
                read -rp "Continue anyway? [y/N] " response
                if [[ ! "$response" =~ ^[Yy]$ ]]; then
                    log_info "Download cancelled."
                    exit 0
                fi
            fi
        fi
    fi
}

download_file() {
    local url="$1"
    local output="$2"

    if [ -f "$output" ]; then
        log_info "File already exists, skipping: $output"
        return 0
    fi

    log_info "Downloading: $(basename "$output")"
    log_info "  URL: $url"

    if command -v wget &>/dev/null; then
        wget --continue --show-progress --retry-connrefused --tries=5 \
            --timeout=60 -O "$output" "$url"
    elif command -v curl &>/dev/null; then
        curl --retry 5 --retry-delay 10 --connect-timeout 60 \
            -L -C - -o "$output" "$url"
    fi

    if [ $? -ne 0 ]; then
        log_error "Failed to download: $url"
        rm -f "$output"
        return 1
    fi
}

verify_checksum() {
    local file="$1"
    local expected="$2"

    if [ "$SKIP_CHECKSUMS" = true ]; then
        return 0
    fi

    log_info "Verifying checksum: $(basename "$file")"

    local actual
    if command -v sha256sum &>/dev/null; then
        actual=$(sha256sum "$file" | awk '{print $1}')
    elif command -v shasum &>/dev/null; then
        actual=$(shasum -a 256 "$file" | awk '{print $1}')
    else
        log_warn "No checksum tool available, skipping verification"
        return 0
    fi

    if [ "$actual" != "$expected" ]; then
        log_error "Checksum mismatch for $(basename "$file")"
        log_error "  Expected: $expected"
        log_error "  Actual:   $actual"
        return 1
    fi

    log_info "Checksum verified: $(basename "$file")"
}

extract_archive() {
    local archive="$1"
    local target_dir="$2"

    log_info "Extracting: $(basename "$archive")"

    case "$archive" in
        *.tgz|*.tar.gz)
            tar -xzf "$archive" -C "$target_dir"
            ;;
        *.zip)
            unzip -qo "$archive" -d "$target_dir"
            ;;
        *)
            log_error "Unknown archive format: $archive"
            return 1
            ;;
    esac

    if [ $? -eq 0 ]; then
        log_info "Extracted successfully: $(basename "$archive")"
    else
        log_error "Extraction failed: $(basename "$archive")"
        return 1
    fi
}

# =============================================================================
# Parse Arguments
# =============================================================================

DATAROOT="$DEFAULT_DATAROOT"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mini)
            DOWNLOAD_MINI=true
            shift
            ;;
        --full)
            DOWNLOAD_FULL=true
            shift
            ;;
        --maps-only)
            DOWNLOAD_MAPS_ONLY=true
            shift
            ;;
        --dataroot)
            DATAROOT="$2"
            shift 2
            ;;
        --skip-checksums)
            SKIP_CHECKSUMS=true
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

print_banner

# Validate arguments
if [ "$DOWNLOAD_MINI" = false ] && [ "$DOWNLOAD_FULL" = false ] && [ "$DOWNLOAD_MAPS_ONLY" = false ]; then
    log_error "Please specify --mini, --full, or --maps-only"
    echo ""
    print_usage
    exit 1
fi

# Check dependencies
check_dependencies

# Create directory structure
log_info "Setting up directory: $DATAROOT"
mkdir -p "$DATAROOT"/{samples,sweeps,maps,v1.0-trainval,v1.0-mini}
mkdir -p "$DATAROOT/downloads"

DOWNLOAD_DIR="$DATAROOT/downloads"

# Check disk space
if [ "$DOWNLOAD_FULL" = true ]; then
    check_disk_space "$DATAROOT" 350
elif [ "$DOWNLOAD_MINI" = true ]; then
    check_disk_space "$DATAROOT" 10
fi

# Prompt for credentials
echo ""
echo "-----------------------------------------------------------------------"
echo "  nuScenes requires authentication to download."
echo "  If you don't have an account, register at: https://www.nuscenes.org/"
echo "  After logging in, find your access token on your account page."
echo "-----------------------------------------------------------------------"
echo ""
read -rp "Enter your nuScenes access token (or press Enter to use stored): " TOKEN

if [ -z "$TOKEN" ] && [ -f "$HOME/.nuscenes_token" ]; then
    TOKEN=$(cat "$HOME/.nuscenes_token")
    log_info "Using stored token from ~/.nuscenes_token"
elif [ -n "$TOKEN" ]; then
    echo "$TOKEN" > "$HOME/.nuscenes_token"
    chmod 600 "$HOME/.nuscenes_token"
    log_info "Token saved to ~/.nuscenes_token"
else
    log_error "No token provided and no stored token found."
    log_error "Please register at https://www.nuscenes.org/ and provide your token."
    exit 1
fi

# Download nuScenes mini
if [ "$DOWNLOAD_MINI" = true ]; then
    log_info "========================================="
    log_info "Downloading nuScenes v1.0-mini (~4 GB)"
    log_info "========================================="

    for file in "${MINI_FILES[@]}"; do
        download_file "${NUSCENES_BASE_URL}/${file}?token=${TOKEN}" \
            "${DOWNLOAD_DIR}/${file}"

        # Verify checksum if available
        if [ -n "${CHECKSUMS[$file]:-}" ]; then
            verify_checksum "${DOWNLOAD_DIR}/${file}" "${CHECKSUMS[$file]}"
        fi

        # Extract
        extract_archive "${DOWNLOAD_DIR}/${file}" "$DATAROOT"
    done
fi

# Download nuScenes trainval
if [ "$DOWNLOAD_FULL" = true ]; then
    log_info "========================================="
    log_info "Downloading nuScenes v1.0-trainval (~300 GB)"
    log_info "This will take a long time..."
    log_info "========================================="

    for file in "${TRAINVAL_FILES[@]}"; do
        download_file "${NUSCENES_BASE_URL}/${file}?token=${TOKEN}" \
            "${DOWNLOAD_DIR}/${file}"

        # Verify checksum if available
        if [ -n "${CHECKSUMS[$file]:-}" ]; then
            verify_checksum "${DOWNLOAD_DIR}/${file}" "${CHECKSUMS[$file]}"
        fi

        # Extract
        extract_archive "${DOWNLOAD_DIR}/${file}" "$DATAROOT"
    done
fi

# Download map expansion
if [ "$DOWNLOAD_MINI" = true ] || [ "$DOWNLOAD_FULL" = true ] || [ "$DOWNLOAD_MAPS_ONLY" = true ]; then
    log_info "========================================="
    log_info "Downloading nuScenes Map Expansion v1.3"
    log_info "========================================="

    download_file "${NUSCENES_BASE_URL}/${MAP_FILE}?token=${TOKEN}" \
        "${DOWNLOAD_DIR}/${MAP_FILE}"

    if [ -n "${CHECKSUMS[$MAP_FILE]:-}" ]; then
        verify_checksum "${DOWNLOAD_DIR}/${MAP_FILE}" "${CHECKSUMS[$MAP_FILE]}"
    fi

    # Extract maps to the maps directory
    extract_archive "${DOWNLOAD_DIR}/${MAP_FILE}" "$DATAROOT/maps"
fi

# =============================================================================
# Verify Directory Structure
# =============================================================================

log_info "========================================="
log_info "Verifying directory structure"
log_info "========================================="

echo ""
echo "Expected directory structure:"
echo "  $DATAROOT/"
echo "  +-- samples/"
echo "  |   +-- CAM_FRONT/"
echo "  |   +-- CAM_FRONT_LEFT/"
echo "  |   +-- CAM_FRONT_RIGHT/"
echo "  |   +-- CAM_BACK/"
echo "  |   +-- CAM_BACK_LEFT/"
echo "  |   +-- CAM_BACK_RIGHT/"
echo "  |   +-- LIDAR_TOP/"
echo "  +-- sweeps/"
echo "  +-- maps/"
echo "  |   +-- expansion/"
echo "  |   +-- basemap/"
echo "  +-- v1.0-trainval/ (or v1.0-mini/)"
echo "  |   +-- *.json (annotation files)"
echo "  +-- downloads/ (raw archives)"
echo ""

# Check key directories exist
EXPECTED_DIRS=("samples" "maps")
if [ "$DOWNLOAD_MINI" = true ]; then
    EXPECTED_DIRS+=("v1.0-mini")
fi
if [ "$DOWNLOAD_FULL" = true ]; then
    EXPECTED_DIRS+=("v1.0-trainval")
fi

all_good=true
for dir in "${EXPECTED_DIRS[@]}"; do
    if [ -d "$DATAROOT/$dir" ]; then
        log_info "  [OK] $dir/"
    else
        log_warn "  [MISSING] $dir/"
        all_good=false
    fi
done

echo ""
if [ "$all_good" = true ]; then
    log_info "Download and extraction complete!"
    log_info "Dataset root: $(realpath "$DATAROOT")"
    echo ""
    echo "Next steps:"
    echo "  1. Run map data preparation:"
    echo "     python scripts/prepare_map_data.py --dataroot $DATAROOT --version v1.0-trainval"
    echo ""
    echo "  2. Start training:"
    echo "     python tools/train.py configs/stream_mapnet_nuscenes.py"
else
    log_warn "Some directories are missing. Please check the download."
fi

echo ""
log_info "Total disk usage: $(du -sh "$DATAROOT" 2>/dev/null | cut -f1)"
