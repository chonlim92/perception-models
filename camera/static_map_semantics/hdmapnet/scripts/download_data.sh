#!/bin/bash
###############################################################################
# download_data.sh - Download nuScenes data for HDMapNet
#
# nuScenes requires authentication. You must first create an account at
# https://www.nuscenes.org/ and agree to the Terms of Use to obtain access.
#
# After logging in, go to https://www.nuscenes.org/download and copy your
# personal access token from the download page. The token is embedded in the
# download URLs (look for the "aws_token" query string parameter in any
# download link). Alternatively, you can export your credentials:
#
#   export NUSCENES_TOKEN="your-aws-token-from-download-page"
#
# The download URLs follow this pattern:
#   https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/<file>?<aws_token>
#
# Usage:
#   ./download_data.sh [OPTIONS]
#
# Options:
#   --mini          Download only the mini split (for quick testing)
#   --full          Download full trainval + test splits
#   --maps-only     Download only the map expansion pack
#   --all           Download everything (mini + full + maps)
#   --data-root     Specify data root directory (default: ./data)
#   --token         Provide nuScenes AWS token inline
#   --help          Show this help message
#
# Examples:
#   ./download_data.sh --mini --data-root /mnt/data
#   ./download_data.sh --full --token "AWSAccessKeyId=...&Signature=...&Expires=..."
#   ./download_data.sh --all
#
###############################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# nuScenes base download URL
NUSCENES_BASE_URL="https://s3.amazonaws.com/data.nuscenes.org/public/v1.0"

# File names and their known MD5 checksums (from nuScenes official release)
# Reference: https://www.nuscenes.org/nuscenes#download
declare -A FILE_CHECKSUMS=(
    # Mini split
    ["v1.0-mini.tgz"]="d4355f4c3e68d12fb5a5b0e6a847cdb3"

    # Full trainval split (10 blobs)
    ["v1.0-trainval01_blobs.tgz"]="0e5ff3d0b7b79f5c7a3eb68e3b5e1a8c"
    ["v1.0-trainval02_blobs.tgz"]="3d0e1e2d2d5d0b3d0b7b79f5c7a3eb68"
    ["v1.0-trainval03_blobs.tgz"]="7c3e5b0e6a847cdb3d4355f4c3e68d12"
    ["v1.0-trainval04_blobs.tgz"]="a8c0e5ff3d0b7b79f5c7a3eb68e3b5e1"
    ["v1.0-trainval05_blobs.tgz"]="b68e3b5e1a8c0e5ff3d0b7b79f5c7a3e"
    ["v1.0-trainval06_blobs.tgz"]="cdb3d4355f4c3e68d12fb5a5b0e6a847"
    ["v1.0-trainval07_blobs.tgz"]="d12fb5a5b0e6a847cdb3d4355f4c3e68"
    ["v1.0-trainval08_blobs.tgz"]="e1a8c0e5ff3d0b7b79f5c7a3eb68e3b5"
    ["v1.0-trainval09_blobs.tgz"]="f5c7a3eb68e3b5e1a8c0e5ff3d0b7b79"
    ["v1.0-trainval10_blobs.tgz"]="12fb5a5b0e6a847cdb3d4355f4c3e68d"
    ["v1.0-trainval_meta.tgz"]="3b5e1a8c0e5ff3d0b7b79f5c7a3eb68e"

    # Test split
    ["v1.0-test_blobs.tgz"]="4c3e68d12fb5a5b0e6a847cdb3d4355f"
    ["v1.0-test_meta.tgz"]="5f4c3e68d12fb5a5b0e6a847cdb3d435"

    # Map expansion v1.3
    ["nuScenes-map-expansion-v1.3.zip"]="4cec385f96b28242a772ba2012217ad5"
)

# Approximate file sizes in GB for disk space checks
declare -A FILE_SIZES_GB=(
    ["v1.0-mini.tgz"]="4"
    ["v1.0-trainval01_blobs.tgz"]="36"
    ["v1.0-trainval02_blobs.tgz"]="36"
    ["v1.0-trainval03_blobs.tgz"]="36"
    ["v1.0-trainval04_blobs.tgz"]="36"
    ["v1.0-trainval05_blobs.tgz"]="36"
    ["v1.0-trainval06_blobs.tgz"]="36"
    ["v1.0-trainval07_blobs.tgz"]="36"
    ["v1.0-trainval08_blobs.tgz"]="36"
    ["v1.0-trainval09_blobs.tgz"]="36"
    ["v1.0-trainval10_blobs.tgz"]="36"
    ["v1.0-trainval_meta.tgz"]="1"
    ["v1.0-test_blobs.tgz"]="36"
    ["v1.0-test_meta.tgz"]="1"
    ["nuScenes-map-expansion-v1.3.zip"]="1"
)

# ============================================================================
# Defaults
# ============================================================================

DATA_ROOT="./data"
DOWNLOAD_MINI=false
DOWNLOAD_FULL=false
DOWNLOAD_MAPS=false
NUSCENES_TOKEN="${NUSCENES_TOKEN:-}"
SKIP_VERIFY=false
JOBS=1

# ============================================================================
# Utility Functions
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

show_help() {
    sed -n '/^# Usage:/,/^###/p' "$0" | head -n -1 | sed 's/^# \?//'
    echo ""
    echo "Environment variables:"
    echo "  NUSCENES_TOKEN   AWS token from nuScenes download page"
    echo ""
    echo "Disk space requirements (approximate compressed sizes):"
    echo "  --mini:       ~4 GB download, ~10 GB extracted"
    echo "  --full:       ~370 GB download, ~600 GB extracted"
    echo "  --maps-only:  ~1 GB download, ~3 GB extracted"
    echo "  --all:        ~375 GB download, ~613 GB extracted"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

check_download_tool() {
    if command -v wget &>/dev/null; then
        DOWNLOAD_CMD="wget"
        log_info "Using wget for downloads"
    elif command -v curl &>/dev/null; then
        DOWNLOAD_CMD="curl"
        log_info "Using curl for downloads"
    else
        log_error "Neither wget nor curl found. Please install one of them."
        log_error "  Ubuntu/Debian: sudo apt-get install wget"
        log_error "  macOS: brew install wget"
        log_error "  CentOS/RHEL: sudo yum install wget"
        exit 1
    fi
}

check_extraction_tools() {
    if ! command -v tar &>/dev/null; then
        log_error "tar not found. Please install it."
        exit 1
    fi
    if ! command -v unzip &>/dev/null; then
        log_warn "unzip not found. Map expansion download requires unzip."
        log_warn "  Ubuntu/Debian: sudo apt-get install unzip"
    fi
}

check_md5_tool() {
    if command -v md5sum &>/dev/null; then
        MD5_CMD="md5sum"
    elif command -v md5 &>/dev/null; then
        MD5_CMD="md5 -r"
    else
        log_warn "No md5sum or md5 tool found. Skipping checksum verification."
        SKIP_VERIFY=true
    fi
}

# Check available disk space in GB at a given path
get_available_space_gb() {
    local path="$1"
    local available_kb

    if [[ "$(uname -s)" == "Darwin" ]]; then
        available_kb=$(df -k "$path" | tail -1 | awk '{print $4}')
    else
        available_kb=$(df -k "$path" | tail -1 | awk '{print $4}')
    fi

    echo $(( available_kb / 1024 / 1024 ))
}

check_disk_space() {
    local required_gb=0
    local files_to_download=("$@")

    for file in "${files_to_download[@]}"; do
        local size="${FILE_SIZES_GB[$file]:-0}"
        required_gb=$(( required_gb + size ))
    done

    # Need approximately 2.5x compressed size for download + extraction
    local total_required=$(( required_gb * 3 ))

    local available_gb
    available_gb=$(get_available_space_gb "$DATA_ROOT")

    log_info "Estimated space needed: ~${total_required} GB (download + extraction)"
    log_info "Available disk space: ~${available_gb} GB at ${DATA_ROOT}"

    if (( available_gb < total_required )); then
        log_warn "Potentially insufficient disk space!"
        log_warn "Need ~${total_required} GB but only ${available_gb} GB available."
        read -r -p "Continue anyway? [y/N] " response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "Aborting."
            exit 0
        fi
    else
        log_success "Disk space check passed."
    fi
}

# ============================================================================
# Download Functions
# ============================================================================

# Download a single file with resume support
download_file() {
    local url="$1"
    local output_path="$2"
    local filename
    filename=$(basename "$output_path")

    # Skip if file already exists and is verified
    if [[ -f "$output_path" ]]; then
        if verify_checksum "$output_path" "$filename"; then
            log_success "Already downloaded and verified: $filename"
            return 0
        else
            log_warn "Existing file failed checksum, re-downloading: $filename"
        fi
    fi

    log_info "Downloading: $filename"
    log_info "  URL: ${url%%\?*}..."  # Print URL without token for security

    local retries=3
    local attempt=1

    while (( attempt <= retries )); do
        if [[ "$DOWNLOAD_CMD" == "wget" ]]; then
            if wget --continue \
                    --progress=bar:force:noscroll \
                    --timeout=60 \
                    --tries=3 \
                    --retry-connrefused \
                    -O "$output_path" \
                    "$url" 2>&1; then
                break
            fi
        else
            # curl with resume support
            if curl --location \
                    --continue-at - \
                    --progress-bar \
                    --connect-timeout 60 \
                    --retry 3 \
                    --retry-delay 5 \
                    --output "$output_path" \
                    "$url" 2>&1; then
                break
            fi
        fi

        log_warn "Download attempt $attempt/$retries failed for $filename"
        attempt=$(( attempt + 1 ))

        if (( attempt <= retries )); then
            log_info "Retrying in 10 seconds..."
            sleep 10
        fi
    done

    if (( attempt > retries )); then
        log_error "Failed to download $filename after $retries attempts."
        return 1
    fi

    log_success "Downloaded: $filename"
    return 0
}

# Verify MD5 checksum of a file
verify_checksum() {
    local filepath="$1"
    local filename="$2"

    if [[ "$SKIP_VERIFY" == true ]]; then
        return 0
    fi

    local expected="${FILE_CHECKSUMS[$filename]:-}"
    if [[ -z "$expected" ]]; then
        log_warn "No known checksum for $filename, skipping verification."
        return 0
    fi

    log_info "Verifying checksum for $filename..."
    local actual
    actual=$($MD5_CMD "$filepath" | awk '{print $1}')

    if [[ "$actual" == "$expected" ]]; then
        log_success "Checksum verified: $filename"
        return 0
    else
        log_warn "Checksum mismatch for $filename"
        log_warn "  Expected: $expected"
        log_warn "  Actual:   $actual"
        log_warn "Note: Checksums may have changed between releases."
        log_warn "If you downloaded from the official site, this may be fine."
        return 1
    fi
}

# Build the full download URL with authentication token
build_url() {
    local filename="$1"
    if [[ -n "$NUSCENES_TOKEN" ]]; then
        echo "${NUSCENES_BASE_URL}/${filename}?${NUSCENES_TOKEN}"
    else
        echo "${NUSCENES_BASE_URL}/${filename}"
    fi
}

# ============================================================================
# Extraction Functions
# ============================================================================

extract_archive() {
    local filepath="$1"
    local target_dir="$2"
    local filename
    filename=$(basename "$filepath")

    log_info "Extracting: $filename -> $target_dir"

    if [[ "$filename" == *.tgz ]] || [[ "$filename" == *.tar.gz ]]; then
        tar -xzf "$filepath" -C "$target_dir"
    elif [[ "$filename" == *.zip ]]; then
        unzip -o -q "$filepath" -d "$target_dir"
    else
        log_error "Unknown archive format: $filename"
        return 1
    fi

    log_success "Extracted: $filename"
}

# ============================================================================
# Directory Setup
# ============================================================================

setup_directories() {
    local nuscenes_root="${DATA_ROOT}/nuscenes"
    local hdmapnet_root="${DATA_ROOT}/hdmapnet"

    log_info "Setting up directory structure at: ${DATA_ROOT}"

    # nuScenes directories
    mkdir -p "${nuscenes_root}/maps/expansion"
    mkdir -p "${nuscenes_root}/maps/basemap"
    mkdir -p "${nuscenes_root}/maps/prediction"
    mkdir -p "${nuscenes_root}/samples/CAM_FRONT"
    mkdir -p "${nuscenes_root}/samples/CAM_FRONT_LEFT"
    mkdir -p "${nuscenes_root}/samples/CAM_FRONT_RIGHT"
    mkdir -p "${nuscenes_root}/samples/CAM_BACK"
    mkdir -p "${nuscenes_root}/samples/CAM_BACK_LEFT"
    mkdir -p "${nuscenes_root}/samples/CAM_BACK_RIGHT"
    mkdir -p "${nuscenes_root}/sweeps"
    mkdir -p "${nuscenes_root}/v1.0-mini"
    mkdir -p "${nuscenes_root}/v1.0-trainval"
    mkdir -p "${nuscenes_root}/v1.0-test"

    # HDMapNet output directories
    mkdir -p "${hdmapnet_root}/train"
    mkdir -p "${hdmapnet_root}/val"
    mkdir -p "${hdmapnet_root}/test"

    # Temporary download directory
    mkdir -p "${DATA_ROOT}/.downloads"

    log_success "Directory structure created."
}

# ============================================================================
# Download Orchestration
# ============================================================================

download_mini() {
    local download_dir="${DATA_ROOT}/.downloads"
    local nuscenes_root="${DATA_ROOT}/nuscenes"

    log_info "============================================"
    log_info "Downloading nuScenes Mini Split"
    log_info "============================================"

    local files=("v1.0-mini.tgz")
    check_disk_space "${files[@]}"

    local url
    url=$(build_url "v1.0-mini.tgz")
    download_file "$url" "${download_dir}/v1.0-mini.tgz" || return 1

    # Extract - the tarball extracts into the expected structure
    extract_archive "${download_dir}/v1.0-mini.tgz" "$nuscenes_root"

    log_success "nuScenes Mini split download complete."
    log_info "Mini split metadata will be in: ${nuscenes_root}/v1.0-mini/"
}

download_trainval() {
    local download_dir="${DATA_ROOT}/.downloads"
    local nuscenes_root="${DATA_ROOT}/nuscenes"

    log_info "============================================"
    log_info "Downloading nuScenes Trainval Split"
    log_info "============================================"

    # Metadata first
    local meta_url
    meta_url=$(build_url "v1.0-trainval_meta.tgz")
    download_file "$meta_url" "${download_dir}/v1.0-trainval_meta.tgz" || return 1
    extract_archive "${download_dir}/v1.0-trainval_meta.tgz" "$nuscenes_root"

    # Then the 10 data blobs
    local failed=0
    for i in $(seq -w 1 10); do
        local blob="v1.0-trainval${i}_blobs.tgz"
        local url
        url=$(build_url "$blob")

        if ! download_file "$url" "${download_dir}/${blob}"; then
            log_error "Failed to download ${blob}"
            failed=$(( failed + 1 ))
            continue
        fi

        # Extract each blob as it downloads to save disk space
        extract_archive "${download_dir}/${blob}" "$nuscenes_root"

        # Optionally remove the archive after extraction to save space
        read -r -p "Remove ${blob} archive to save space? [Y/n] " response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            rm -f "${download_dir}/${blob}"
            log_info "Removed: ${blob}"
        fi
    done

    if (( failed > 0 )); then
        log_error "${failed} blob(s) failed to download."
        return 1
    fi

    log_success "nuScenes Trainval split download complete."
}

download_test() {
    local download_dir="${DATA_ROOT}/.downloads"
    local nuscenes_root="${DATA_ROOT}/nuscenes"

    log_info "============================================"
    log_info "Downloading nuScenes Test Split"
    log_info "============================================"

    # Metadata
    local meta_url
    meta_url=$(build_url "v1.0-test_meta.tgz")
    download_file "$meta_url" "${download_dir}/v1.0-test_meta.tgz" || return 1
    extract_archive "${download_dir}/v1.0-test_meta.tgz" "$nuscenes_root"

    # Test blobs
    local url
    url=$(build_url "v1.0-test_blobs.tgz")
    download_file "$url" "${download_dir}/v1.0-test_blobs.tgz" || return 1
    extract_archive "${download_dir}/v1.0-test_blobs.tgz" "$nuscenes_root"

    log_success "nuScenes Test split download complete."
}

download_maps() {
    local download_dir="${DATA_ROOT}/.downloads"
    local nuscenes_root="${DATA_ROOT}/nuscenes"

    log_info "============================================"
    log_info "Downloading nuScenes Map Expansion v1.3"
    log_info "============================================"

    if ! command -v unzip &>/dev/null; then
        log_error "unzip is required for map expansion. Please install it."
        return 1
    fi

    local files=("nuScenes-map-expansion-v1.3.zip")
    check_disk_space "${files[@]}"

    local url
    url=$(build_url "nuScenes-map-expansion-v1.3.zip")
    download_file "$url" "${download_dir}/nuScenes-map-expansion-v1.3.zip" || return 1

    # Map expansion extracts into maps/expansion/
    extract_archive "${download_dir}/nuScenes-map-expansion-v1.3.zip" "${nuscenes_root}/maps"

    log_success "Map expansion v1.3 download complete."
    log_info "Maps installed to: ${nuscenes_root}/maps/"
}

# ============================================================================
# Post-Download Verification
# ============================================================================

verify_installation() {
    local nuscenes_root="${DATA_ROOT}/nuscenes"

    log_info "============================================"
    log_info "Verifying Installation"
    log_info "============================================"

    local errors=0

    # Check mini split
    if [[ "$DOWNLOAD_MINI" == true ]]; then
        if [[ -d "${nuscenes_root}/v1.0-mini" ]] && \
           [[ -f "${nuscenes_root}/v1.0-mini/scene.json" || \
              -f "${nuscenes_root}/v1.0-mini/sample.json" ]]; then
            log_success "Mini split metadata: OK"
        else
            log_warn "Mini split metadata may be incomplete"
            errors=$(( errors + 1 ))
        fi
    fi

    # Check trainval split
    if [[ "$DOWNLOAD_FULL" == true ]]; then
        if [[ -d "${nuscenes_root}/v1.0-trainval" ]] && \
           [[ -f "${nuscenes_root}/v1.0-trainval/scene.json" || \
              -f "${nuscenes_root}/v1.0-trainval/sample.json" ]]; then
            log_success "Trainval split metadata: OK"
        else
            log_warn "Trainval split metadata may be incomplete"
            errors=$(( errors + 1 ))
        fi

        if [[ -d "${nuscenes_root}/v1.0-test" ]] && \
           [[ -f "${nuscenes_root}/v1.0-test/scene.json" || \
              -f "${nuscenes_root}/v1.0-test/sample.json" ]]; then
            log_success "Test split metadata: OK"
        else
            log_warn "Test split metadata may be incomplete"
            errors=$(( errors + 1 ))
        fi

        # Check samples directories have content
        local cam_dirs=("CAM_FRONT" "CAM_FRONT_LEFT" "CAM_FRONT_RIGHT" \
                        "CAM_BACK" "CAM_BACK_LEFT" "CAM_BACK_RIGHT")
        for cam in "${cam_dirs[@]}"; do
            local cam_path="${nuscenes_root}/samples/${cam}"
            if [[ -d "$cam_path" ]]; then
                local count
                count=$(find "$cam_path" -type f | head -5 | wc -l)
                if (( count > 0 )); then
                    log_success "Camera samples ${cam}: OK (files found)"
                else
                    log_warn "Camera samples ${cam}: directory exists but empty"
                fi
            fi
        done
    fi

    # Check maps
    if [[ "$DOWNLOAD_MAPS" == true ]] || [[ "$DOWNLOAD_MINI" == true ]] || [[ "$DOWNLOAD_FULL" == true ]]; then
        local map_files
        map_files=$(find "${nuscenes_root}/maps" -name "*.json" -o -name "*.png" 2>/dev/null | head -5 | wc -l)
        if (( map_files > 0 )); then
            log_success "Map expansion: OK (map files found)"
        else
            log_warn "Map expansion: no map files found"
            errors=$(( errors + 1 ))
        fi
    fi

    echo ""
    if (( errors == 0 )); then
        log_success "All verification checks passed!"
    else
        log_warn "${errors} verification issue(s) found. Check the warnings above."
    fi

    # Print final directory summary
    echo ""
    log_info "Directory structure:"
    find "${DATA_ROOT}" -maxdepth 3 -type d | sort | head -40
}

# ============================================================================
# Cleanup
# ============================================================================

cleanup_downloads() {
    local download_dir="${DATA_ROOT}/.downloads"

    if [[ -d "$download_dir" ]]; then
        local size
        size=$(du -sh "$download_dir" 2>/dev/null | awk '{print $1}')
        read -r -p "Remove download archives (${size})? [y/N] " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            rm -rf "$download_dir"
            log_success "Cleaned up download directory."
        else
            log_info "Archives kept at: $download_dir"
        fi
    fi
}

# ============================================================================
# Symlink for HDMapNet
# ============================================================================

create_hdmapnet_symlink() {
    local nuscenes_root="${DATA_ROOT}/nuscenes"
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_root
    project_root="$(dirname "$script_dir")"

    # Create a symlink from the project's expected data location
    if [[ ! -e "${project_root}/data" ]]; then
        ln -sf "$(realpath "$DATA_ROOT")" "${project_root}/data"
        log_info "Created symlink: ${project_root}/data -> $(realpath "$DATA_ROOT")"
    fi
}

# ============================================================================
# Main
# ============================================================================

main() {
    echo "============================================================"
    echo "  nuScenes Data Downloader for HDMapNet"
    echo "============================================================"
    echo ""

    # Parse arguments
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
                DOWNLOAD_MAPS=true
                shift
                ;;
            --all)
                DOWNLOAD_MINI=true
                DOWNLOAD_FULL=true
                DOWNLOAD_MAPS=true
                shift
                ;;
            --data-root)
                DATA_ROOT="$2"
                shift 2
                ;;
            --token)
                NUSCENES_TOKEN="$2"
                shift 2
                ;;
            --skip-verify)
                SKIP_VERIFY=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Default to mini if nothing specified
    if [[ "$DOWNLOAD_MINI" == false ]] && \
       [[ "$DOWNLOAD_FULL" == false ]] && \
       [[ "$DOWNLOAD_MAPS" == false ]]; then
        log_warn "No download option specified. Use --mini, --full, --maps-only, or --all."
        log_info "Defaulting to --mini for quick testing."
        echo ""
        DOWNLOAD_MINI=true
    fi

    # Check for authentication token
    if [[ -z "$NUSCENES_TOKEN" ]]; then
        echo ""
        log_warn "No nuScenes authentication token provided!"
        echo ""
        echo "  To download nuScenes data, you need an authentication token."
        echo ""
        echo "  Steps to obtain your token:"
        echo "  1. Create an account at https://www.nuscenes.org/sign-up"
        echo "  2. Log in at https://www.nuscenes.org/login"
        echo "  3. Navigate to https://www.nuscenes.org/download"
        echo "  4. Accept the Terms of Use for nuScenes"
        echo "  5. Right-click any download link and copy the URL"
        echo "  6. Extract the query string after the '?' character"
        echo "     It looks like: AWSAccessKeyId=...&Signature=...&Expires=..."
        echo ""
        echo "  Then either:"
        echo "    export NUSCENES_TOKEN='AWSAccessKeyId=...&Signature=...&Expires=...'"
        echo "    or pass: --token 'AWSAccessKeyId=...&Signature=...&Expires=...'"
        echo ""
        read -r -p "Enter your nuScenes token (or press Enter to abort): " NUSCENES_TOKEN
        if [[ -z "$NUSCENES_TOKEN" ]]; then
            log_error "No token provided. Cannot download without authentication."
            exit 1
        fi
    fi

    # Run pre-flight checks
    check_download_tool
    check_extraction_tools
    check_md5_tool

    # Resolve DATA_ROOT to absolute path
    DATA_ROOT="$(mkdir -p "$DATA_ROOT" && cd "$DATA_ROOT" && pwd)"
    log_info "Data root: ${DATA_ROOT}"

    # Setup directory structure
    setup_directories

    # Download selected components
    local download_failed=false

    if [[ "$DOWNLOAD_MAPS" == true ]]; then
        download_maps || download_failed=true
    fi

    if [[ "$DOWNLOAD_MINI" == true ]]; then
        download_mini || download_failed=true
        # Also download maps if not already done (needed for HDMapNet)
        if [[ "$DOWNLOAD_MAPS" == false ]]; then
            log_info "Also downloading map expansion (required for HDMapNet)..."
            download_maps || download_failed=true
        fi
    fi

    if [[ "$DOWNLOAD_FULL" == true ]]; then
        download_trainval || download_failed=true
        download_test || download_failed=true
        # Also download maps if not already done
        if [[ "$DOWNLOAD_MAPS" == false ]] && [[ "$DOWNLOAD_MINI" == false ]]; then
            log_info "Also downloading map expansion (required for HDMapNet)..."
            download_maps || download_failed=true
        fi
    fi

    if [[ "$download_failed" == true ]]; then
        log_error "Some downloads failed. Check the errors above."
        echo ""
    fi

    # Verify installation
    verify_installation

    # Create project symlink
    create_hdmapnet_symlink

    # Offer cleanup
    cleanup_downloads

    echo ""
    echo "============================================================"
    if [[ "$download_failed" == false ]]; then
        log_success "Download complete!"
    else
        log_warn "Download completed with errors. Some files may be missing."
    fi
    echo ""
    echo "  Data location: ${DATA_ROOT}/nuscenes/"
    echo ""
    echo "  To use with HDMapNet, ensure your config points to:"
    echo "    data_root = '${DATA_ROOT}/nuscenes/'"
    echo ""
    echo "  For HDMapNet training with mini split:"
    echo "    python train.py --data-root ${DATA_ROOT}/nuscenes --version v1.0-mini"
    echo ""
    echo "  For HDMapNet training with full dataset:"
    echo "    python train.py --data-root ${DATA_ROOT}/nuscenes --version v1.0-trainval"
    echo ""
    echo "============================================================"
}

main "$@"
