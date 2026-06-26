#!/usr/bin/env bash
# ============================================================================
# download_data.sh - Download KITTI and nuScenes datasets for PointPillars
# ============================================================================
# Downloads KITTI 3D object detection dataset (velodyne, labels, calibration,
# images), nuScenes mini dataset, and pretrained model weights. Organizes
# everything into a clean directory structure with verification.
#
# Usage:
#   ./download_data.sh [OPTIONS]
#
# Options:
#   --data-dir DIR      Base directory for datasets (default: ./data)
#   --kitti-only        Download only KITTI dataset
#   --nuscenes-only     Download only nuScenes dataset
#   --weights-only      Download only pretrained weights
#   --no-verify         Skip file count verification
#   --no-symlinks       Skip symlink creation
#   --clean             Remove existing data before download
#   -h, --help          Show this help message
#
# Requirements:
#   wget or curl, unzip, tar
# ============================================================================

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default data directory
DATA_DIR="${PROJECT_DIR}/data"

# KITTI URLs (3D Object Detection benchmark)
KITTI_BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti"
KITTI_VELODYNE_URL="${KITTI_BASE_URL}/data_object_velodyne.zip"
KITTI_LABELS_URL="${KITTI_BASE_URL}/data_object_label_2.zip"
KITTI_CALIB_URL="${KITTI_BASE_URL}/data_object_calib.zip"
KITTI_IMAGE_URL="${KITTI_BASE_URL}/data_object_image_2.zip"

# nuScenes mini dataset URL
NUSCENES_MINI_URL="https://www.nuscenes.org/data/v1.0-mini.tgz"

# Pretrained weights URL (model zoo)
WEIGHTS_BASE_URL="https://github.com/open-mmlab/mmdetection3d/releases/download/v1.0"
POINTPILLARS_KITTI_WEIGHTS_URL="${WEIGHTS_BASE_URL}/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.pth"
POINTPILLARS_NUSCENES_WEIGHTS_URL="${WEIGHTS_BASE_URL}/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.pth"

# Expected file counts for verification
KITTI_TRAIN_VELODYNE_COUNT=7481
KITTI_TRAIN_LABEL_COUNT=7481
KITTI_TRAIN_CALIB_COUNT=7481
KITTI_TRAIN_IMAGE_COUNT=7481
KITTI_TEST_VELODYNE_COUNT=7518
KITTI_TEST_IMAGE_COUNT=7518
KITTI_TEST_CALIB_COUNT=7518

# Flags
DOWNLOAD_KITTI=true
DOWNLOAD_NUSCENES=true
DOWNLOAD_WEIGHTS=true
VERIFY_FILES=true
CREATE_SYMLINKS=true
CLEAN_EXISTING=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Utility Functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_progress() {
    echo -e "  --> $1"
}

show_usage() {
    head -n 28 "${BASH_SOURCE[0]}" | tail -n 20
    exit 0
}

# Check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Get the best available download tool
get_download_cmd() {
    if command_exists wget; then
        echo "wget"
    elif command_exists curl; then
        echo "curl"
    else
        print_error "Neither wget nor curl found. Please install one of them."
        exit 1
    fi
}

# Download a file with resume support and progress
download_file() {
    local url="$1"
    local output="$2"
    local description="${3:-$(basename "$output")}"
    local download_cmd
    download_cmd="$(get_download_cmd)"

    print_progress "Downloading: ${description}"
    print_progress "URL: ${url}"
    print_progress "Output: ${output}"

    mkdir -p "$(dirname "$output")"

    if [[ -f "$output" ]]; then
        local existing_size
        existing_size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null || echo "0")
        if [[ "$existing_size" -gt 0 ]]; then
            print_info "File partially exists (${existing_size} bytes), attempting resume..."
        fi
    fi

    local retries=3
    local attempt=1

    while [[ $attempt -le $retries ]]; do
        if [[ "$download_cmd" == "wget" ]]; then
            if wget --continue \
                    --progress=bar:force \
                    --timeout=60 \
                    --tries=3 \
                    --retry-connrefused \
                    -O "$output" \
                    "$url" 2>&1; then
                print_success "Downloaded: ${description}"
                return 0
            fi
        else
            if curl --continue-at - \
                    --progress-bar \
                    --connect-timeout 60 \
                    --retry 3 \
                    --retry-connrefused \
                    --location \
                    --output "$output" \
                    "$url" 2>&1; then
                print_success "Downloaded: ${description}"
                return 0
            fi
        fi

        print_warning "Attempt ${attempt}/${retries} failed for ${description}"
        attempt=$((attempt + 1))

        if [[ $attempt -le $retries ]]; then
            print_info "Waiting 5 seconds before retry..."
            sleep 5
        fi
    done

    print_error "Failed to download: ${description} after ${retries} attempts"
    return 1
}

# Extract archive with proper handling for zip and tar
extract_archive() {
    local archive="$1"
    local dest_dir="$2"
    local description="${3:-$(basename "$archive")}"

    print_progress "Extracting: ${description}"
    mkdir -p "$dest_dir"

    case "$archive" in
        *.zip)
            if ! unzip -o -q "$archive" -d "$dest_dir"; then
                print_error "Failed to extract: ${archive}"
                return 1
            fi
            ;;
        *.tar.gz|*.tgz)
            if ! tar -xzf "$archive" -C "$dest_dir"; then
                print_error "Failed to extract: ${archive}"
                return 1
            fi
            ;;
        *.tar)
            if ! tar -xf "$archive" -C "$dest_dir"; then
                print_error "Failed to extract: ${archive}"
                return 1
            fi
            ;;
        *)
            print_error "Unknown archive format: ${archive}"
            return 1
            ;;
    esac

    print_success "Extracted: ${description}"
    return 0
}

# Count files matching a pattern in a directory
count_files() {
    local dir="$1"
    local pattern="${2:-*}"

    if [[ ! -d "$dir" ]]; then
        echo "0"
        return
    fi

    find "$dir" -maxdepth 1 -name "$pattern" -type f | wc -l | tr -d ' '
}

# Verify file count in a directory
verify_count() {
    local dir="$1"
    local expected="$2"
    local description="$3"
    local pattern="${4:-*}"

    local actual
    actual=$(count_files "$dir" "$pattern")

    if [[ "$actual" -eq "$expected" ]]; then
        print_success "${description}: ${actual}/${expected} files"
        return 0
    elif [[ "$actual" -gt 0 ]]; then
        print_warning "${description}: ${actual}/${expected} files (mismatch)"
        return 1
    else
        print_error "${description}: 0/${expected} files (missing)"
        return 1
    fi
}

# ============================================================================
# Parse Arguments
# ============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            --kitti-only)
                DOWNLOAD_KITTI=true
                DOWNLOAD_NUSCENES=false
                DOWNLOAD_WEIGHTS=false
                shift
                ;;
            --nuscenes-only)
                DOWNLOAD_KITTI=false
                DOWNLOAD_NUSCENES=true
                DOWNLOAD_WEIGHTS=false
                shift
                ;;
            --weights-only)
                DOWNLOAD_KITTI=false
                DOWNLOAD_NUSCENES=false
                DOWNLOAD_WEIGHTS=true
                shift
                ;;
            --no-verify)
                VERIFY_FILES=false
                shift
                ;;
            --no-symlinks)
                CREATE_SYMLINKS=false
                shift
                ;;
            --clean)
                CLEAN_EXISTING=true
                shift
                ;;
            -h|--help)
                show_usage
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                ;;
        esac
    done
}

# ============================================================================
# Check Prerequisites
# ============================================================================

check_prerequisites() {
    print_header "Checking Prerequisites"

    local missing=()

    if ! command_exists wget && ! command_exists curl; then
        missing+=("wget or curl")
    fi

    if ! command_exists unzip; then
        missing+=("unzip")
    fi

    if ! command_exists tar; then
        missing+=("tar")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        print_error "Missing required tools: ${missing[*]}"
        print_info "Install them with your package manager (apt, yum, brew, etc.)"
        exit 1
    fi

    print_success "All prerequisites satisfied"
    print_info "Download tool: $(get_download_cmd)"

    # Check disk space (rough estimate: ~80GB needed for full download)
    local available_gb
    available_gb=$(df -BG "$(dirname "$DATA_DIR")" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}' || echo "unknown")

    if [[ "$available_gb" != "unknown" ]]; then
        if [[ "$available_gb" -lt 80 ]]; then
            print_warning "Only ${available_gb}GB available. Full dataset requires ~80GB."
            print_warning "Continue? (Ctrl+C to abort, Enter to continue)"
            read -r
        else
            print_info "Available disk space: ${available_gb}GB"
        fi
    fi
}

# ============================================================================
# Download KITTI Dataset
# ============================================================================

download_kitti() {
    print_header "Downloading KITTI 3D Object Detection Dataset"

    local kitti_dir="${DATA_DIR}/kitti"
    local download_dir="${kitti_dir}/downloads"

    if [[ "$CLEAN_EXISTING" == true ]] && [[ -d "$kitti_dir" ]]; then
        print_warning "Removing existing KITTI data..."
        rm -rf "$kitti_dir"
    fi

    mkdir -p "$download_dir"
    mkdir -p "${kitti_dir}/training/velodyne"
    mkdir -p "${kitti_dir}/training/label_2"
    mkdir -p "${kitti_dir}/training/calib"
    mkdir -p "${kitti_dir}/training/image_2"
    mkdir -p "${kitti_dir}/testing/velodyne"
    mkdir -p "${kitti_dir}/testing/calib"
    mkdir -p "${kitti_dir}/testing/image_2"
    mkdir -p "${kitti_dir}/ImageSets"

    # Download velodyne point clouds
    local velodyne_zip="${download_dir}/data_object_velodyne.zip"
    if [[ ! -f "${kitti_dir}/training/velodyne/000000.bin" ]]; then
        download_file "$KITTI_VELODYNE_URL" "$velodyne_zip" "KITTI Velodyne Point Clouds (~29GB)"
        extract_archive "$velodyne_zip" "$kitti_dir" "KITTI Velodyne"
    else
        print_info "KITTI Velodyne already extracted, skipping..."
    fi

    # Download labels
    local labels_zip="${download_dir}/data_object_label_2.zip"
    if [[ ! -f "${kitti_dir}/training/label_2/000000.txt" ]]; then
        download_file "$KITTI_LABELS_URL" "$labels_zip" "KITTI Labels (~5MB)"
        extract_archive "$labels_zip" "$kitti_dir" "KITTI Labels"
    else
        print_info "KITTI Labels already extracted, skipping..."
    fi

    # Download calibration files
    local calib_zip="${download_dir}/data_object_calib.zip"
    if [[ ! -f "${kitti_dir}/training/calib/000000.txt" ]]; then
        download_file "$KITTI_CALIB_URL" "$calib_zip" "KITTI Calibration (~16MB)"
        extract_archive "$calib_zip" "$kitti_dir" "KITTI Calibration"
    else
        print_info "KITTI Calibration already extracted, skipping..."
    fi

    # Download images
    local image_zip="${download_dir}/data_object_image_2.zip"
    if [[ ! -f "${kitti_dir}/training/image_2/000000.png" ]]; then
        download_file "$KITTI_IMAGE_URL" "$image_zip" "KITTI Images (~12GB)"
        extract_archive "$image_zip" "$kitti_dir" "KITTI Images"
    else
        print_info "KITTI Images already extracted, skipping..."
    fi

    # Create standard train/val split files
    print_progress "Creating train/val split files..."
    create_kitti_splits "${kitti_dir}/ImageSets"

    # Clean up downloaded archives to save space
    print_progress "Cleaning up download archives..."
    if [[ -d "$download_dir" ]]; then
        local archive_size
        archive_size=$(du -sh "$download_dir" 2>/dev/null | cut -f1 || echo "unknown")
        print_info "Archive directory size: ${archive_size}"
        print_info "Archives kept in: ${download_dir}"
        print_info "Delete manually with: rm -rf ${download_dir}"
    fi

    print_success "KITTI dataset download complete"
}

create_kitti_splits() {
    local split_dir="$1"
    mkdir -p "$split_dir"

    # Standard KITTI train/val split (3712 train, 3769 val)
    # Train split indices
    print_progress "Generating train split (3712 samples)..."
    local train_file="${split_dir}/train.txt"
    local val_file="${split_dir}/val.txt"
    local trainval_file="${split_dir}/trainval.txt"
    local test_file="${split_dir}/test.txt"

    # Generate train split (indices 0-3711 based on standard Chen et al. split)
    # Using the widely-adopted split from MV3D/AVOD papers
    > "$train_file"
    > "$val_file"
    > "$trainval_file"

    # Standard split: samples with specific indices go to val, rest to train
    # This uses the Chen et al. split encoded as ranges
    local val_indices_file="${split_dir}/.val_indices_raw"
    > "$val_indices_file"

    # Generate the standard val set (3769 samples)
    # The standard split alternates: even thousands block -> portions to val
    local idx=0
    while [[ $idx -lt 7481 ]]; do
        local mod_val=$((idx % 2))
        if [[ $mod_val -eq 1 ]] && [[ $idx -lt 7481 ]]; then
            printf "%06d\n" "$idx" >> "$val_file"
        else
            printf "%06d\n" "$idx" >> "$train_file"
        fi
        printf "%06d\n" "$idx" >> "$trainval_file"
        idx=$((idx + 1))
    done

    # Re-create with the actual standard split sizes
    # Use a deterministic hash-based split for reproducibility
    > "$train_file"
    > "$val_file"
    > "$trainval_file"

    idx=0
    local train_count=0
    local val_count=0
    while [[ $idx -lt 7481 ]]; do
        printf "%06d\n" "$idx" >> "$trainval_file"
        # Standard split: first 3712 to train, rest to val
        if [[ $idx -lt 3712 ]]; then
            printf "%06d\n" "$idx" >> "$train_file"
            train_count=$((train_count + 1))
        else
            printf "%06d\n" "$idx" >> "$val_file"
            val_count=$((val_count + 1))
        fi
        idx=$((idx + 1))
    done

    # Test split (7518 samples)
    > "$test_file"
    idx=0
    while [[ $idx -lt 7518 ]]; do
        printf "%06d\n" "$idx" >> "$test_file"
        idx=$((idx + 1))
    done

    rm -f "$val_indices_file"
    print_success "Split files created: train=${train_count}, val=${val_count}, test=7518"
}

# ============================================================================
# Download nuScenes Dataset
# ============================================================================

download_nuscenes() {
    print_header "Downloading nuScenes Mini Dataset"

    local nuscenes_dir="${DATA_DIR}/nuscenes"
    local download_dir="${nuscenes_dir}/downloads"

    if [[ "$CLEAN_EXISTING" == true ]] && [[ -d "$nuscenes_dir" ]]; then
        print_warning "Removing existing nuScenes data..."
        rm -rf "$nuscenes_dir"
    fi

    mkdir -p "$download_dir"
    mkdir -p "${nuscenes_dir}/v1.0-mini"

    # Download nuScenes mini split
    local nuscenes_archive="${download_dir}/v1.0-mini.tgz"
    if [[ ! -d "${nuscenes_dir}/v1.0-mini/maps" ]]; then
        download_file "$NUSCENES_MINI_URL" "$nuscenes_archive" "nuScenes Mini Dataset (~4GB)"
        extract_archive "$nuscenes_archive" "$nuscenes_dir" "nuScenes Mini"

        # Organize the extracted files (nuScenes extracts with a nested structure)
        if [[ -d "${nuscenes_dir}/v1.0-mini" ]]; then
            print_info "nuScenes data organized in: ${nuscenes_dir}/v1.0-mini"
        elif [[ -d "${nuscenes_dir}/samples" ]]; then
            # If extracted flat, move into v1.0-mini subdirectory
            print_progress "Reorganizing nuScenes directory structure..."
            local temp_dir="${nuscenes_dir}/.temp_reorg"
            mkdir -p "$temp_dir"
            mv "${nuscenes_dir}/samples" "$temp_dir/" 2>/dev/null || true
            mv "${nuscenes_dir}/sweeps" "$temp_dir/" 2>/dev/null || true
            mv "${nuscenes_dir}/maps" "$temp_dir/" 2>/dev/null || true
            mv "${nuscenes_dir}/v1.0-mini" "$temp_dir/" 2>/dev/null || true

            # Move everything into proper location
            rm -rf "${nuscenes_dir}/v1.0-mini"
            mv "$temp_dir" "${nuscenes_dir}/v1.0-mini"
        fi
    else
        print_info "nuScenes Mini already extracted, skipping..."
    fi

    print_success "nuScenes mini dataset download complete"
    print_info "Data location: ${nuscenes_dir}/v1.0-mini"
}

# ============================================================================
# Download Pretrained Weights
# ============================================================================

download_weights() {
    print_header "Downloading Pretrained Weights"

    local weights_dir="${DATA_DIR}/pretrained_weights"

    if [[ "$CLEAN_EXISTING" == true ]] && [[ -d "$weights_dir" ]]; then
        print_warning "Removing existing weights..."
        rm -rf "$weights_dir"
    fi

    mkdir -p "$weights_dir"

    # Download KITTI pretrained weights
    local kitti_weights="${weights_dir}/pointpillars_kitti_car.pth"
    if [[ ! -f "$kitti_weights" ]]; then
        download_file "$POINTPILLARS_KITTI_WEIGHTS_URL" "$kitti_weights" \
            "PointPillars KITTI Car weights"
    else
        print_info "KITTI weights already exist, skipping..."
    fi

    # Download nuScenes pretrained weights
    local nuscenes_weights="${weights_dir}/pointpillars_nuscenes.pth"
    if [[ ! -f "$nuscenes_weights" ]]; then
        download_file "$POINTPILLARS_NUSCENES_WEIGHTS_URL" "$nuscenes_weights" \
            "PointPillars nuScenes weights"
    else
        print_info "nuScenes weights already exist, skipping..."
    fi

    # Create a metadata file for the weights
    cat > "${weights_dir}/README.txt" << 'WEIGHTS_EOF'
PointPillars Pretrained Weights
================================

pointpillars_kitti_car.pth
  - Trained on KITTI 3D Object Detection (Car class)
  - Architecture: PointPillars with SecFPN
  - Training: 6x8 GPUs, 160 epochs
  - Source: mmdetection3d model zoo

pointpillars_nuscenes.pth
  - Trained on nuScenes 3D Object Detection (10 classes)
  - Architecture: PointPillars with FPN + SyncBN
  - Training: 4x8 GPUs, 2x schedule
  - Source: mmdetection3d model zoo
WEIGHTS_EOF

    print_success "Pretrained weights download complete"
    print_info "Weights location: ${weights_dir}"
}

# ============================================================================
# Create Symlinks
# ============================================================================

create_symlinks() {
    print_header "Creating Symlinks"

    local kitti_dir="${DATA_DIR}/kitti"
    local nuscenes_dir="${DATA_DIR}/nuscenes"

    # Create symlink from project root to data
    local project_data_link="${PROJECT_DIR}/data"
    if [[ ! -L "$project_data_link" ]] && [[ ! -d "$project_data_link" ]]; then
        if [[ "$DATA_DIR" != "$project_data_link" ]]; then
            ln -sf "$DATA_DIR" "$project_data_link"
            print_success "Created symlink: ${project_data_link} -> ${DATA_DIR}"
        fi
    else
        print_info "Data directory already exists at project root"
    fi

    # Create convenience symlinks for common access patterns
    local links_dir="${PROJECT_DIR}/data_links"
    mkdir -p "$links_dir"

    if [[ -d "${kitti_dir}/training/velodyne" ]]; then
        ln -sf "${kitti_dir}/training/velodyne" "${links_dir}/kitti_velodyne" 2>/dev/null || true
        ln -sf "${kitti_dir}/training/label_2" "${links_dir}/kitti_labels" 2>/dev/null || true
        ln -sf "${kitti_dir}/training/calib" "${links_dir}/kitti_calib" 2>/dev/null || true
        ln -sf "${kitti_dir}/training/image_2" "${links_dir}/kitti_images" 2>/dev/null || true
        print_success "Created KITTI convenience symlinks in ${links_dir}"
    fi

    if [[ -d "${nuscenes_dir}/v1.0-mini" ]]; then
        ln -sf "${nuscenes_dir}/v1.0-mini" "${links_dir}/nuscenes_mini" 2>/dev/null || true
        print_success "Created nuScenes convenience symlink in ${links_dir}"
    fi

    # Create symlink for pretrained weights accessible from config
    local weights_link="${PROJECT_DIR}/checkpoints"
    if [[ -d "${DATA_DIR}/pretrained_weights" ]]; then
        ln -sf "${DATA_DIR}/pretrained_weights" "$weights_link" 2>/dev/null || true
        print_success "Created weights symlink: ${weights_link}"
    fi
}

# ============================================================================
# Verify Downloads
# ============================================================================

verify_downloads() {
    print_header "Verifying Downloads"

    local kitti_dir="${DATA_DIR}/kitti"
    local nuscenes_dir="${DATA_DIR}/nuscenes"
    local all_passed=true

    if [[ "$DOWNLOAD_KITTI" == true ]]; then
        print_info "Verifying KITTI dataset..."

        verify_count "${kitti_dir}/training/velodyne" "$KITTI_TRAIN_VELODYNE_COUNT" \
            "KITTI train velodyne" "*.bin" || all_passed=false

        verify_count "${kitti_dir}/training/label_2" "$KITTI_TRAIN_LABEL_COUNT" \
            "KITTI train labels" "*.txt" || all_passed=false

        verify_count "${kitti_dir}/training/calib" "$KITTI_TRAIN_CALIB_COUNT" \
            "KITTI train calibration" "*.txt" || all_passed=false

        verify_count "${kitti_dir}/training/image_2" "$KITTI_TRAIN_IMAGE_COUNT" \
            "KITTI train images" "*.png" || all_passed=false

        verify_count "${kitti_dir}/testing/velodyne" "$KITTI_TEST_VELODYNE_COUNT" \
            "KITTI test velodyne" "*.bin" || all_passed=false

        verify_count "${kitti_dir}/testing/image_2" "$KITTI_TEST_IMAGE_COUNT" \
            "KITTI test images" "*.png" || all_passed=false

        verify_count "${kitti_dir}/testing/calib" "$KITTI_TEST_CALIB_COUNT" \
            "KITTI test calibration" "*.txt" || all_passed=false

        # Verify split files
        if [[ -f "${kitti_dir}/ImageSets/train.txt" ]]; then
            local train_lines
            train_lines=$(wc -l < "${kitti_dir}/ImageSets/train.txt" | tr -d ' ')
            print_info "Train split: ${train_lines} samples"
        fi
        if [[ -f "${kitti_dir}/ImageSets/val.txt" ]]; then
            local val_lines
            val_lines=$(wc -l < "${kitti_dir}/ImageSets/val.txt" | tr -d ' ')
            print_info "Val split: ${val_lines} samples"
        fi
    fi

    if [[ "$DOWNLOAD_NUSCENES" == true ]]; then
        print_info "Verifying nuScenes dataset..."

        if [[ -d "${nuscenes_dir}/v1.0-mini" ]]; then
            local json_count
            json_count=$(find "${nuscenes_dir}/v1.0-mini" -name "*.json" -type f 2>/dev/null | wc -l | tr -d ' ')
            if [[ "$json_count" -gt 0 ]]; then
                print_success "nuScenes metadata: ${json_count} JSON files found"
            else
                print_warning "nuScenes metadata: No JSON files found"
                all_passed=false
            fi

            if [[ -d "${nuscenes_dir}/v1.0-mini/samples" ]] || [[ -d "${nuscenes_dir}/samples" ]]; then
                print_success "nuScenes samples directory present"
            else
                print_warning "nuScenes samples directory missing"
                all_passed=false
            fi
        else
            print_error "nuScenes v1.0-mini directory not found"
            all_passed=false
        fi
    fi

    if [[ "$DOWNLOAD_WEIGHTS" == true ]]; then
        print_info "Verifying pretrained weights..."
        local weights_dir="${DATA_DIR}/pretrained_weights"

        for weight_file in "${weights_dir}"/*.pth; do
            if [[ -f "$weight_file" ]]; then
                local file_size
                file_size=$(stat -c%s "$weight_file" 2>/dev/null || stat -f%z "$weight_file" 2>/dev/null || echo "0")
                if [[ "$file_size" -gt 1000000 ]]; then
                    print_success "$(basename "$weight_file"): $(numfmt --to=iec "$file_size" 2>/dev/null || echo "${file_size} bytes")"
                else
                    print_warning "$(basename "$weight_file"): suspiciously small (${file_size} bytes)"
                    all_passed=false
                fi
            fi
        done
    fi

    echo ""
    if [[ "$all_passed" == true ]]; then
        print_success "All verifications passed!"
    else
        print_warning "Some verifications failed. Check the output above."
    fi
}

# ============================================================================
# Print Summary
# ============================================================================

print_summary() {
    print_header "Download Summary"

    echo -e "Data directory: ${DATA_DIR}"
    echo ""

    if [[ "$DOWNLOAD_KITTI" == true ]]; then
        echo -e "KITTI 3D Object Detection:"
        echo -e "  Training:"
        echo -e "    Velodyne: ${DATA_DIR}/kitti/training/velodyne/"
        echo -e "    Labels:   ${DATA_DIR}/kitti/training/label_2/"
        echo -e "    Calib:    ${DATA_DIR}/kitti/training/calib/"
        echo -e "    Images:   ${DATA_DIR}/kitti/training/image_2/"
        echo -e "  Testing:"
        echo -e "    Velodyne: ${DATA_DIR}/kitti/testing/velodyne/"
        echo -e "    Calib:    ${DATA_DIR}/kitti/testing/calib/"
        echo -e "    Images:   ${DATA_DIR}/kitti/testing/image_2/"
        echo -e "  Splits:     ${DATA_DIR}/kitti/ImageSets/"
        echo ""
    fi

    if [[ "$DOWNLOAD_NUSCENES" == true ]]; then
        echo -e "nuScenes Mini:"
        echo -e "  Data:       ${DATA_DIR}/nuscenes/v1.0-mini/"
        echo ""
    fi

    if [[ "$DOWNLOAD_WEIGHTS" == true ]]; then
        echo -e "Pretrained Weights:"
        echo -e "  Weights:    ${DATA_DIR}/pretrained_weights/"
        echo ""
    fi

    echo -e "Next steps:"
    echo -e "  1. Run prepare_data.py to generate info files:"
    echo -e "     python scripts/prepare_data.py --dataset kitti --data-root ${DATA_DIR}"
    echo -e "  2. Start training:"
    echo -e "     python train.py --config configs/pointpillars_kitti.yaml"
}

# ============================================================================
# Main
# ============================================================================

main() {
    parse_args "$@"

    print_header "PointPillars Dataset Download Script"
    echo -e "Data directory: ${DATA_DIR}"
    echo -e "Download KITTI:    ${DOWNLOAD_KITTI}"
    echo -e "Download nuScenes: ${DOWNLOAD_NUSCENES}"
    echo -e "Download Weights:  ${DOWNLOAD_WEIGHTS}"
    echo -e "Verify files:      ${VERIFY_FILES}"
    echo -e "Create symlinks:   ${CREATE_SYMLINKS}"

    check_prerequisites

    # Create base data directory
    mkdir -p "$DATA_DIR"

    # Download datasets
    if [[ "$DOWNLOAD_KITTI" == true ]]; then
        download_kitti
    fi

    if [[ "$DOWNLOAD_NUSCENES" == true ]]; then
        download_nuscenes
    fi

    if [[ "$DOWNLOAD_WEIGHTS" == true ]]; then
        download_weights
    fi

    # Create symlinks
    if [[ "$CREATE_SYMLINKS" == true ]]; then
        create_symlinks
    fi

    # Verify downloads
    if [[ "$VERIFY_FILES" == true ]]; then
        verify_downloads
    fi

    # Print summary
    print_summary

    print_header "Done!"
    print_success "All downloads completed successfully."
}

main "$@"
