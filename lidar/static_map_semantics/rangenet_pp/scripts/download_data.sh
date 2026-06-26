#!/bin/bash
#
# download_data.sh - Download SemanticKITTI dataset for RangeNet++
#
# Downloads:
#   1. KITTI odometry velodyne point clouds (sequences 00-21)
#   2. KITTI odometry calibration files
#   3. SemanticKITTI labels
#
# Usage: ./download_data.sh [-d TARGET_DIR] [-k] [-h]

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

VELODYNE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
CALIB_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip"
LABELS_URL="http://www.semantic-kitti.org/assets/data_odometry_labels.zip"

DEFAULT_TARGET_DIR="./dataset"
KEEP_ZIPS=false
TARGET_DIR=""

# Expected minimum file counts per sequence for velodyne scans
declare -A EXPECTED_COUNTS=(
    ["00"]=4541 ["01"]=1101 ["02"]=4661 ["03"]=801 ["04"]=271
    ["05"]=2761 ["06"]=1101 ["07"]=1101 ["08"]=4071 ["09"]=1591
    ["10"]=1201 ["11"]=921  ["12"]=1061 ["13"]=3281 ["14"]=631
    ["15"]=1901 ["16"]=1731 ["17"]=491  ["18"]=1801 ["19"]=4981
    ["20"]=831  ["21"]=2721
)

# Sequences that have SemanticKITTI labels (00-10 are training, 11-21 are test)
LABEL_SEQUENCES=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10")

# =============================================================================
# Functions
# =============================================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Download SemanticKITTI dataset (KITTI odometry velodyne, calibration, and labels).

Options:
  -d DIR    Target directory for dataset (default: ${DEFAULT_TARGET_DIR})
  -k        Keep downloaded zip files after extraction
  -h        Show this help message and exit

The script will create the following directory structure:
  <TARGET_DIR>/
    sequences/
      00/
        velodyne/     (*.bin point cloud files)
        labels/       (*.label semantic label files, sequences 00-10 only)
        calib.txt
      01/
        ...
      ...
      21/
        ...

Requirements:
  - wget or curl (for downloading)
  - unzip (for extraction)
  - ~80 GB free disk space (velodyne ~65GB, labels ~6GB, calib ~30MB)

Examples:
  $(basename "$0")                      # Download to ./dataset
  $(basename "$0") -d /data/kitti       # Download to /data/kitti
  $(basename "$0") -d /data/kitti -k    # Download and keep zip files

EOF
}

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_warn() {
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

# Detect available download tool
detect_downloader() {
    if command -v wget &>/dev/null; then
        echo "wget"
    elif command -v curl &>/dev/null; then
        echo "curl"
    else
        log_error "Neither wget nor curl found. Please install one of them."
        exit 1
    fi
}

# Download a file with progress bar
# Arguments: $1 = URL, $2 = output file, $3 = downloader tool
download_file() {
    local url="$1"
    local output="$2"
    local downloader="$3"

    log_info "Downloading: $(basename "$output")"
    log_info "  URL: ${url}"
    log_info "  Destination: ${output}"

    if [[ -f "$output" ]]; then
        log_warn "File already exists: ${output}. Skipping download."
        return 0
    fi

    local ret=0
    if [[ "$downloader" == "wget" ]]; then
        wget --progress=bar:force:noscroll -O "$output" "$url" || ret=$?
    else
        curl -L --progress-bar -o "$output" "$url" || ret=$?
    fi

    if [[ $ret -ne 0 ]]; then
        log_error "Download failed for: ${url} (exit code: ${ret})"
        rm -f "$output"
        return 1
    fi

    if [[ ! -s "$output" ]]; then
        log_error "Downloaded file is empty: ${output}"
        rm -f "$output"
        return 1
    fi

    log_info "Download complete: $(basename "$output") ($(du -h "$output" | cut -f1))"
    return 0
}

# Extract a zip file
# Arguments: $1 = zip file, $2 = destination directory
extract_zip() {
    local zipfile="$1"
    local destdir="$2"

    log_info "Extracting: $(basename "$zipfile") -> ${destdir}"

    if ! unzip -o -q "$zipfile" -d "$destdir"; then
        log_error "Extraction failed for: ${zipfile}"
        return 1
    fi

    log_info "Extraction complete: $(basename "$zipfile")"
    return 0
}

# Organize extracted files into the expected directory structure
organize_files() {
    local target="$1"
    local sequences_dir="${target}/sequences"

    log_info "Organizing files into target structure..."

    # The KITTI odometry zip files extract to dataset/sequences/XX/
    # Check if the files were extracted with a different root
    # Typical extraction paths:
    #   data_odometry_velodyne.zip -> dataset/sequences/XX/velodyne/XXXXXX.bin
    #   data_odometry_calib.zip   -> dataset/sequences/XX/calib.txt
    #   data_odometry_labels.zip  -> dataset/sequences/XX/labels/XXXXXX.label

    # Handle velodyne data - may extract to "dataset/sequences/" structure
    if [[ -d "${target}/dataset/sequences" ]]; then
        log_info "Moving extracted data from nested 'dataset' directory..."
        # Move contents from nested dataset/sequences to target/sequences
        mkdir -p "$sequences_dir"
        for seq_dir in "${target}/dataset/sequences/"*/; do
            if [[ -d "$seq_dir" ]]; then
                local seq_name
                seq_name=$(basename "$seq_dir")
                if [[ -d "${sequences_dir}/${seq_name}" ]]; then
                    # Merge contents
                    cp -rn "$seq_dir"* "${sequences_dir}/${seq_name}/" 2>/dev/null || true
                else
                    mv "$seq_dir" "${sequences_dir}/${seq_name}"
                fi
            fi
        done
        rm -rf "${target}/dataset"
    fi

    # Handle if extracted to "data_odometry_velodyne/dataset/sequences/" etc.
    for prefix in data_odometry_velodyne data_odometry_calib data_odometry_labels; do
        if [[ -d "${target}/${prefix}/dataset/sequences" ]]; then
            log_info "Moving extracted data from '${prefix}/dataset/sequences'..."
            mkdir -p "$sequences_dir"
            for seq_dir in "${target}/${prefix}/dataset/sequences/"*/; do
                if [[ -d "$seq_dir" ]]; then
                    local seq_name
                    seq_name=$(basename "$seq_dir")
                    mkdir -p "${sequences_dir}/${seq_name}"
                    cp -rn "$seq_dir"* "${sequences_dir}/${seq_name}/" 2>/dev/null || true
                fi
            done
            rm -rf "${target}/${prefix}"
        elif [[ -d "${target}/${prefix}/sequences" ]]; then
            log_info "Moving extracted data from '${prefix}/sequences'..."
            mkdir -p "$sequences_dir"
            for seq_dir in "${target}/${prefix}/sequences/"*/; do
                if [[ -d "$seq_dir" ]]; then
                    local seq_name
                    seq_name=$(basename "$seq_dir")
                    mkdir -p "${sequences_dir}/${seq_name}"
                    cp -rn "$seq_dir"* "${sequences_dir}/${seq_name}/" 2>/dev/null || true
                fi
            done
            rm -rf "${target}/${prefix}"
        fi
    done

    # Ensure all sequence directories exist
    for seq in $(printf '%02d\n' $(seq 0 21)); do
        mkdir -p "${sequences_dir}/${seq}/velodyne"
    done

    log_info "File organization complete."
}

# Verify downloaded and extracted data
verify_data() {
    local target="$1"
    local sequences_dir="${target}/sequences"
    local all_ok=true
    local total_velodyne=0
    local total_labels=0

    log_info "Verifying dataset integrity..."
    echo ""
    printf "%-10s %-15s %-15s %-10s\n" "Sequence" "Velodyne" "Labels" "Status"
    printf "%-10s %-15s %-15s %-10s\n" "--------" "--------" "------" "------"

    for seq in $(printf '%02d\n' $(seq 0 21)); do
        local vel_dir="${sequences_dir}/${seq}/velodyne"
        local lbl_dir="${sequences_dir}/${seq}/labels"
        local vel_count=0
        local lbl_count=0
        local status="OK"

        if [[ -d "$vel_dir" ]]; then
            vel_count=$(find "$vel_dir" -name "*.bin" -type f 2>/dev/null | wc -l)
        fi

        if [[ -d "$lbl_dir" ]]; then
            lbl_count=$(find "$lbl_dir" -name "*.label" -type f 2>/dev/null | wc -l)
        fi

        total_velodyne=$((total_velodyne + vel_count))
        total_labels=$((total_labels + lbl_count))

        # Check velodyne count against expected
        local expected=${EXPECTED_COUNTS[$seq]:-0}
        if [[ $vel_count -lt $expected ]]; then
            status="INCOMPLETE"
            all_ok=false
        fi

        # Check labels for sequences 00-10
        local is_label_seq=false
        for ls in "${LABEL_SEQUENCES[@]}"; do
            if [[ "$seq" == "$ls" ]]; then
                is_label_seq=true
                break
            fi
        done

        if [[ "$is_label_seq" == true && $lbl_count -lt $expected ]]; then
            status="INCOMPLETE"
            all_ok=false
        fi

        local lbl_display="${lbl_count}"
        if [[ "$is_label_seq" == false ]]; then
            lbl_display="${lbl_count} (test)"
        fi

        printf "%-10s %-15s %-15s %-10s\n" "$seq" "$vel_count" "$lbl_display" "$status"
    done

    echo ""
    log_info "Total velodyne scans: ${total_velodyne}"
    log_info "Total label files: ${total_labels}"

    if [[ "$all_ok" == true ]]; then
        log_info "Verification PASSED: All sequences have expected file counts."
    else
        log_warn "Verification WARNING: Some sequences have fewer files than expected."
        log_warn "This may indicate incomplete downloads or a different dataset version."
    fi

    return 0
}

# Print final summary
print_summary() {
    local target="$1"
    local elapsed="$2"

    echo ""
    echo "============================================================================="
    echo "  SemanticKITTI Dataset Download Summary"
    echo "============================================================================="
    echo ""
    echo "  Target directory : $(realpath "$target")"
    echo "  Time elapsed     : ${elapsed} seconds"
    echo ""
    echo "  Downloaded files:"
    echo "    - KITTI odometry velodyne point clouds (sequences 00-21)"
    echo "    - KITTI odometry calibration files"
    echo "    - SemanticKITTI semantic labels (sequences 00-10)"
    echo ""
    echo "  Directory structure:"
    echo "    ${target}/sequences/XX/velodyne/*.bin"
    echo "    ${target}/sequences/XX/labels/*.label  (sequences 00-10)"
    echo "    ${target}/sequences/XX/calib.txt"
    echo ""
    if [[ "$KEEP_ZIPS" == true ]]; then
        echo "  Zip files kept in: ${target}/"
    else
        echo "  Zip files cleaned up."
    fi
    echo ""
    echo "  Dataset size:"
    du -sh "$target" 2>/dev/null | awk '{print "    Total: " $1}'
    echo ""
    echo "============================================================================="
    echo ""
}

# =============================================================================
# Main
# =============================================================================

main() {
    local start_time
    start_time=$(date +%s)

    # Parse command-line arguments
    while getopts ":d:kh" opt; do
        case $opt in
            d)
                TARGET_DIR="$OPTARG"
                ;;
            k)
                KEEP_ZIPS=true
                ;;
            h)
                usage
                exit 0
                ;;
            \?)
                log_error "Invalid option: -$OPTARG"
                usage
                exit 1
                ;;
            :)
                log_error "Option -$OPTARG requires an argument."
                usage
                exit 1
                ;;
        esac
    done

    # Set target directory
    if [[ -z "$TARGET_DIR" ]]; then
        TARGET_DIR="$DEFAULT_TARGET_DIR"
    fi

    echo ""
    echo "============================================================================="
    echo "  SemanticKITTI Dataset Downloader"
    echo "============================================================================="
    echo ""
    log_info "Target directory: ${TARGET_DIR}"
    log_info "Keep zip files: ${KEEP_ZIPS}"
    echo ""

    # Check prerequisites
    log_info "Checking prerequisites..."

    local downloader
    downloader=$(detect_downloader)
    log_info "Download tool: ${downloader}"

    if ! command -v unzip &>/dev/null; then
        log_error "'unzip' is not installed. Please install it first."
        exit 1
    fi
    log_info "Extraction tool: unzip"

    # Check available disk space (rough estimate: need ~80GB)
    local available_space
    available_space=$(df -BG "$(dirname "$TARGET_DIR")" 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G' || echo "unknown")
    if [[ "$available_space" != "unknown" && "$available_space" -lt 80 ]]; then
        log_warn "Low disk space detected: ${available_space}GB available (recommended: 80GB+)"
        log_warn "Continue anyway? Downloads are large (~80GB total)"
        read -r -p "Press Enter to continue or Ctrl+C to abort..."
    fi

    # Create target directory
    mkdir -p "$TARGET_DIR"

    # =========================================================================
    # Step 1: Download all zip files
    # =========================================================================

    log_info "=========================================="
    log_info "Step 1/4: Downloading data files..."
    log_info "=========================================="

    local velodyne_zip="${TARGET_DIR}/data_odometry_velodyne.zip"
    local calib_zip="${TARGET_DIR}/data_odometry_calib.zip"
    local labels_zip="${TARGET_DIR}/data_odometry_labels.zip"

    # Download velodyne data (largest file, ~65GB)
    download_file "$VELODYNE_URL" "$velodyne_zip" "$downloader"

    # Download calibration data (~30MB)
    download_file "$CALIB_URL" "$calib_zip" "$downloader"

    # Download SemanticKITTI labels (~6GB)
    download_file "$LABELS_URL" "$labels_zip" "$downloader"

    # =========================================================================
    # Step 2: Extract zip files
    # =========================================================================

    log_info "=========================================="
    log_info "Step 2/4: Extracting archives..."
    log_info "=========================================="

    extract_zip "$velodyne_zip" "$TARGET_DIR"
    extract_zip "$calib_zip" "$TARGET_DIR"
    extract_zip "$labels_zip" "$TARGET_DIR"

    # =========================================================================
    # Step 3: Organize directory structure
    # =========================================================================

    log_info "=========================================="
    log_info "Step 3/4: Organizing directory structure..."
    log_info "=========================================="

    organize_files "$TARGET_DIR"

    # =========================================================================
    # Step 4: Clean up zip files (unless -k flag set)
    # =========================================================================

    if [[ "$KEEP_ZIPS" == false ]]; then
        log_info "=========================================="
        log_info "Step 4/4: Cleaning up zip files..."
        log_info "=========================================="

        rm -f "$velodyne_zip"
        rm -f "$calib_zip"
        rm -f "$labels_zip"
        log_info "Zip files removed."
    else
        log_info "=========================================="
        log_info "Step 4/4: Keeping zip files as requested."
        log_info "=========================================="
    fi

    # =========================================================================
    # Verification and Summary
    # =========================================================================

    verify_data "$TARGET_DIR"

    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    print_summary "$TARGET_DIR" "$elapsed"

    log_info "Done! Dataset is ready for use with RangeNet++."
}

# Run main function
main "$@"
