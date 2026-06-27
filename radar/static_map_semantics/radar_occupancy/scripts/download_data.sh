# [IMPLEMENTED BY CLAUDE - was missing]
#!/bin/bash
set -e

# =============================================================================
# download_data.sh - Download and prepare nuScenes radar data
# =============================================================================
# This script guides the user through downloading the nuScenes dataset for
# radar-based occupancy grid prediction. Since nuScenes is a proprietary dataset
# requiring registration, this script provides instructions and automates the
# directory setup and post-download processing steps.
#
# Usage:
#   ./download_data.sh [--version VERSION] [--data-root PATH]
#
# Examples:
#   ./download_data.sh
#   ./download_data.sh --version v1.0-trainval --data-root /mnt/data/nuscenes
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration Variables
# -----------------------------------------------------------------------------
NUSCENES_VERSION="${NUSCENES_VERSION:-v1.0-mini}"
DATA_ROOT="${DATA_ROOT:-./data/nuscenes}"
DOWNLOAD_URL="https://www.nuscenes.org/data/"

# Color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

print_header() {
    echo ""
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
    echo ""
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Download and prepare nuScenes radar data for occupancy grid prediction."
    echo ""
    echo "Options:"
    echo "  --version VERSION    nuScenes version to download (default: v1.0-mini)"
    echo "                       Available: v1.0-mini, v1.0-trainval, v1.0-test"
    echo "  --data-root PATH     Root directory for data storage (default: ./data/nuscenes)"
    echo "  --help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  NUSCENES_VERSION     Same as --version"
    echo "  DATA_ROOT            Same as --data-root"
    echo ""
    echo "Note: nuScenes requires free registration at https://www.nuscenes.org/"
    echo "      You must accept the Terms of Use before downloading."
}

# -----------------------------------------------------------------------------
# Parse Command Line Arguments
# -----------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            NUSCENES_VERSION="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Print Usage Information
# -----------------------------------------------------------------------------

print_header "nuScenes Radar Data Download Script"

echo "Configuration:"
echo "  Version:       ${NUSCENES_VERSION}"
echo "  Data Root:     ${DATA_ROOT}"
echo "  Download URL:  ${DOWNLOAD_URL}"
echo ""

# -----------------------------------------------------------------------------
# Create Directory Structure
# -----------------------------------------------------------------------------

print_header "Step 1: Creating Directory Structure"

DIRS=(
    "${DATA_ROOT}/radar"
    "${DATA_ROOT}/lidar"
    "${DATA_ROOT}/annotations"
    "${DATA_ROOT}/maps"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_info "Created directory: $dir"
    else
        print_info "Directory already exists: $dir"
    fi
done

# Create a subdirectory for raw downloads
mkdir -p "${DATA_ROOT}/downloads"
print_info "Created downloads directory: ${DATA_ROOT}/downloads"

# -----------------------------------------------------------------------------
# Check if Data Already Exists
# -----------------------------------------------------------------------------

print_header "Step 2: Checking for Existing Data"

DATA_EXISTS=false

if [ -d "${DATA_ROOT}/radar" ] && [ "$(ls -A ${DATA_ROOT}/radar 2>/dev/null)" ]; then
    print_warn "Radar data already exists in ${DATA_ROOT}/radar/"
    DATA_EXISTS=true
fi

if [ -f "${DATA_ROOT}/annotations/instances.json" ] || \
   [ -f "${DATA_ROOT}/annotations/sample_annotation.json" ]; then
    print_warn "Annotation files already exist in ${DATA_ROOT}/annotations/"
    DATA_EXISTS=true
fi

if [ "$DATA_EXISTS" = true ]; then
    echo ""
    print_warn "Some data already exists. Re-running may overwrite existing files."
    read -p "Do you want to continue? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Aborted by user."
        exit 0
    fi
fi

# -----------------------------------------------------------------------------
# Instructions for Manual Download
# -----------------------------------------------------------------------------

print_header "Step 3: Download Instructions"

echo -e "${YELLOW}=================================================================${NC}"
echo -e "${YELLOW}  MANUAL DOWNLOAD REQUIRED${NC}"
echo -e "${YELLOW}=================================================================${NC}"
echo ""
echo "nuScenes is a proprietary dataset that requires registration."
echo "Please follow these steps to download the data:"
echo ""
echo "  1. Visit: ${DOWNLOAD_URL}"
echo "  2. Create a free account or log in"
echo "  3. Accept the Terms of Use"
echo "  4. Download the following files for version '${NUSCENES_VERSION}':"
echo ""

case $NUSCENES_VERSION in
    v1.0-mini)
        echo "     - nuScenes-mini (metadata + sensor data): ~4 GB"
        echo "       File: v1.0-mini.tgz"
        ;;
    v1.0-trainval)
        echo "     - Metadata: v1.0-trainval_meta.tgz (~300 MB)"
        echo "     - File blobs (10 parts):"
        echo "       v1.0-trainval01_blobs.tgz through v1.0-trainval10_blobs.tgz"
        echo "       Total: ~300 GB"
        echo ""
        echo "     For radar-only experiments, you may only need:"
        echo "     - Metadata + radar blobs (significantly smaller)"
        ;;
    v1.0-test)
        echo "     - Metadata: v1.0-test_meta.tgz"
        echo "     - File blobs: v1.0-test_blobs.tgz"
        ;;
    *)
        print_warn "Unknown version '${NUSCENES_VERSION}'. Check the nuScenes website for available files."
        ;;
esac

echo ""
echo "  5. Place the downloaded .tgz files in:"
echo "     ${DATA_ROOT}/downloads/"
echo ""
echo "  6. Re-run this script to extract and process the data."
echo ""
echo -e "${YELLOW}=================================================================${NC}"
echo ""

# Wait for user confirmation before proceeding
read -p "Have you downloaded the files to ${DATA_ROOT}/downloads/? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Please download the files and re-run this script."
    print_info "Directory structure has been created. You can place files in:"
    print_info "  ${DATA_ROOT}/downloads/"
    exit 0
fi

# -----------------------------------------------------------------------------
# Extract Downloaded Files
# -----------------------------------------------------------------------------

print_header "Step 4: Extracting Downloaded Files"

DOWNLOAD_DIR="${DATA_ROOT}/downloads"
EXTRACT_COUNT=0

if [ -d "$DOWNLOAD_DIR" ]; then
    for archive in "${DOWNLOAD_DIR}"/*.tgz "${DOWNLOAD_DIR}"/*.tar.gz; do
        if [ -f "$archive" ]; then
            print_info "Extracting: $(basename $archive)"
            tar -xzf "$archive" -C "${DATA_ROOT}/" --strip-components=0
            EXTRACT_COUNT=$((EXTRACT_COUNT + 1))
        fi
    done
fi

if [ $EXTRACT_COUNT -eq 0 ]; then
    print_warn "No .tgz or .tar.gz files found in ${DOWNLOAD_DIR}/"
    print_warn "Please ensure downloaded files are placed in that directory."
    exit 1
else
    print_info "Successfully extracted ${EXTRACT_COUNT} archive(s)."
fi

# Organize radar data into expected structure
# nuScenes extracts with a specific directory layout; symlink or move as needed
if [ -d "${DATA_ROOT}/samples/RADAR_FRONT" ]; then
    print_info "Organizing radar sweep data..."
    # Link all radar channels into the radar directory
    for channel in RADAR_FRONT RADAR_FRONT_LEFT RADAR_FRONT_RIGHT RADAR_BACK_LEFT RADAR_BACK_RIGHT; do
        if [ -d "${DATA_ROOT}/samples/${channel}" ]; then
            ln -sfn "${DATA_ROOT}/samples/${channel}" "${DATA_ROOT}/radar/${channel}" 2>/dev/null || true
            print_info "  Linked: ${channel}"
        fi
    done
fi

# Organize lidar data (used for ground truth generation)
if [ -d "${DATA_ROOT}/samples/LIDAR_TOP" ]; then
    print_info "Organizing lidar data for GT generation..."
    ln -sfn "${DATA_ROOT}/samples/LIDAR_TOP" "${DATA_ROOT}/lidar/LIDAR_TOP" 2>/dev/null || true
fi

# Organize annotation data
if [ -d "${DATA_ROOT}/${NUSCENES_VERSION}" ]; then
    print_info "Organizing annotation metadata..."
    cp -n "${DATA_ROOT}/${NUSCENES_VERSION}"/*.json "${DATA_ROOT}/annotations/" 2>/dev/null || true
fi

# Organize map data
if [ -d "${DATA_ROOT}/maps" ] && [ -d "${DATA_ROOT}/maps/expansion" ]; then
    print_info "Map data already in expected location."
fi

# -----------------------------------------------------------------------------
# Generate Radar Pillar Format Data
# -----------------------------------------------------------------------------

print_header "Step 5: Generating Radar Pillar Format Data"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_data.py"

if [ -f "$PREPARE_SCRIPT" ]; then
    print_info "Running data preparation script..."
    print_info "Command: python ${PREPARE_SCRIPT} --data-root ${DATA_ROOT} --version ${NUSCENES_VERSION}"
    echo ""

    python "${PREPARE_SCRIPT}" \
        --data-root "${DATA_ROOT}" \
        --version "${NUSCENES_VERSION}" \
        --output-dir "${DATA_ROOT}/radar_pillar"

    if [ $? -eq 0 ]; then
        print_info "Radar pillar format data generated successfully."
    else
        print_error "Data preparation failed. Check the error messages above."
        print_error "You can re-run manually with:"
        print_error "  python ${PREPARE_SCRIPT} --data-root ${DATA_ROOT} --version ${NUSCENES_VERSION}"
        exit 1
    fi
else
    print_warn "prepare_data.py not found at: ${PREPARE_SCRIPT}"
    print_warn "Skipping radar pillar format generation."
    print_warn "You can run it manually later once the script is available:"
    print_warn "  python prepare_data.py --data-root ${DATA_ROOT} --version ${NUSCENES_VERSION}"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

print_header "Summary"

echo "Download and preparation complete!"
echo ""
echo "Directory structure:"
echo "  ${DATA_ROOT}/"
echo "  ├── radar/           - Radar point cloud data (5 channels)"
echo "  ├── lidar/           - LiDAR data (for ground truth generation)"
echo "  ├── annotations/     - Scene metadata and annotations"
echo "  ├── maps/            - HD map data"
echo "  ├── radar_pillar/    - Processed radar pillar format (if generated)"
echo "  └── downloads/       - Original downloaded archives"
echo ""

# Print file counts
echo "File counts:"
for dir in radar lidar annotations maps; do
    if [ -d "${DATA_ROOT}/${dir}" ]; then
        count=$(find "${DATA_ROOT}/${dir}" -type f 2>/dev/null | wc -l)
        echo "  ${dir}/: ${count} files"
    fi
done

echo ""
echo "Next steps:"
echo "  1. Verify the data integrity"
echo "  2. Run training with: python train.py --config configs/radar_occupancy.yaml"
echo "  3. See README.md for full documentation"
echo ""
print_info "Done!"
