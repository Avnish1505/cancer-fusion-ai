#!/usr/bin/env bash
# =============================================
# Cancer Fusion AI - Data Setup Script
# Downloads and prepares the HAM10000 dataset
# =============================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if kaggle CLI is installed
check_kaggle() {
    if ! command -v kaggle &> /dev/null; then
        log_error "Kaggle CLI not found. Please install it first:"
        echo "  pip install kaggle"
        echo "  Then configure your API token at ~/.kaggle/kaggle.json"
        return 1
    fi
    return 0
}

# Download dataset from Kaggle
download_dataset() {
    local dataset_name="skin-cancer-mnist-ham10000"
    local download_path="./data"
    
    log_info "Creating data directory: $download_path"
    mkdir -p "$download_path"
    
    log_info "Downloading HAM10000 dataset from Kaggle..."
    if kaggle datasets download -d kmader/$dataset_name -p "$download_path" --unzip; then
        log_info "Dataset downloaded successfully!"
        
        # Verify files exist
        if [[ -f "$download_path/HAM10000_metadata.csv" ]] && \
           [[ -d "$download_path/HAM10000_images_part_1" ]] && \
           [[ -d "$download_path/HAM10000_images_part_2" ]]; then
            log_info "Dataset files verified:"
            ls -lh "$download_path"/HAM10000_*
            return 0
        else
            log_error "Dataset files not found after extraction!"
            return 1
        fi
    else
        log_error "Failed to download dataset from Kaggle"
        return 1
    fi
}

# Create sample/mini dataset for testing/CI
create_sample_data() {
    local data_path="./data"
    local sample_path="./data_sample"
    
    log_info "Creating sample dataset for testing/CI..."
    mkdir -p "$sample_path"
    
    # Copy metadata
    if [[ -f "$data_path/HAM10000_metadata.csv" ]]; then
        # Take first 100 rows for quick testing
        head -101 "$data_path/HAM10000_metadata.csv" > "$sample_path/HAM10000_metadata.csv"
        log_info "Created sample metadata with 100 samples"
    else
        log_warn "Metadata file not found, skipping sample creation"
        return 1
    fi
    
    # Create sample image directories
    mkdir -p "$sample_path/HAM10000_images_part_1"
    mkdir -p "$sample_path/HAM10000_images_part_2"
    
    # Copy a few sample images if they exist
    if [[ -d "$data_path/HAM10000_images_part_1" ]]; then
        # Copy first 5 images from part_1
        ls "$data_path/HAM10000_images_part_1" | head -5 | while read img; do
            cp "$data_path/HAM10000_images_part_1/$img" "$sample_path/HAM10000_images_part_1/" 2>/dev/null || true
        done
    fi
    
    if [[ -d "$data_path/HAM10000_images_part_2" ]]; then
        # Copy first 5 images from part_2
        ls "$data_path/HAM10000_images_part_2" | head -5 | while read img; do
            cp "$data_path/HAM10000_images_part_2/$img" "$sample_path/HAM10000_images_part_2/" 2>/dev/null || true
        done
    fi
    
    log_info "Sample dataset created at: $sample_path"
    log_info "To use sample data, set DATA_PATH=$sample_path"
}

# Main execution
main() {
    log_info "Cancer Fusion AI - Data Setup"
    
    # Check arguments
    if [[ "${1:-}" == "--sample" ]]; then
        if check_kaggle && download_dataset; then
            create_sample_data
        fi
        exit 0
    fi
    
    # Download full dataset
    if check_kaggle && download_dataset; then
        log_info "Setup complete! You can now run training."
        log_info "To use sample data for testing: DATA_PATH=./data_sample python -m src.train --config configs/config.yaml"
    else
        log_error "Setup failed. Please check the errors above."
        exit 1
    fi
}

main "$@"