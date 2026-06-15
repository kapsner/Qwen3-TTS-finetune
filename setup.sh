#!/usr/bin/env bash
# Qwen3-TTS Complete Setup Script
# This script sets up everything needed for the fine-tuning pipeline
#
# Requirements:
#   - Python 3.12
#   - CUDA 12.x (for GPU support)
#
# Usage:
#   ./setup.sh          # Interactive mode (prompts for model download)
#   ./setup.sh --auto   # Non-interactive mode (skips model download prompt)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
AUTO_MODE=false
if [[ "$1" == "--auto" ]]; then
    AUTO_MODE=true
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Qwen3-TTS Setup Script${NC}"
if [ "$AUTO_MODE" = true ]; then
    echo -e "${YELLOW}(Running in auto mode)${NC}"
fi
echo -e "${GREEN}========================================${NC}"
echo ""

# Pre-download models (optional - saves time during training)
echo -e "${YELLOW}Step 1: Pre-downloading models (this may take a while)...${NC}"

if [ "$AUTO_MODE" = true ]; then
    # Auto mode: skip model pre-download (will download on first run)
    echo -e "${YELLOW}Auto mode: Skipping model pre-download. Models will be downloaded during first run.${NC}"
else
    # Interactive mode: ask user
    read -p "Do you want to pre-download models now? This saves time during training. (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then

        echo -e "${GREEN}Downloading Qwen3-TTS Tokenizer...${NC}"
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3-TTS-Tokenizer-12Hz', local_dir='./models/Qwen3-TTS-Tokenizer-12Hz')
" 2>/dev/null || echo "Tokenizer download will happen during first run"

        echo -e "${GREEN}Downloading Qwen3-TTS Base Model...${NC}"
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-0.6B-Base', local_dir='./models/Qwen3-TTS-12Hz-0.6B-Base')
" 2>/dev/null || echo "Base model download will happen during first run"

        echo -e "${GREEN}Downloading WhisperX model...${NC}"
        python -c "
import whisperx
whisperx.load_model('large-v3', device='cpu')
" 2>/dev/null || echo "WhisperX model download will happen during first run"

        echo -e "${GREEN}Models downloaded to ./models/${NC}"
    else
        echo -e "${YELLOW}Skipping model pre-download. Models will be downloaded during first run.${NC}"
    fi
fi
echo ""

# Configure HuggingFace cache in venv
echo -e "${GREEN}Step 2: Configuring HuggingFace cache${NC}"
mkdir -p "./models/hf_cache"

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Configuration:"
echo -e "  HF cache: ./models/hf_cache/"
echo ""
echo -e "To run fine-tuning:"
echo -e "  ${YELLOW}./train.sh --audio_dir ./audio_files --ref_audio ./reference.wav --speaker_name my_voice${NC}"
echo ""
echo -e "To deactivate:"
echo -e "  ${YELLOW}deactivate${NC}"
echo ""
