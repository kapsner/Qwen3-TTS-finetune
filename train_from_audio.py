#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-TTS One-Command Fine-Tuning Script

This script provides a complete end-to-end fine-tuning pipeline:
1. Takes a directory of WAV files and a reference audio
2. Automatically transcribes using WhisperX
3. Creates the train_raw.jsonl
4. Prepares data (extracts audio_codes)
5. Trains the model

Usage:
    python train_from_audio.py \
        --audio_dir ./my_audio_files \
        --ref_audio ./reference.wav \
        --speaker_name my_voice \
        --output_dir ./output
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import gc
import time
from pathlib import Path
from typing import List, Dict, Any


def configure_hf_cache():
    """Configure HuggingFace cache inside venv if available."""
    script_dir = Path(__file__).parent.absolute()
    hf_cache = "/opt/models/hf_cache"

    # Only set HF_HOME - let HuggingFace manage subdirectories
    os.environ.setdefault("HF_HOME", str(hf_cache))


def get_attention_implementation():
    """Return best available attention implementation."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "eager"


# Configure HF cache before any HuggingFace imports
configure_hf_cache()

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torchaudio
from tqdm import tqdm


class Qwen3TTSPipeline:
    """End-to-end pipeline for Qwen3-TTS fine-tuning."""

    def __init__(
        self,
        audio_dir: str,
        ref_audio: str,
        speaker_name: str,
        output_dir: str = "./output",
        device: str = "cuda:0",
        tokenizer_model_path: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        init_model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        batch_size: int = 2,
        lr: float = 2e-5,
        num_epochs: int = 3,
        whisper_model: str = "large-v3",
        whisper_compute_type: str = "float16",
        language: str = "en",
    ):
        self.audio_dir = Path(audio_dir)
        self.ref_audio = Path(ref_audio)
        self.speaker_name = speaker_name
        self.output_dir = Path(output_dir)
        self.device = device
        self.tokenizer_model_path = tokenizer_model_path
        self.init_model_path = init_model_path
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.whisper_model = whisper_model
        self.whisper_compute_type = whisper_compute_type
        self.language = language

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Intermediate files
        self.train_raw_jsonl = self.output_dir / "train_raw.jsonl"
        self.train_with_codes_jsonl = self.output_dir / "train_with_codes.jsonl"

        # Detect attention implementation
        self.attn_implementation = get_attention_implementation()

    def validate_audio_files(self) -> List[Path]:
        """Find and validate all WAV files in the audio directory."""
        if not self.audio_dir.exists():
            raise ValueError(f"Audio directory not found: {self.audio_dir}")

        wav_files = list(self.audio_dir.glob("*.wav")) + list(self.audio_dir.glob("*.WAV"))

        if not wav_files:
            raise ValueError(f"No WAV files found in {self.audio_dir}")

        if not self.ref_audio.exists():
            raise ValueError(f"Reference audio not found: {self.ref_audio}")

        # Validate audio files can be loaded
        valid_files = []
        for wav_path in tqdm(wav_files, desc="Validating audio files"):
            try:
                torchaudio.load(str(wav_path))
                valid_files.append(wav_path)
            except Exception as e:
                print(f"Warning: Could not load {wav_path}: {e}")

        print(f"Found {len(valid_files)} valid audio files")
        return valid_files

    def check_dependencies(self) -> None:
        """Check and install all required dependencies."""
        print(f"\n{'='*60}")
        print("Checking and installing dependencies...")
        print(f"{'='*60}\n")

        # Core dependencies
        core_packages = [
            ("torch", "torch"),
            ("torchaudio", "torchaudio"),
            ("numpy", "numpy"),
            ("librosa", "librosa"),
            ("soundfile", "soundfile"),
            ("tqdm", "tqdm"),
        ]

        # ML/TTS dependencies
        ml_packages = [
            ("transformers", "transformers"),
            ("accelerate", "accelerate"),
            ("safetensors", "safetensors"),
            ("huggingface_hub", "huggingface-hub"),
            ("hf_transfer", "hf_transfer"),
        ]

        # Audio processing
        audio_packages = [
            ("whisperx", "whisperx"),
            ("qwen_tts", "qwen-tts"),
        ]

        all_packages = core_packages + ml_packages + audio_packages

        for module_name, package_name in all_packages:
            try:
                __import__(module_name)
                print(f"  {module_name} is installed")
            except ImportError:
                print(f"  Installing {package_name}...")
                subprocess.check_call([
                    "uv", "pip", "install", package_name, "--system", "--break-system-packages"
                ])
                print(f"  {package_name} installed")

        # Check flash_attn (optional)
        try:
            import flash_attn  # noqa: F401
            print(f"  flash_attn is installed (using flash_attention_2)")
        except ImportError:
            print(f"  flash_attn not available (using eager attention - slower but compatible)")

        print("\nAll dependencies are ready!")
        print(f"Attention implementation: {self.attn_implementation}")
        print()

    def transcribe_with_whisperx(self, audio_files: List[Path]) -> List[Dict[str, Any]]:
        """
        Transcribe audio files using WhisperX.

        Returns a list of dictionaries with audio path and transcription.
        """
        # Fix for PyTorch 2.6+ compatibility with pyannote-audio
        # pyannote models use omegaconf which requires weights_only=False
        _original_torch_load = torch.load
        def _patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False  # Force override
            return _original_torch_load(*args, **kwargs)
        torch.load = _patched_torch_load

        import whisperx

        print(f"\n{'='*60}")
        print("STEP 1: Transcribing audio files with WhisperX")
        print(f"{'='*60}\n")

        # Load WhisperX model
        print(f"Loading WhisperX model: {self.whisper_model}")
        device = "cuda" if self.device.startswith("cuda") else "cpu"
        time.sleep(3)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        model = whisperx.load_model(
            self.whisper_model,
            device=device,
            compute_type=self.whisper_compute_type,
        )

        results = []

        for audio_path in tqdm(audio_files, desc="Transcribing"):
            try:
                # Transcribe
                audio = whisperx.load_audio(str(audio_path))
                result = model.transcribe(
                    audio,
                    batch_size=16 if device == "cuda" else 1,
                    language=self.language if self.language != "auto" else None,
                )

                # Get the text
                text = result["segments"][0]["text"].strip()

                results.append({
                    "audio": str(audio_path),
                    "text": text,
                    "ref_audio": str(self.ref_audio),
                })

                print(f"  {audio_path.name}: {text[:100]}...")

            except Exception as e:
                print(f"Warning: Failed to transcribe {audio_path}: {e}")
                continue

        # Cleanup model
        del model
        time.sleep(3)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print(f"\nSuccessfully transcribed {len(results)} files")
        return results

    def create_train_jsonl(self, data: List[Dict[str, Any]]) -> None:
        """Create the train_raw.jsonl file."""
        print(f"\n{'='*60}")
        print("STEP 2: Creating train_raw.jsonl")
        print(f"{'='*60}\n")

        with open(self.train_raw_jsonl, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Created {self.train_raw_jsonl} with {len(data)} entries")

    def prepare_data(self) -> None:
        """Run the data preparation step to extract audio_codes."""
        print(f"\n{'='*60}")
        print("STEP 3: Preparing data (extracting audio_codes)")
        print(f"{'='*60}\n")

        # Import here to ensure qwen-tts is installed
        from qwen_tts import Qwen3TTSTokenizer

        # Load tokenizer
        time.sleep(3)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print(f"Loading tokenizer: {self.tokenizer_model_path}")
        tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
            self.tokenizer_model_path,
            dtype=torch.bfloat16,
            attn_implementation=self.attn_implementation,
            device_map="auto",
        )

        # Read and process JSONL
        total_lines = open(self.train_raw_jsonl).readlines()
        total_lines = [json.loads(line.strip()) for line in total_lines]

        final_lines = []
        batch_lines = []
        batch_audios = []
        BATCH_INFER_NUM = 32

        print(f"Processing {len(total_lines)} audio files...")

        for line in tqdm(total_lines, desc="Encoding audio"):
            batch_lines.append(line)
            batch_audios.append(line["audio"])

            if len(batch_lines) >= BATCH_INFER_NUM:
                enc_res = tokenizer_12hz.encode(batch_audios)
                for code, line in zip(enc_res.audio_codes, batch_lines):
                    line["audio_codes"] = code.cpu().tolist()
                    final_lines.append(line)
                batch_lines.clear()
                batch_audios.clear()

            time.sleep(3)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        # Process remaining
        if len(batch_audios) > 0:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, line in zip(enc_res.audio_codes, batch_lines):
                line["audio_codes"] = code.cpu().tolist()
                final_lines.append(line)

        # Cleanup
        del tokenizer_12hz
        time.sleep(3)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Write output
        final_lines = [json.dumps(line, ensure_ascii=False) for line in final_lines]
        with open(self.train_with_codes_jsonl, "w", encoding="utf-8") as f:
            for line in final_lines:
                f.writelines(line + "\n")

        print(f"Created {self.train_with_codes_jsonl}")

    def train_model(self) -> None:
        """Run the fine-tuning step."""
        print(f"\n{'='*60}")
        print("STEP 4: Fine-tuning model")
        print(f"{'='*60}\n")
        
        # Get the actual model path (could be HF cache or local)
        from huggingface_hub import snapshot_download
        if os.path.isdir(self.init_model_path):
            model_cache_path = self.init_model_path
        else:
            # Download/get cached path for HuggingFace model
            model_cache_path = snapshot_download(self.init_model_path)

        # modify config to fit qwentts-0.6B
        input_config_file = os.path.join(model_cache_path, "config.json")
        with open(input_config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config_dict["talker_config"]["text_hidden_size"] = 2048
        # config_dict["talker_config"]["text_hidden_size"] = 1024
        config_dict["speaker_encoder_config"]["enc_dim"] = 1024
        with open(input_config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        # Import training module
        import sft_12hz

        # 1. Back up the original sys.argv
        original_argv = sys.argv
        try:
            sys.argv = [
                "sft_12hz.py",
                "--init_model_path",
                str(model_cache_path),
                "--output_model_path",
                str(self.output_dir),
                "--train_jsonl",
                str(self.train_with_codes_jsonl),
                "--batch_size",
                str(self.batch_size),
                "--lr",
                str(self.lr),
                "--num_epochs",
                str(self.num_epochs),
                "--speaker_name",
                str(self.speaker_name),
                "--attn_implementation",
                str(self.attn_implementation),
            ]
            # Call the module's main function
            sft_12hz.train()
        finally:
            # Restore the original sys.argv
            sys.argv = original_argv

        time.sleep(3)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("\nTraining complete!")

    def __call__(self) -> None:
        """Run the complete pipeline."""
        print(f"\n{'='*60}")
        print("Qwen3-TTS End-to-End Fine-Tuning Pipeline")
        print(f"{'='*60}\n")

        # Check dependencies
        self.check_dependencies()

        # Validate inputs
        audio_files = self.validate_audio_files()

        if not os.path.exists(self.train_raw_jsonl):
            # Transcribe
            transcription_results = self.transcribe_with_whisperx(audio_files)

            if not transcription_results:
                raise ValueError("No transcriptions were generated. Please check your audio files.")

            # Create JSONL
            self.create_train_jsonl(transcription_results)
        else:
            print(f"\n{'=' * 60}")
            print(
                f"File '{self.train_raw_jsonl}' already present; skipping 'transcribe_with_whisperx()'"
            )
            print(f"{'=' * 60}\n")

        if not os.path.exists(self.train_with_codes_jsonl):
            # Prepare data
            self.prepare_data()
        else:
            print(f"\n{'=' * 60}")
            print(
                f"File '{self.train_with_codes_jsonl}' already present; skipping 'prepare_data()'"
            )
            print(f"{'=' * 60}\n")

        # Train
        self.train_model()

        print(f"\n{'='*60}")
        print("Pipeline complete!")
        print(f"Checkpoints saved to: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS End-to-End Fine-Tuning Pipeline"
    )

    # Input arguments
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing WAV files to use for training",
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        required=True,
        help="Path to reference audio file (WAV)",
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        default="my_speaker",
        help="Name for the speaker being cloned",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save checkpoints and intermediate files",
    )

    # Model arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for training",
    )
    parser.add_argument(
        "--tokenizer_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Path to tokenizer model",
    )
    parser.add_argument(
        "--init_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Path to initial model for fine-tuning",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )

    # Whisper arguments
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size",
    )
    parser.add_argument(
        "--whisper_compute_type",
        type=str,
        default="float16",
        choices=["float16", "int8", "float32"],
        help="Whisper compute type",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        help="Language code (e.g., 'en', 'zh', 'es') or 'auto' for auto-detection",
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = Qwen3TTSPipeline(
        audio_dir=args.audio_dir,
        ref_audio=args.ref_audio,
        speaker_name=args.speaker_name,
        output_dir=args.output_dir,
        device=args.device,
        tokenizer_model_path=args.tokenizer_model_path,
        init_model_path=args.init_model_path,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        whisper_model=args.whisper_model,
        whisper_compute_type=args.whisper_compute_type,
        language=args.language,
    )

    pipeline()


if __name__ == "__main__":
    main()
