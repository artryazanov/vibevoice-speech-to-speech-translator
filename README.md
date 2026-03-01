# VibeVoice Speech-to-Speech Translator

> [!WARNING]
> **This project is currently under active development.** The features and capabilities described below are intended goals and have not all been fully achieved or stabilized yet. Expect bugs, missing features, and breaking changes.

This project implements a professional-grade local speech-to-speech translation pipeline. It combines Microsoft's VibeVoice (ASR) with 8-bit quantization, NLLB-200 for batch text translation, and Coqui XTTS v2 for high-fidelity voice cloning.
## Prerequisites

1. **Python 3.10+**
2. **CUDA-enabled GPU** (recommended for performance)
3. **FFmpeg**: Required for audio/video processing and merging.
4. **Git Submodule**: This project uses a git submodule for the VibeVoice library.

## Installation

1. Clone the repository and initialize submodules:
   ```bash
   git clone https://github.com/artryazanov/vibevoice-speech-to-speech-translator.git
   cd vibevoice-speech-to-speech-translator
   git submodule update --init --recursive
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure VibeVoice code is present (should be handled by git submodule):
   ```text
   .
   ├── external/
   │   └── vibevoice/  <-- Git submodule
   ├── src/
   ├── main.py
   ...
   ```

4. (Optional) Download models manually if not using automatic HuggingFace download.

## Usage
    
Run the translator via CLI:

### 1. Translate YouTube Video
Downloads video, translates audio, and merges back with original video.
```bash
python main.py --input_file "https://www.youtube.com/watch?v=..."
# Output: translated_video.mp4
```

### 2. Translate Local Video File
Translates audio and merges back with original video.
```bash
python main.py --input_file "path/to/video.mp4"
# Output: video_translated.mp4
```

### 3. Translate Audio File
```bash
python main.py --input_file "path/to/audio.mp3" --target_lang "ru"
# Output: audio_translated.wav
```

### Arguments

- `--input_file`: Path to source file (mp3, wav, mp4...) OR YouTube URL.
- `--output_file`: (Optional) Path where the result will be saved. If not provided, it is generated automatically based on input.
- `--target_lang`: Target language code (e.g., `ru`, `es`, `fr`). Default: `ru`.
- `--src_lang`: Source language code (e.g., `en`). Default: `en`.

## Configuration

You can adjust model paths and settings in `src/config.py`.

## Features

- **Pristine Voice Cloning**: Uses XTTS v2 to synthesize speech natively capturing original speaker emotion and timbre continuously across the timeline.
- **VRAM Optimized Pipeline**: Sequentially manages ASR, translation, and TTS lifecycles in isolated temporary contexts with aggressive CUDA garbage collection. The 7B ASR model is loaded via 8-bit quantization, safely staying within 16GB VRAM bounds.
- **Precision Audio Syncing**: Utilizes `pyrubberband` time-stretching algorithms to ensure flawlessly synchronized generated speech that identically maps to source ASR bounding boxes.
- **Batch Translation Processing**: NLLB pipelines are fired efficiently over full text extraction arrays.
- **YouTube Support**: Download and translate videos directly from YouTube URLs.
- **Video Processing**: Supports video file input (`.mp4`, `.mkv`, etc.) and automatically merges translated audio with the original video.

## Architecture

- **ASR**: VibeVoice ASR (8-bit quantized, transcribes and diarizes speakers)
- **Voice Extraction**: Orchestrator slices 3-6 second reference audio segments per speaker.
- **Translation**: NLLB-200 (Batch processes texts to target language)
- **TTS**: Coqui XTTS v2 (Synthesizes speech preserving exact speaker identity using reference wav files)
- **Audio Sync**: `pyrubberband` dynamically time-stretches generated audio chunks to match original start/end metrics.
- **Orchestrator**: Safely wraps all I/O via `tempfile`, manages sequential GPU memory execution, downloads, and final media output merging.

## Docker Support

### Build
To build the Docker image:
```bash
docker build -t vibe-translator .
```

### Run
To run the container, mounting input/output directories and **persisting models**:
```bash
docker run --gpus all \
  -v /path/to/data:/data \
  -v $(pwd)/models:/app/models \
  vibe-translator \
  --input_file "/data/input.mp3" \
  --output_file "/data/output.wav" \
  --target_lang "ru"
```
**Notes:**
- `--gpus all` is required for GPU access.
- `-v /path/to/data:/data` mounts a local directory to `/data` in the container.
- `-v $(pwd)/models:/app/models` mounts a local directory to persist downloaded models.
