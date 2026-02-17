# VibeVoice Speech-to-Speech Translator

This project implements a local speech-to-speech translation pipeline using Microsoft's VibeVoice models for ASR and TTS, and NLLB for text translation.

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

- **Speech-to-Speech Translation**: Preserves speaker identity using VibeVoice.
- **YouTube Support**: Download and translate videos directly from YouTube URLs.
- **Video Processing**: Supports video file input (`.mp4`, `.mkv`, etc.) and automatically merges translated audio with the original video.
- **Automatic Output Naming**: Automatically generates output filenames if not specified.

## Architecture

- **ASR**: VibeVoice ASR (Transcribes and identifies speakers)
- **Translation**: NLLB-200 (Translates text)
- **TTS**: VibeVoice TTS (Synthesizes speech preserving speaker identity)
- **Orchestrator**: Manages the pipeline, downloads, and media merging.

## Docker Support

### Build
To build the Docker image:
```bash
docker build -t vibe-translator .
```

### Run
To run the container, mounting input/output directories:
```bash
docker run --gpus all -v /path/to/data:/data vibe-translator \
  --input_file "/data/input.mp3" \
  --output_file "/data/output.wav" \
  --target_lang "ru"
```
**Notes:**
- `--gpus all` is required for GPU access.
- `-v /path/to/data:/data` mounts a local directory to `/data` in the container.
