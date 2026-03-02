# VibeVoice Speech-to-Speech Translator

> [!WARNING]
> **This project is currently under active development.** While the core speech-to-speech translation pipeline is fully functional, there are known limitations regarding the final output quality. Specifically, machine translation accuracy may vary, and the generated audio segments are often dynamically time-stretched (sped up or slowed down) to synchronize with the original speaker's timing, which can sometimes result in an unnatural voice cadence. Expect bugs, missing features, and breaking changes as we continue to improve these elements.

This project implements a professional-grade local speech-to-speech translation pipeline. It combines Faster-Whisper Large-v3 for high-quality transcription, Pyannote Audio for precise speaker diarization, NLLB-200 for batch text translation, and Coqui XTTS v2 for high-fidelity voice cloning.
## Prerequisites

1. **Python 3.10+**
2. **CUDA-enabled GPU** (recommended for performance)
3. **FFmpeg**: Required for audio/video processing and merging.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/artryazanov/vibevoice-speech-to-speech-translator.git
   cd vibevoice-speech-to-speech-translator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables for gated models (Pyannote):
   - You must accept user conditions for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on Hugging Face.
   - Create a Hugging Face Access Token.
   - Copy `.env.example` to `.env` and insert your token:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` to include your token: `HF_TOKEN=your_token_here`

4. Ensure any required submodules are present (if applicable):
   ```text
   .
   ├── src/
   ├── .env
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
- **VRAM Optimized Pipeline**: Sequentially manages ASR (Whisper), Diarization (Pyannote), translation, and TTS lifecycles in isolated temporary contexts with aggressive CUDA garbage collection. This strictly limits peak memory usage, preventing Whisper and Pyannote from co-existing in VRAM.
- **Precision Audio Syncing**: Utilizes `pyrubberband` time-stretching algorithms to ensure flawlessly synchronized generated speech that identically maps to source ASR bounding boxes.
- **Batch Translation Processing**: NLLB pipelines are fired efficiently over full text extraction arrays.
- **YouTube Support**: Download and translate videos directly from YouTube URLs.
- **Video Processing**: Supports video file input (`.mp4`, `.mkv`, etc.) and automatically merges translated audio with the original video.

## Architecture

- **ASR**: Faster-Whisper Large-v3 (transcription) combined with Pyannote Audio (speaker diarization).
- **Voice Extraction**: Orchestrator slices reference audio segments per speaker based on Pyannote's diarization tracks.
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
  -v $(pwd)/output:/data \
  -v $(pwd)/models:/app/models \
  --env-file .env \
  vibe-translator \
  --input_file "/data/input.mp3" \
  --output_file "/data/output.wav" \
  --target_lang "ru"
```
**Notes:**
- `--gpus all` is required for GPU access.
- `-v /path/to/data:/data` mounts a local directory to `/data` in the container.
- `-v $(pwd)/models:/app/models` mounts a local directory to persist downloaded models.
