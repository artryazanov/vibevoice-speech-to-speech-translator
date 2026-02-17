import logging
import time
import io
import os
import soundfile as sf
import numpy as np
from pathlib import Path
from pydub import AudioSegment

from src.config import config
from src.audio_utils import AudioUtils
from src.core.asr_engine import ASREngine
from src.core.tts_engine import TTSEngine
from src.core.translator import TranslatorEngine
from src.core.downloader import download_content

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalTranslationOrchestrator:
    def __init__(self):
        self.asr = ASREngine()
        self.translator = TranslatorEngine()
        self.tts = TTSEngine()
        
    def process(self, input_path: str, output_path: str, target_lang: str, src_lang: str = "en"):
        # 1. Determine input type (URL or File)
        is_url = input_path.startswith("http://") or input_path.startswith("https://")
        downloaded_temp_file = None
        original_video_path = None
        
        # Determine if output file is video (by extension)
        is_video_output = Path(output_path).suffix.lower() in [".mp4", ".mov", ".mkv", ".avi", ".webm"]

        try:
            if is_url:
                logger.info("URL detected. Initiating download...")
                # Download. If output is video, prefer downloading video.
                downloaded_temp_file = download_content(input_path, config.processing.temp_dir, prefer_video=is_video_output)
                input_path = str(downloaded_temp_file)
            
            # Check if input file is video (downloaded or local)
            input_p = Path(input_path)
            if input_p.suffix.lower() in [".mp4", ".mov", ".mkv", ".avi", ".webm"]:
                original_video_path = str(input_p)
                logger.info(f"Video input detected: {original_video_path}")

            start_time = time.time()
            logger.info(f"Starting pipeline: {input_path} -> {output_path} ({target_lang})")
            
            # 2. ASR & Diarization
            # Pydub (inside AudioUtils/ASR) will extract audio from video file automatically
            segments = self.asr.transcribe(input_path)
            logger.info(f"Detected {len(segments)} segments.")
            
            processed_segments = []
            
            # 3. Loop: Translate -> TTS
            for i, seg in enumerate(segments):
                original_text = seg['text']
                speaker_id = seg['speaker']
                
                logger.info(f"[{i+1}/{len(segments)}] Speaker {speaker_id}: {original_text}")
                
                translated_text = self.translator.translate(original_text, src_lang, target_lang)
                logger.info(f"   -> {translated_text}")
                
                audio_data = self.tts.synthesize(translated_text, speaker_id)
                
                try:
                    with io.BytesIO() as wav_io:
                        sf.write(wav_io, audio_data, config.processing.target_sample_rate, format='WAV')
                        wav_io.seek(0)
                        seg_audio = AudioSegment.from_wav(wav_io)
                    
                    processed_segments.append({
                        'audio': seg_audio,
                        'start_ms': int(seg['start'] * 1000),
                        'speaker': speaker_id
                    })
                except Exception as e:
                    logger.error(f"Failed to convert TTS output to audio segment: {e}")
                    continue
            
            # 4. Merge Timeline
            try:
                 original_audio = AudioUtils.load_audio(input_path)
                 total_duration = len(original_audio)
            except Exception:
                 if processed_segments:
                     total_duration = processed_segments[-1]['start_ms'] + len(processed_segments[-1]['audio']) + 1000
                 else:
                     total_duration = 0

            final_audio = AudioUtils.merge_audio_timeline(processed_segments, total_duration)
            
            # 5. Save Output (Audio or Video)
            if is_video_output and original_video_path:
                logger.info("Merging translated audio with original video...")
                
                # Save audio to temporary file first
                temp_audio_path = config.processing.temp_dir / "temp_final_audio.wav"
                AudioUtils.save_audio(final_audio, temp_audio_path)
                
                # Merge
                AudioUtils.merge_video_audio(original_video_path, str(temp_audio_path), output_path)
                
                # Remove temporary audio
                if temp_audio_path.exists():
                    os.remove(temp_audio_path)
            else:
                # Just save audio
                AudioUtils.save_audio(final_audio, output_path)
            
            logger.info(f"Done in {time.time() - start_time:.2f}s. Saved to {output_path}")

        finally:
            # Cleanup downloaded file
            if downloaded_temp_file and downloaded_temp_file.exists():
                logger.info(f"Cleaning up downloaded file: {downloaded_temp_file}")
                os.remove(downloaded_temp_file)
