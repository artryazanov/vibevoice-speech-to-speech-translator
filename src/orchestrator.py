import logging
import time
import io
import os
import gc
import soundfile as sf
import numpy as np
import tempfile
import torch
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
        # Models are loaded dynamically to save VRAM
        pass
        
    def _free_vram(self, model_holder):
        """Forces garbage collection and clears CUDA cache."""
        del model_holder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _extract_speaker_samples(self, audio: AudioSegment, segments: list, temp_dir: Path) -> dict:
        """Extracts audio chunks for each speaker to be used as TTS reference."""
        speaker_samples = {}
        for seg in segments:
            speaker_id = seg['speaker']
            if speaker_id not in speaker_samples:
                start_ms = int(seg['start'] * 1000)
                end_ms = int(seg['end'] * 1000)
                
                # Try to grab 3 to 6 seconds for better quality
                # If segment is too short, we just take it
                duration = end_ms - start_ms
                if duration < 1000:
                    continue # Skip very short segments for reference
                    
                chunk = audio[start_ms:end_ms]
                sample_path = temp_dir / f"ref_{speaker_id}.wav"
                AudioUtils.save_audio(chunk, str(sample_path))
                speaker_samples[speaker_id] = str(sample_path)
                
        # Fallback for speakers without a good reference
        default_ref = list(speaker_samples.values())[0] if speaker_samples else None
        for seg in segments:
            speaker_id = seg['speaker']
            if speaker_id not in speaker_samples and default_ref:
                speaker_samples[speaker_id] = default_ref
                
        return speaker_samples

    def process(self, input_path: str, output_path: str, target_lang: str, src_lang: str = "en"):
        # Use tempfile to ensure OS cleaning of temporary artifacts
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # 1. Determine input type (URL or File)
            is_url = input_path.startswith("http://") or input_path.startswith("https://")
            downloaded_temp_file = None
            original_video_path = None
            
            is_video_output = Path(output_path).suffix.lower() in [".mp4", ".mov", ".mkv", ".avi", ".webm"]

            try:
                if is_url:
                    logger.info("URL detected. Initiating download...")
                    downloaded_temp_file = download_content(input_path, temp_dir_path, prefer_video=is_video_output)
                    input_path = str(downloaded_temp_file)
                
                input_p = Path(input_path)
                if input_p.suffix.lower() in [".mp4", ".mov", ".mkv", ".avi", ".webm"]:
                    original_video_path = str(input_p)
                    logger.info(f"Video input detected: {original_video_path}")

                start_time = time.time()
                logger.info(f"Starting pipeline: {input_path} -> {output_path} ({target_lang})")
                
                # --- STAGE 1: ASR & Diarization ---
                logger.info("--- STAGE 1: ASR ---")
                asr = ASREngine()
                segments = asr.transcribe(input_path)
                logger.info(f"Detected {len(segments)} segments.")
                
                self._free_vram(asr)
                
                if not segments:
                    logger.warning("No speech detected. Exiting.")
                    return
                
                # --- STAGE 2: Extract Speaker References ---
                logger.info("--- STAGE 2: Extracting References ---")
                try:
                    original_audio = AudioUtils.load_audio(input_path)
                    total_duration = len(original_audio)
                    speaker_refs = self._extract_speaker_samples(original_audio, segments, temp_dir_path)
                except Exception as e:
                    logger.error(f"Failed to load original audio or extract references: {e}")
                    raise
                    
                # --- STAGE 3: Batch Translation ---
                logger.info("--- STAGE 3: Translation ---")
                translator = TranslatorEngine()
                
                # Gather texts
                texts_to_translate = [seg['text'] for seg in segments]
                translated_texts = translator.translate_batch(texts_to_translate, src_lang, target_lang)
                
                for i, seg in enumerate(segments):
                    seg['translated_text'] = translated_texts[i]
                    logger.info(f"Speaker {seg['speaker']}: {seg['text'][:30]}... -> {seg['translated_text'][:30]}...")
                    
                self._free_vram(translator)
                
                # --- STAGE 4: TTS Synthesis ---
                logger.info("--- STAGE 4: TTS Synthesis ---")
                tts = TTSEngine()
                
                processed_segments = []
                for i, seg in enumerate(segments):
                    text = seg['translated_text']
                    speaker_id = seg['speaker']
                    ref_audio_path = speaker_refs.get(speaker_id)
                    
                    audio_data = tts.synthesize(text, ref_audio_path, target_lang)
                    
                    try:
                        if len(audio_data) > 0:
                            with io.BytesIO() as wav_io:
                                # target sample rate should ideally match TTS output
                                sf.write(wav_io, audio_data, config.processing.target_sample_rate, format='WAV')
                                wav_io.seek(0)
                                seg_audio = AudioSegment.from_wav(wav_io)
                            
                            processed_segments.append({
                                'audio': seg_audio,
                                'start_ms': int(seg['start'] * 1000),
                                'end_ms': int(seg['end'] * 1000),
                                'speaker': speaker_id
                            })
                    except Exception as e:
                        logger.error(f"Failed to convert TTS output to audio segment: {e}")
                        continue
                        
                self._free_vram(tts)
                
                # --- STAGE 5: Merge Timeline and Output ---
                logger.info("--- STAGE 5: Audio Synchronization ---")
                final_audio = AudioUtils.merge_audio_timeline(processed_segments, total_duration)
                
                if is_video_output and original_video_path:
                    logger.info("Merging translated audio with original video...")
                    temp_audio_path = temp_dir_path / "temp_final_audio.wav"
                    AudioUtils.save_audio(final_audio, temp_audio_path)
                    AudioUtils.merge_video_audio(original_video_path, str(temp_audio_path), output_path)
                else:
                    AudioUtils.save_audio(final_audio, output_path)
                
                logger.info(f"Done in {time.time() - start_time:.2f}s. Saved to {output_path}")

            except Exception as e:
                logger.error(f"Pipeline failed: {e}")
                raise
