import logging
import time
import io
import soundfile as sf
import numpy as np
from pydub import AudioSegment

from src.config import config
from src.audio_utils import AudioUtils
from src.core.asr_engine import ASREngine
from src.core.tts_engine import TTSEngine
from src.core.translator import TranslatorEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalTranslationOrchestrator:
    def __init__(self):
        self.asr = ASREngine()
        self.translator = TranslatorEngine()
        self.tts = TTSEngine()
        
    def process(self, input_path: str, output_path: str, target_lang: str, src_lang: str = "en"):
        start_time = time.time()
        logger.info(f"Starting pipeline: {input_path} -> {output_path} ({target_lang})")
        
        # 1. ASR & Diarization (Single Pass)
        # VibeVoice ASR handles speaker identification and timestamps.
        segments = self.asr.transcribe(input_path)
        logger.info(f"Detected {len(segments)} segments.")
        
        processed_segments = []
        
        # 2. Loop through segments: Translate -> TTS
        for i, seg in enumerate(segments):
            original_text = seg['text']
            speaker_id = seg['speaker']
            
            logger.info(f"[{i+1}/{len(segments)}] Speaker {speaker_id}: {original_text}")
            
            # Translation
            translated_text = self.translator.translate(original_text, src_lang, target_lang)
            logger.info(f"   -> {translated_text}")
            
            # TTS
            # Generate audio (numpy array float32 typically)
            audio_data = self.tts.synthesize(translated_text, speaker_id)
            
            # Conversion numpy -> AudioSegment
            # We assume TTS outputs 24000Hz (VibeVoice standard)
            # Must save to temporary buffer for pydub to read
            try:
                with io.BytesIO() as wav_io:
                    # sf.write expects data, samplerate
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
            
        # 3. Merge Timeline
        # Load original audio to get total duration
        try:
             original_audio = AudioUtils.load_audio(input_path)
             total_duration = len(original_audio)
        except Exception:
             # Fallback if original audio load fails (rare)
             if processed_segments:
                 total_duration = processed_segments[-1]['start_ms'] + len(processed_segments[-1]['audio']) + 1000
             else:
                 total_duration = 0

        final_audio = AudioUtils.merge_audio_timeline(processed_segments, total_duration)
        
        # 4. Save
        AudioUtils.save_audio(final_audio, output_path)
        
        logger.info(f"Done in {time.time() - start_time:.2f}s. Saved to {output_path}")
