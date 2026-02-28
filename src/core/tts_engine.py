import torch
import logging
import numpy as np
from typing import Optional

try:
    from TTS.api import TTS
except ImportError:
    logging.warning("TTS module not found. Ensure Coqui TTS is installed.")
    TTS = None

from src.config import config

logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self):
        if not TTS:
            raise ImportError("TTS module not available.")

        logger.info(f"Loading XTTS Model: {config.models.tts_model_path}")
        self.device = config.models.device
        
        # xtts_v2 is quite efficient, we load it using Coqui's API
        self.model = TTS(config.models.tts_model_path).to(self.device)

    def synthesize(self, text: str, speaker_wav: str, language: str = "ru") -> np.ndarray:
        """Generates raw audio bytes (or numpy array)."""
        if not text or not speaker_wav:
            logger.warning("Synthesis skipped due to missing text or speaker reference.")
            return np.array([])
            
        logger.info(f"Synthesizing for lang {language} with ref {speaker_wav}: {text[:30]}...")
        
        try:
            with torch.no_grad():
                audio_values = self.model.tts(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=language
                )
            
            if isinstance(audio_values, list):
                audio_values = np.array(audio_values)
                
            return audio_values
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return np.array([])
