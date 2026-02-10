import torch
import logging
from typing import Optional

# Import VibeVoice classes. 
try:
    from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizer
except ImportError:
    logging.warning("VibeVoice modules not found. Ensure 'vibevoice/' directory is in project root.")
    VibeVoiceForConditionalGeneration = None
    VibeVoiceTokenizer = None

from src.config import config

logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self):
        if not VibeVoiceForConditionalGeneration or not VibeVoiceTokenizer:
            raise ImportError("VibeVoice modules not available.")

        logger.info("Loading VibeVoice TTS...")
        self.device = config.models.device
        self.dtype = torch.float16 if config.models.compute_dtype == "float16" else torch.bfloat16
        
        self.model = VibeVoiceForConditionalGeneration.from_pretrained(
            config.models.tts_model_path,
            torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        
        self.tokenizer = VibeVoiceTokenizer.from_pretrained(config.models.tts_model_path)
        
        # Cache for speaker prompts to maintain consistency
        self.speaker_prompts = {} 

    def get_speaker_prompt(self, speaker_id: str):
        """
        Retrieves or assigns a voice prompt/embedding for a specific speaker.
        """
        if speaker_id not in self.speaker_prompts:
            # TODO: Implement actual voice selection logic (e.g., from 'voices/' directory)
            # For now, we return None or a default if available. 
            # VibeVoice usually needs a reference audio path or embedding.
            pass
        return self.speaker_prompts.get(speaker_id)

    def synthesize(self, text: str, speaker_id: str) -> bytes:
        """Generates raw audio bytes (or numpy array)."""
        logger.info(f"Synthesizing for {speaker_id}: {text[:30]}...")
        
        # 1. Tokenize
        text_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # 2. Generate
        with torch.no_grad():
            output = self.model.generate(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                max_new_tokens=4000, 
                temperature=0.7,
                do_sample=True
                # speaker_embedding=self.get_speaker_prompt(speaker_id) 
            )
            
        # 3. Convert to audio
        # Assuming output is the waveform or codes that can be decoded.
        # This part heavily depends on the specific VibeVoice output format.
        # Standard huggingface TTS models return an object with 'waveform' or similar.
        
        # Placeholder: assuming output is a tensor representing waveform.
        # If it returns codes, we'd need a vocoder. 
        # VibeVoice is likely end-to-end or returns values we can use.
        if hasattr(output, 'waveform'):
             audio_values = output.waveform.cpu().numpy()
        else:
             # Fallback/Guess for VibeVoice structure
             audio_values = output.cpu().numpy()
        
        return audio_values
