import torch
import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import VibeVoice classes. 
# NOTE: These imports assume the user has placed the 'vibevoice' source code in the project root.
try:
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
except ImportError:
    # Fallback/Mock for environment where vibevoice is not yet present (e.g. CI/Linter)
    # This prevents immediate crash if the user hasn't copied the files yet.
    logging.warning("VibeVoice modules not found. Ensure 'vibevoice/' directory is in project root.")
    VibeVoiceASRForConditionalGeneration = None
    VibeVoiceASRProcessor = None

from src.config import config

logger = logging.getLogger(__name__)

class ASREngine:
    def __init__(self):
        if not VibeVoiceASRForConditionalGeneration or not VibeVoiceASRProcessor:
            raise ImportError("VibeVoice modules not available. Please install the VibeVoice library in the project root.")

        logger.info("Loading VibeVoice ASR...")
        self.device = config.models.device
        self.dtype = torch.float16 if config.models.compute_dtype == "float16" else torch.bfloat16
        
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            config.models.asr_model_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B",
            cache_dir=config.models.cache_dir
        )
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            config.models.asr_model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            load_in_8bit=True, # Added for VRAM management
            trust_remote_code=True,
            cache_dir=config.models.cache_dir
        )
        self.model.eval()

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Returns a list of segments:
        [{'speaker': 'Speaker A', 'start': 0.5, 'end': 2.3, 'text': 'Hello world'}, ...]
        """
        logger.info(f"Transcribing {audio_path}...")
        
        # VibeVoice processor handles audio loading if path is passed
        inputs = self.processor(
            audio=audio_path,
            sampling_rate=16000, # Standard for ASR
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Cast float tensors to the model's dtype
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(self.dtype)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.0, # Greedy for accuracy
                do_sample=False
            )
            
        # Decode output
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        transcription = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        # Parse structured output from VibeVoice
        try:
            # Assuming post_process_transcription exists in the processor (as per VibeVoice examples)
            if hasattr(self.processor, 'post_process_transcription'):
                structured_segments = self.processor.post_process_transcription(transcription)
            else:
                # Basic fallback if method is missing or API changed - needs actual implementation details
                # For now, we assume standard VibeVoice structured output parsing
                logger.warning("post_process_transcription not found, returning raw text in single segment")
                return [{'speaker': 'Unknown', 'start': 0.0, 'end': 0.0, 'text': transcription}]

            results = []
            for seg in structured_segments:
                results.append({
                    'speaker': seg.get('speaker_id', 'Unknown'),
                    'start': float(seg.get('start_time', 0.0)),
                    'end': float(seg.get('end_time', 0.0)),
                    'text': seg.get('text', '').strip()
                })
            return results
        except Exception as e:
            logger.error(f"Error parsing transcription: {e}. Raw text: {transcription}")
            return []
