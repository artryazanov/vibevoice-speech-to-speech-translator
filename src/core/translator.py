import torch
import logging
from transformers import pipeline
from src.config import config

logger = logging.getLogger(__name__)

class TranslatorEngine:
    def __init__(self):
        logger.info(f"Loading Translator ({config.models.translator_model_path})...")
        # pipeline expects device as int (-1 for CPU, 0+ for GPU)
        self.device = 0 if config.models.device == "cuda" else -1
        
        self.translator = pipeline(
            "translation",
            model=config.models.translator_model_path,
            device=self.device,
            torch_dtype=torch.float16 if config.models.compute_dtype == "float16" else torch.float32,
            model_kwargs={"cache_dir": config.models.cache_dir}
        )

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translated text using NLLB.
        NLLB requires FLORES-200 language codes (e.g., eng_Latn, rus_Cyrl).
        """
        if not text:
            return ""
            
        # Simplified mapping logic for demo purposes. 
        # Ideally, this should be a robust mapping function or strict input requirement.
        lang_map = {
            "en": "eng_Latn",
            "ru": "rus_Cyrl",
            "fr": "fra_Latn",
            "es": "spa_Latn",
            "de": "deu_Latn",
            # Add more as needed
        }
        
        # Default to English/Russian if not found, or use input as is if it looks like a code
        src_code = lang_map.get(src_lang, src_lang if "_" in src_lang else "eng_Latn")
        tgt_code = lang_map.get(tgt_lang, tgt_lang if "_" in tgt_lang else "rus_Cyrl")
        
        try:
            # NLLB pipeline handles the forced_bos_token automatically if src/tgt are passed
            # But standard pipeline might need specific args. 
            # For NLLB/M2M100, we usually use `src_lang` and `tgt_lang` in generate
            result = self.translator(text, src_lang=src_code, tgt_lang=tgt_code, max_length=512)
            return result[0]['translation_text']
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text # Return original text on failure
