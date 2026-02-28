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

    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """
        Translates a batch of texts using NLLB.
        NLLB requires FLORES-200 language codes (e.g., eng_Latn, rus_Cyrl).
        """
        if not texts:
            return []
            
        # Simplified mapping logic for demo purposes.
        lang_map = {
            "en": "eng_Latn",
            "ru": "rus_Cyrl",
            "fr": "fra_Latn",
            "es": "spa_Latn",
            "de": "deu_Latn",
        }
        
        src_code = lang_map.get(src_lang, src_lang if "_" in src_lang else "eng_Latn")
        tgt_code = lang_map.get(tgt_lang, tgt_lang if "_" in tgt_lang else "rus_Cyrl")
        
        try:
            # the pipeline can process a list of texts
            results = self.translator(texts, src_lang=src_code, tgt_lang=tgt_code, max_length=512, batch_size=len(texts))
            return [res['translation_text'] for res in results]
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            return texts # Return original texts on failure
            
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Single translation wrapper for compatibility."""
        return self.translate_batch([text], src_lang, tgt_lang)[0]
