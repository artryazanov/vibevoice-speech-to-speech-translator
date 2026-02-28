import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class ModelConfig:
    asr_model_path: str = "microsoft/VibeVoice-ASR" # Local path or HuggingFace ID
    tts_model_path: str = "tts_models/multilingual/multi-dataset/xtts_v2" # Local path or HuggingFace ID
    translator_model_path: str = "facebook/nllb-200-distilled-600M" # Lightweight and high-quality translation model
    device: str = "cuda"  # "cuda", "cpu", "mps"
    compute_dtype: str = "float16" # bfloat16 or float16
    cache_dir: str = "models" # Directory to store downloaded models

@dataclass
class ProcessingConfig:
    temp_dir: Path = Path("temp_files")
    output_dir: Path = Path("output")
    target_sample_rate: int = 24000
    
    def __post_init__(self):
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)

@dataclass
class AppConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

# Global config instance (can be replaced with DI if desired)
config = AppConfig()
