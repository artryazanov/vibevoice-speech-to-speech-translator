import sys
from pathlib import Path
import fire

# Add path to VibeVoice submodule
# This must be done before importing any module that depends on 'vibevoice' package
vibe_path = Path(__file__).parent / "external" / "vibevoice"
if vibe_path.exists():
    sys.path.append(str(vibe_path))
else:
    print(f"Warning: VibeVoice submodule not found at {vibe_path}")

from src.orchestrator import LocalTranslationOrchestrator

def main(input_file: str, output_file: str, target_lang: str = "ru", src_lang: str = "en"):
    """
    Local video/audio translator with voice cloning (VibeVoice).
    
    Args:
        input_file: Path to source file (mp3, wav, mp4...).
        output_file: Path to save result.
        target_lang: Target language code (e.g. "ru", "en").
        src_lang: Source language code (e.g. "en").
    """
    orchestrator = LocalTranslationOrchestrator()
    orchestrator.process(input_file, output_file, target_lang, src_lang)

if __name__ == "__main__":
    fire.Fire(main)
