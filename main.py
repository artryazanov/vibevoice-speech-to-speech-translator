import sys
from pathlib import Path
from typing import Optional
import fire

from src.orchestrator import LocalTranslationOrchestrator

def main(input_file: str, output_file: Optional[str] = None, target_lang: str = "ru", src_lang: str = "en"):
    """
    Local video/audio translator with voice cloning (VibeVoice).
    
    Args:
        input_file: Path to source file (mp3, wav, mp4...) OR YouTube URL.
        output_file: Path to save result. If None, defaults to input name + _translated.
        target_lang: Target language code (e.g. "ru", "en").
        src_lang: Source language code (e.g. "en").
    """
    
    # Logic to generate output filename
    if output_file is None:
        is_url = input_file.startswith("http://") or input_file.startswith("https://")
        
        if is_url:
            # For URL, default to saving as video .mp4, name "translated_video.mp4"
            # (since we only know the real filename after downloading,
            # but the user needs to know where the result will be saved)
            output_file = "translated_video.mp4"
        else:
            input_p = Path(input_file)
            stem = input_p.stem
            suffix = input_p.suffix.lower()
            
            # If input file is video, output is also video
            if suffix in [".mp4", ".mov", ".mkv", ".avi", ".webm"]:
                new_suffix = suffix
            else:
                new_suffix = ".wav"
                
            output_file = f"{stem}_translated{new_suffix}"
            
        print(f"Output path not provided. Defaulting to: {output_file}")

    orchestrator = LocalTranslationOrchestrator()
    orchestrator.process(input_file, output_file, target_lang, src_lang)

if __name__ == "__main__":
    fire.Fire(main)
