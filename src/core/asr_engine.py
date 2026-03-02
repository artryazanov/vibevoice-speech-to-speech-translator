import gc
import torch
import logging
from typing import List, Dict, Any
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from src.config import config

logger = logging.getLogger(__name__)

class ASREngine:
    def __init__(self):
        self.device = config.models.device
        self.compute_type = "float16" if config.models.compute_dtype == "float16" else "float32"

    def _free_memory(self, model):
        """Local memory cleanup during the ASR phase"""
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        # === 1. TRANSCRIPTION PHASE ===
        logger.info(f"Loading Faster-Whisper ({config.models.asr_model_path})...")
        whisper_model = WhisperModel(
            config.models.asr_model_path, 
            device=self.device, 
            compute_type=self.compute_type,
            download_root=config.models.cache_dir
        )
        
        logger.info("Transcribing audio...")
        segments_generator, _ = whisper_model.transcribe(
            audio_path, 
            beam_size=5, 
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Save results to a list in memory, as the generator only works in a loop
        whisper_segments = []
        for s in segments_generator:
            whisper_segments.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})
            
        # STRICT VRAM UNLOADING
        logger.info("Unloading Faster-Whisper from VRAM...")
        self._free_memory(whisper_model)

        # === 2. DIARIZATION PHASE ===
        logger.info("Loading Pyannote Diarization...")
        diarization_pipeline = Pipeline.from_pretrained(
            config.models.diarization_model_path, 
            use_auth_token=config.models.hf_token
        )
        
        if diarization_pipeline is None:
            raise RuntimeError(
                f"Failed to load Pyannote diarization pipeline '{config.models.diarization_model_path}'. "
                "This is likely because the model is gated and requires a valid Hugging Face token. "
                "Please ensure you have accepted the user conditions on the Hugging Face model page "
                "and that you are passing the HF_TOKEN environment variable correctly (e.g., "
                "`docker run -e HF_TOKEN=your_token ...`)."
            )
            
        diarization_pipeline.to(torch.device(self.device))
        
        logger.info("Processing diarization...")
        diarization_result = diarization_pipeline(audio_path)
        
        # STRICT VRAM UNLOADING
        logger.info("Unloading Pyannote from VRAM...")
        self._free_memory(diarization_pipeline)

        # === 3. MERGING RESULTS ===
        logger.info("Merging text with speakers...")
        return self._assign_speakers(whisper_segments, diarization_result)

    def _assign_speakers(self, whisper_segments: List[Dict], diarization_result) -> List[Dict[str, Any]]:
        """Matches Whisper text with speaker timecodes from Pyannote"""
        final_segments = []
        
        # Convert Pyannote results into a convenient list of dictionaries
        speakers = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            speakers.append({"start": turn.start, "end": turn.end, "speaker": speaker})

        for w_seg in whisper_segments:
            w_start, w_end = w_seg["start"], w_seg["end"]
            
            # Find the speaker whose time segment has the maximum overlap with the text
            max_overlap = 0
            assigned_speaker = "Unknown"
            
            for spk in speakers:
                # Calculate segment overlap
                overlap_start = max(w_start, spk["start"])
                overlap_end = min(w_end, spk["end"])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    assigned_speaker = spk["speaker"]
            
            final_segments.append({
                'speaker': assigned_speaker,
                'start': w_start,
                'end': w_end,
                'text': w_seg["text"]
            })
            
        return final_segments
