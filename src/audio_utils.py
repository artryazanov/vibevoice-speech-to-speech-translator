import logging
import subprocess
from pathlib import Path
from typing import Union, List, Dict
from pydub import AudioSegment
import torch
import torchaudio

logger = logging.getLogger(__name__)

class AudioUtils:
    @staticmethod
    def load_audio(file_path: Union[str, Path]) -> AudioSegment:
        """Loads an audio file using pydub."""
        try:
            return AudioSegment.from_file(str(file_path))
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            raise

    @staticmethod
    def save_audio(audio: AudioSegment, output_path: Union[str, Path], format: str = "wav"):
        """Saves an audio segment to a file."""
        audio.export(str(output_path), format=format)

    @staticmethod
    def resample_waveform(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Resamples a torchaudio waveform tensor."""
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
            return resampler(waveform)
        return waveform

    @staticmethod
    def merge_audio_timeline(segments: List[Dict], total_duration_ms: int) -> AudioSegment:
        """
        Assembles the final audio file from segments.
        segments: list of dicts {'audio': AudioSegment, 'start_ms': int}
        """
        final_audio = AudioSegment.silent(duration=total_duration_ms)
        
        # Simple strategy: overlay segments onto the timeline at their original start times.
        # Future improvement: implement timeline shifting logic if translation exceeds original duration.
        current_pos = 0
        
        for seg in sorted(segments, key=lambda x: x['start_ms']):
            start_ms = seg['start_ms']
            audio_chunk = seg['audio']
            
            # If the previous segment overlaps with this one, shift the current one start time
            # However, for speech-to-speech where we want to keep original timing as much as possible,
            # we might want to allow some overlap or compress silence.
            # For now, we strictly ensure no overlap by shifting forward if needed.
            if start_ms < current_pos:
                start_ms = current_pos
            
            final_audio = final_audio.overlay(audio_chunk, position=start_ms)
            current_pos = start_ms + len(audio_chunk)
            
        return final_audio

    @staticmethod
    def merge_video_audio(video_path: str, audio_path: str, output_path: str):
        """
        Merges video stream from video_path with audio stream from audio_path.
        Saves the result to output_path.
        """
        logger.info(f"Merging video '{video_path}' with audio '{audio_path}' -> '{output_path}'")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",  # Copy video without re-encoding
            "-c:a", "aac",   # Encode audio to aac
            "-map", "0:v:0", # Take video from 1st file
            "-map", "1:a:0", # Take audio from 2nd file
            "-shortest",     # Cut by the shortest stream
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg merge failed: {result.stderr}")
                raise RuntimeError(f"FFmpeg merge failed: {result.stderr}")
            logger.info("Video merge successful.")
        except Exception as e:
            logger.error(f"Failed to merge video and audio: {e}")
            raise
