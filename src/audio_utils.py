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
        segments: list of dicts {'audio': AudioSegment, 'start_ms': int, 'end_ms': int}
        """
        import soundfile as sf
        import pyrubberband as pyrb
        import io
        import numpy as np

        final_audio = AudioSegment.silent(duration=total_duration_ms)
        
        for seg in sorted(segments, key=lambda x: x['start_ms']):
            start_ms = seg['start_ms']
            end_ms = seg.get('end_ms', start_ms + len(seg['audio']))
            target_duration_ms = end_ms - start_ms
            
            if target_duration_ms <= 0:
                continue
                
            audio_chunk = seg['audio']
            original_duration_ms = len(audio_chunk)
            
            # Apply time-stretching if durations differ significantly
            if original_duration_ms > 0 and abs(original_duration_ms - target_duration_ms) > 50:
                # Calculate stretch ratio (original / target)
                # pyrubberband time_stretch rate where > 1.0 means faster (shorter output)
                rate = original_duration_ms / target_duration_ms
                
                # Export chunk to numpy array
                samples = np.array(audio_chunk.get_array_of_samples()).astype(np.float32) / 32768.0
                
                # Reshape for multi-channel if needed
                if audio_chunk.channels > 1:
                    samples = samples.reshape((-1, audio_chunk.channels))
                
                # Stretch
                try:
                    stretched_samples = pyrb.time_stretch(samples, audio_chunk.frame_rate, rate)
                    
                    # Convert back to AudioSegment
                    # ensure 16-bit PCM
                    stretched_samples = np.clip(stretched_samples * 32767.0, -32768, 32767).astype(np.int16)
                    with io.BytesIO() as wav_io:
                        sf.write(wav_io, stretched_samples, audio_chunk.frame_rate, format='WAV', subtype='PCM_16')
                        wav_io.seek(0)
                        audio_chunk = AudioSegment.from_wav(wav_io)
                except Exception as e:
                    logger.warning(f"Time-stretching failed: {e}. Using original chunk.")
            
            final_audio = final_audio.overlay(audio_chunk, position=start_ms)
            
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
