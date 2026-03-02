import sys
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



# Mock other heavy dependencies usually not present in minimal envs
sys.modules['torch'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()
sys.modules['torchaudio.transforms'] = MagicMock()
sys.modules['pydub'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['librosa'] = MagicMock()
sys.modules['soundfile'] = MagicMock()


class TestImports(unittest.TestCase):
    def test_config_import(self):
        from src.config import config
        self.assertIsNotNone(config)
        self.assertEqual(config.models.device, "cuda")

    def test_audio_utils_import(self):
        from src.audio_utils import AudioUtils
        self.assertTrue(hasattr(AudioUtils, 'load_audio'))

    def test_core_engines_import(self):
        # These will use the mocks we set up
        from src.core.asr_engine import ASREngine
        from src.core.tts_engine import TTSEngine
        from src.core.translator import TranslatorEngine
        
        self.assertTrue(True) # Just checking if imports succeed without crashing

if __name__ == '__main__':
    unittest.main()
