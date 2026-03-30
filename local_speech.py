"""
Local Speech Recognition for Jarvis
==================================
Multiple local speech recognition engines for offline transcription.
Engines:
1. Faster Whisper (local, fast, accurate)
2. Vosk (lightweight, small models)
3. SpeechRecognition (multiple backends)
"""

import os
import sys
import time
import wave
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import queue

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import local speech recognition libraries
HAS_FASTER_WHISPER = False
HAS_VOSK = False
HAS_SPEECH_RECOGNITION = False
HAS_OPENAI_WHISPER = False

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
    print("[OK] Faster Whisper available")
except ImportError:
    print("[!] Faster Whisper not found")

try:
    import vosk
    HAS_VOSK = True
    print("[OK] Vosk available")
except ImportError:
    print("[!] Vosk not found")

try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
    print("[OK] SpeechRecognition available")
except ImportError:
    print("[!] SpeechRecognition not found")

try:
    import whisper
    HAS_OPENAI_WHISPER = True
    print("[OK] OpenAI Whisper available")
except ImportError:
    print("[!] OpenAI Whisper not found")

# Configuration
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Available local engines
ENGINES = {
    "faster_whisper": {
        "available": HAS_FASTER_WHISPER,
        "description": "Fast local Whisper (recommended)",
        "model_sizes": ["tiny", "base", "small", "medium", "large"],
        "default_size": "base"
    },
    "vosk": {
        "available": HAS_VOSK,
        "description": "Lightweight, small models",
        "model_sizes": ["tiny", "small", "medium", "large"],
        "default_size": "small"
    },
    "speech_recognition": {
        "available": HAS_SPEECH_RECOGNITION,
        "description": "Multiple backends (Google, Sphinx, etc.)",
        "backends": ["google", "sphinx", "wit", "azure", "ibm"]
    },
    "openai_whisper": {
        "available": HAS_OPENAI_WHISPER,
        "description": "Original OpenAI Whisper (local)",
        "model_sizes": ["tiny", "base", "small", "medium", "large"],
        "default_size": "base"
    }
}

class LocalTranscriber:
    """Local speech recognition with fallback engines."""
    
    def __init__(self, preferred_engine: str = "faster_whisper", model_size: str = None):
        self.preferred_engine = preferred_engine
        self.model_size = model_size or ENGINES[preferred_engine].get("default_size", "base")
        self.engine = None
        self.model = None
        self.current_engine = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize engines in order of preference
        self._initialize_engines()
        
    def _initialize_engines(self):
        """Initialize available speech recognition engines."""
        engines_to_try = [
            ("faster_whisper", self._init_faster_whisper),
            ("vosk", self._init_vosk),
            ("speech_recognition", self._init_speech_recognition),
            ("openai_whisper", self._init_openai_whisper)
        ]
        
        # Try preferred engine first
        if self.preferred_engine in [e[0] for e in engines_to_try]:
            engines_to_try = [
                (self.preferred_engine, next(m[1] for m in engines_to_try if m[0] == self.preferred_engine))
            ] + [(e[0], e[1]) for e in engines_to_try if e[0] != self.preferred_engine]
        
        for engine_name, init_func in engines_to_try:
            if ENGINES[engine_name]["available"]:
                try:
                    self.logger.info(f"Trying to initialize {engine_name}...")
                    if init_func():
                        self.current_engine = engine_name
                        self.logger.info(f"Successfully initialized {engine_name}")
                        return
                except Exception as e:
                    self.logger.warning(f"Failed to initialize {engine_name}: {e}")
                    continue
        
        raise RuntimeError("No speech recognition engines available!")
    
    def _init_faster_whisper(self) -> bool:
        """Initialize Faster Whisper."""
        try:
            # Download model if needed
            model_path = MODELS_DIR / f"whisper_{self.model_size}"
            
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                download_root=str(MODELS_DIR)
            )
            self.engine = "faster_whisper"
            return True
        except Exception as e:
            self.logger.error(f"Faster Whisper init failed: {e}")
            return False
    
    def _init_vosk(self) -> bool:
        """Initialize Vosk."""
        try:
            # Download model if needed
            model_url = f"https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
            model_path = MODELS_DIR / "vosk-model-small-en-us-0.15"
            
            if not model_path.exists():
                self.logger.info("Downloading Vosk model...")
                import urllib.request
                import zipfile
                
                zip_path = MODELS_DIR / "vosk_model.zip"
                urllib.request.urlretrieve(model_url, zip_path)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(MODELS_DIR)
                
                zip_path.unlink()
            
            self.model = vosk.Model(str(model_path))
            self.engine = "vosk"
            return True
        except Exception as e:
            self.logger.error(f"Vosk init failed: {e}")
            return False
    
    def _init_speech_recognition(self) -> bool:
        """Initialize SpeechRecognition."""
        try:
            self.model = sr.Recognizer()
            self.engine = "speech_recognition"
            return True
        except Exception as e:
            self.logger.error(f"SpeechRecognition init failed: {e}")
            return False
    
    def _init_openai_whisper(self) -> bool:
        """Initialize OpenAI Whisper."""
        try:
            self.model = whisper.load_model(self.model_size)
            self.engine = "openai_whisper"
            return True
        except Exception as e:
            self.logger.error(f"OpenAI Whisper init failed: {e}")
            return False
    
    def transcribe_file(self, audio_path: Path, engine: str = None) -> str:
        """Transcribe audio file using specified or preferred engine."""
        engine = engine or self.current_engine
        
        if engine == "faster_whisper":
            return self._transcribe_faster_whisper(audio_path)
        elif engine == "vosk":
            return self._transcribe_vosk(audio_path)
        elif engine == "speech_recognition":
            return self._transcribe_speech_recognition(audio_path)
        elif engine == "openai_whisper":
            return self._transcribe_openai_whisper(audio_path)
        else:
            raise ValueError(f"Unknown engine: {engine}")
    
    def transcribe_file_with_fallback(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe with fallback engines and return results."""
        results = {}
        
        # Try primary engine first
        try:
            text = self.transcribe_file(audio_path)
            results[self.current_engine] = {
                "text": text,
                "success": True,
                "engine": self.current_engine,
                "confidence": "high" if self.current_engine in ["faster_whisper", "openai_whisper"] else "medium"
            }
        except Exception as e:
            results[self.current_engine] = {
                "text": "",
                "success": False,
                "error": str(e),
                "engine": self.current_engine
            }
            
            # Try fallback engines
            fallback_engines = ["speech_recognition", "vosk", "openai_whisper", "faster_whisper"]
            for engine in fallback_engines:
                if engine != self.current_engine and ENGINES[engine]["available"]:
                    try:
                        text = self.transcribe_file(audio_path, engine)
                        results[engine] = {
                            "text": text,
                            "success": True,
                            "engine": engine,
                            "confidence": "high" if engine in ["faster_whisper", "openai_whisper"] else "medium"
                        }
                        break  # Stop at first successful fallback
                    except Exception as e:
                        results[engine] = {
                            "text": "",
                            "success": False,
                            "error": str(e),
                            "engine": engine
                        }
        
        return results
    
    def _transcribe_faster_whisper(self, audio_path: Path) -> str:
        """Transcribe using Faster Whisper."""
        segments, info = self.model.transcribe(str(audio_path), beam_size=5)
        
        # Combine all segments
        text = " ".join(segment.text for segment in segments)
        return text.strip()
    
    def _transcribe_vosk(self, audio_path: Path) -> str:
        """Transcribe using Vosk."""
        # Convert audio to WAV if needed
        if audio_path.suffix != '.wav':
            wav_path = self._convert_to_wav(audio_path)
        else:
            wav_path = audio_path
        
        # Read audio
        wf = wave.open(str(wav_path), 'rb')
        rec = vosk.KaldiRecognizer(self.model, wf.getframerate())
        
        # Process audio
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = rec.Result()
                return json.loads(result)['text']
        
        # Get final result
        final_result = rec.FinalResult()
        return json.loads(final_result)['text']
    
    def _transcribe_speech_recognition(self, audio_path: Path) -> str:
        """Transcribe using SpeechRecognition."""
        r = self.model
        
        with sr.AudioFile(str(audio_path)) as source:
            audio = r.record(source)
        
        # Try different backends
        backends = ["google", "sphinx"]  # Add more as needed
        
        for backend in backends:
            try:
                if backend == "google":
                    text = r.recognize_google(audio)
                elif backend == "sphinx":
                    text = r.recognize_sphinx(audio)
                else:
                    continue
                
                return text
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                self.logger.warning(f"Backend {backend} failed: {e}")
                continue
        
        raise RuntimeError("All SpeechRecognition backends failed")
    
    def _transcribe_openai_whisper(self, audio_path: Path) -> str:
        """Transcribe using OpenAI Whisper."""
        result = self.model.transcribe(str(audio_path))
        return result["text"].strip()
    
    def _convert_to_wav(self, audio_path: Path) -> Path:
        """Convert audio to WAV format if needed."""
        # This is a simplified version - you might want to use ffmpeg for better conversion
        wav_path = audio_path.with_suffix('.wav')
        
        if audio_path.suffix == '.mp3':
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(str(audio_path))
                audio.export(str(wav_path), format="wav")
                return wav_path
            except ImportError:
                raise RuntimeError("pydub required for MP3 conversion")
        
        return audio_path  # Assume it's already WAV or convertible
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about available engines."""
        return {
            "current_engine": self.current_engine,
            "model_size": self.model_size,
            "available_engines": {k: v for k, v in ENGINES.items() if v["available"]},
            "all_engines": ENGINES
        }

# Test function
def test_local_transcription():
    """Test local transcription with a sample audio file."""
    print("=== Testing Local Speech Recognition ===")
    
    try:
        transcriber = LocalTranscriber()
        info = transcriber.get_engine_info()
        
        print(f"Current engine: {info['current_engine']}")
        print(f"Model size: {info['model_size']}")
        print(f"Available engines: {list(info['available_engines'].keys())}")
        
        # Test with a simple audio file if available
        test_audio = BASE_DIR / "test_audio.wav"
        if test_audio.exists():
            print(f"Testing with: {test_audio}")
            results = transcriber.transcribe_file_with_fallback(test_audio)
            
            for engine, result in results.items():
                if result["success"]:
                    print(f"✅ {engine}: {result['text']}")
                else:
                    print(f"❌ {engine}: {result['error']}")
        else:
            print("No test audio file found. Create one to test transcription.")
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_local_transcription()
