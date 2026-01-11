"""
Voice Interface for WebCane3 Accessibility.
Provides Speech-to-Text (STT) and Text-to-Speech (TTS) for blind users.
- STT: Uses Groq Whisper API
- TTS: Uses NVIDIA Riva TTS (Magpie-Multilingual) via gRPC
"""

import os
import time
import threading
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Optional

# Audio recording
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[Voice] sounddevice/soundfile not installed. Run: pip install sounddevice soundfile")

# Audio playback
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Groq client (for STT)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# NVIDIA Riva TTS
try:
    import riva.client
    from riva.client.proto.riva_audio_pb2 import AudioEncoding
    RIVA_AVAILABLE = True
except ImportError:
    RIVA_AVAILABLE = False
    print("[Voice] nvidia-riva-client not installed. Run: pip install nvidia-riva-client")

from .config import Config


class VoiceInterface:
    """
    Voice interface for blind users.
    - STT: Groq Whisper API (cloud)
    - TTS: NVIDIA Riva Magpie (cloud)
    """
    
    # Voice settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    RECORDING_DURATION = 5  # seconds
    
    # NVIDIA Riva TTS settings
    RIVA_SERVER = "grpc.nvcf.nvidia.com:443"
    RIVA_FUNCTION_ID = "877104f7-e885-42b9-8de8-f6e4c6303969"
    RIVA_VOICE = "Magpie-Multilingual.EN-US.Aria"
    RIVA_LANGUAGE = "en-US"
    RIVA_SAMPLE_RATE = 22050
    
    def __init__(self, api_key: str = None):
        """Initialize voice interface with NVIDIA Riva TTS and Groq STT."""
        self.groq_client = None
        self.riva_service = None
        self.available = False
        self.tts_available = False
        self.stt_available = False
        
        # Audio file paths
        self.temp_dir = tempfile.gettempdir()
        self.recording_path = os.path.join(self.temp_dir, "webcane_recording.wav")
        
        # Initialize Groq for STT
        if GROQ_AVAILABLE:
            try:
                stt_key = api_key or Config.GROQ_API_KEY2
                if stt_key:
                    self.groq_client = Groq(api_key=stt_key)
                    self.stt_available = AUDIO_AVAILABLE
            except Exception as e:
                print(f"[Voice] Groq STT init failed: {e}")
        
        # Initialize NVIDIA Riva TTS
        if RIVA_AVAILABLE and PYGAME_AVAILABLE:
            try:
                tts_key = Config.NVIDIA_API_TTS
                if tts_key:
                    # Create metadata for authentication (list of tuples)
                    metadata = [
                        ("function-id", self.RIVA_FUNCTION_ID),
                        ("authorization", f"Bearer {tts_key}")
                    ]
                    
                    # Create Riva auth with correct parameters
                    auth = riva.client.Auth(
                        ssl_root_cert=None,
                        ssl_client_cert=None,
                        ssl_client_key=None,
                        use_ssl=True,
                        uri=self.RIVA_SERVER,
                        metadata_args=metadata
                    )
                    self.riva_service = riva.client.SpeechSynthesisService(auth)
                    self.tts_available = True
                    print("[Voice] NVIDIA Riva TTS ready")
                else:
                    print("[Voice] NVIDIA_API_TTS not configured")
            except Exception as e:
                print(f"[Voice] NVIDIA Riva TTS init failed: {e}")
        
        # Set overall availability
        self.available = self.stt_available or self.tts_available
        
        if self.available:
            status = []
            if self.stt_available:
                status.append("STT")
            if self.tts_available:
                status.append("TTS-Riva")
            print(f"[Voice] Ready ({', '.join(status)})")
        else:
            print("[Voice] Not available - missing dependencies or API keys")
    
    def listen(self, duration: float = None) -> Optional[str]:
        """
        Record audio and transcribe to text using Groq Whisper.
        """
        if not self.stt_available or not self.groq_client:
            print("[Voice] STT not available")
            return None
        
        duration = duration or self.RECORDING_DURATION
        
        try:
            print(f"[Voice] Recording for {duration} seconds... (speak now)")
            
            # Record audio
            recording = sd.rec(
                int(duration * self.SAMPLE_RATE),
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype='int16'
            )
            sd.wait()
            
            # Save to file
            sf.write(self.recording_path, recording, self.SAMPLE_RATE)
            print("[Voice] Recording complete, transcribing...")
            
            # Transcribe with Whisper via Groq
            with open(self.recording_path, "rb") as audio_file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(self.recording_path, audio_file.read()),
                    model="whisper-large-v3-turbo",
                    temperature=0,
                    response_format="verbose_json"
                )
            
            text = transcription.text.strip()
            print(f"[Voice] Transcribed: \"{text}\"")
            return text
            
        except Exception as e:
            print(f"[Voice] STT error: {e}")
            return None
    
    def speak(self, text: str, blocking: bool = False):
        """
        Convert text to speech using NVIDIA Riva TTS.
        Non-blocking by default - audio plays in background.
        """
        if not self.tts_available:
            print(f"[Voice] (would say): {text}")
            return
        
        if blocking:
            self._speak_sync(text)
        else:
            thread = threading.Thread(target=self._speak_sync, args=(text,))
            thread.daemon = True
            thread.start()
    
    def _speak_sync(self, text: str):
        """Synchronous TTS using NVIDIA Riva."""
        # Use unique temp file for each TTS call
        unique_speech_path = os.path.join(self.temp_dir, f"webcane_speech_{uuid.uuid4().hex[:8]}.wav")
        
        try:
            # Stop any currently playing audio
            if PYGAME_AVAILABLE:
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                except:
                    pass
                time.sleep(0.1)
            
            # Generate speech with NVIDIA Riva
            resp = self.riva_service.synthesize(
                text,
                self.RIVA_VOICE,
                self.RIVA_LANGUAGE,
                sample_rate_hz=self.RIVA_SAMPLE_RATE,
                encoding=AudioEncoding.LINEAR_PCM
            )
            
            # Write to WAV file
            with wave.open(unique_speech_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.RIVA_SAMPLE_RATE)
                wf.writeframesraw(resp.audio)
            
            # Play audio
            if PYGAME_AVAILABLE:
                pygame.mixer.music.load(unique_speech_path)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Unload and clean up
                pygame.mixer.music.unload()
                time.sleep(0.1)
                
                # Delete temp file
                try:
                    os.remove(unique_speech_path)
                except:
                    pass
                    
        except Exception as e:
            error_msg = str(e)
            # Try to get details if available
            try:
                error_msg = e.details()
            except:
                pass
            print(f"[Voice] TTS error: {error_msg}")
    
    def speak_status(self, action: str, target: str = "", success: bool = None):
        """Speak a formatted status message."""
        if success is None:
            if action == "navigating":
                self.speak(f"Navigating to {target}")
            elif action == "searching":
                self.speak(f"Searching for {target}")
            elif action == "clicking":
                self.speak(f"Clicking on {target}")
            elif action == "typing":
                self.speak(f"Typing {target}")
            else:
                self.speak(f"{action} {target}")
        elif success:
            self.speak(f"Done. {action} successful.")
        else:
            self.speak(f"Failed to {action}.")
    
    def announce(self, message: str):
        """Speak an announcement (blocking for important messages)."""
        self.speak(message, blocking=True)
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            if os.path.exists(self.recording_path):
                os.remove(self.recording_path)
        except:
            pass
