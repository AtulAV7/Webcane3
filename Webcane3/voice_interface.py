"""
Voice Interface for WebCane3 Accessibility.
Provides Speech-to-Text (STT) and Text-to-Speech (TTS) for blind users.
Uses Groq API for both Whisper (STT) and Orpheus (TTS).
"""

import os
import time
import threading
import tempfile
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

# Groq client
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from .config import Config


class VoiceInterface:
    """
    Voice interface for blind users.
    - STT: Record voice and transcribe to text
    - TTS: Convert text to speech and play (non-blocking)
    """
    
    # Voice settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    RECORDING_DURATION = 5  # seconds
    
    def __init__(self, api_key: str = None):
        """
        Initialize voice interface.
        
        Args:
            api_key: Groq API key (uses GROQ_API_KEY2 from config)
        """
        self.client = None
        self.available = False
        self.tts_available = False
        self.stt_available = False
        
        # Audio file paths
        self.temp_dir = tempfile.gettempdir()
        self.recording_path = os.path.join(self.temp_dir, "webcane_recording.wav")
        self.speech_path = os.path.join(self.temp_dir, "webcane_speech.wav")
        
        # Thread-safe flag for TTS queue
        self._speaking = False
        self._speech_queue = []
        
        if not GROQ_AVAILABLE:
            print("[Voice] Groq SDK not available")
            return
        
        try:
            api_key = api_key or Config.GROQ_API_KEY2
            if not api_key:
                print("[Voice] No Groq API key (GROQ_API_KEY2) provided")
                return
            
            self.client = Groq(api_key=api_key)
            self.available = True
            
            # Check capabilities
            self.stt_available = AUDIO_AVAILABLE
            self.tts_available = PYGAME_AVAILABLE
            
            status = []
            if self.stt_available:
                status.append("STT")
            if self.tts_available:
                status.append("TTS")
            
            if status:
                print(f"[Voice] Ready ({', '.join(status)})")
            else:
                print("[Voice] API ready but audio libraries missing")
                
        except Exception as e:
            print(f"[Voice] Setup failed: {e}")
    
    def listen(self, duration: float = None) -> Optional[str]:
        """
        Record audio and transcribe to text.
        
        Args:
            duration: Recording duration in seconds (default: 5)
            
        Returns:
            Transcribed text, or None on failure
        """
        if not self.available or not self.stt_available:
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
            
            # Transcribe with Whisper
            with open(self.recording_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
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
        Convert text to speech and play.
        Non-blocking by default - audio plays in background.
        
        Args:
            text: Text to speak
            blocking: If True, wait for audio to finish
        """
        if not self.available:
            print(f"[Voice] (would say): {text}")
            return
        
        if blocking:
            self._speak_sync(text)
        else:
            # Run in background thread
            thread = threading.Thread(target=self._speak_sync, args=(text,))
            thread.daemon = True
            thread.start()
    
    def _speak_sync(self, text: str):
        """Synchronous TTS - generates and plays audio."""
        # Use unique temp file for each TTS call to avoid lock conflicts
        import uuid
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
            
            # Generate speech using Groq Orpheus (speed 1.25 = slightly faster)
            response = self.client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice="autumn",
                response_format="wav",
                input=text,
                speed=1.25  # Slightly faster (range: 0.5 to 5.0)
            )
            
            # Save to unique file
            response.write_to_file(unique_speech_path)
            
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
                
                # Try to delete temp file
                try:
                    os.remove(unique_speech_path)
                except:
                    pass
            else:
                print(f"[Voice] Audio saved to {unique_speech_path}")
                
        except Exception as e:
            # Don't crash on TTS errors - just print and continue
            error_msg = str(e)
            if "terms acceptance" in error_msg.lower():
                print(f"[Voice] TTS model requires terms acceptance on Groq console")
            elif "permission" in error_msg.lower():
                print(f"[Voice] TTS file permission issue - continuing")
            else:
                print(f"[Voice] TTS error (continuing without audio): {error_msg[:100]}")
    
    def speak_status(self, action: str, target: str = "", success: bool = None):
        """
        Speak a formatted status message.
        
        Args:
            action: Action type (navigating, searching, clicking, etc.)
            target: Target of the action
            success: Whether the action succeeded (None = in progress)
        """
        if success is None:
            # In progress
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
            if os.path.exists(self.speech_path):
                os.remove(self.speech_path)
        except:
            pass
