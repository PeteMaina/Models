# Installation of necessary dependancies

pip install torch torchaudio transformers datasets librosa soundfile matplotlib gradio pydub nltk scipy
pip install pyttsx3

!python advanced_speech_system.py

# Importing necessary libraries

import os
import time
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan
)
from datasets import load_dataset
import gradio as gr
from pydub import AudioSegment
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import time
import gc

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

class AdvancedSpeechSystem:
    def __init__(self, use_gpu=True):
        """
        Initialize the Advanced Speech System with both TTS and STT capabilities
        
        Args:
            use_gpu: Whether to use GPU if available
        """
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"Using device: {self.device}")
        
        self.tts_model = None
        self.tts_processor = None
        self.vocoder = None
        self.speaker_embeddings = None
        
        self.stt_model = None
        self.stt_processor = None
        
        # Cache to avoid reloading models
        self.models_loaded = {"tts": False, "stt": False}
        
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        
    def _load_tts_model(self):
        """Load Text-to-Speech model components"""
        if self.models_loaded["tts"]:
            return
            
        print("Loading TTS models...")
        start_time = time.time()
        
        # Load SpeechT5 model and processor
        self.tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        
        # Load HiFi-GAN vocoder for high-quality audio synthesis
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
        # Load speaker embeddings from a multi-speaker dataset
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)
        
        self.models_loaded["tts"] = True
        print(f"TTS models loaded in {time.time() - start_time:.2f} seconds")
    
    def _load_stt_model(self):
        """Load Speech-to-Text model components"""
        if self.models_loaded["stt"]:
            return
            
        print("Loading STT models...")
        start_time = time.time()
        
        # Load Whisper large-v3 model and processor for state-of-the-art speech recognition
        self.stt_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.stt_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(self.device)
        
        self.models_loaded["stt"] = True
        print(f"STT models loaded in {time.time() - start_time:.2f} seconds")
    
    def text_to_speech(self, text, speaker_style="default", output_path=None):
        """
        Convert text to speech using advanced neural TTS
        
        Args:
            text: Text to convert to speech
            speaker_style: Style of speaker voice
            output_path: Path to save audio file (optional)
            
        Returns:
            Path to the generated audio file
        """
        self._load_tts_model()
        
        # Process long text by splitting into sentences
        sentences = sent_tokenize(text)
        audio_segments = []
        
        print(f"Processing {len(sentences)} sentence(s)...")
        
        for i, sentence in enumerate(sentences):
            print(f"Processing sentence {i+1}/{len(sentences)}")
            
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Preprocess the text input
            inputs = self.tts_processor(text=sentence, return_tensors="pt").to(self.device)
            
            # Generate speech with speaker embedding for voice characteristics
            speech = self.tts_model.generate_speech(
                inputs["input_ids"], 
                self.speaker_embeddings, 
                vocoder=self.vocoder
            )
            
            # Convert to numpy and store
            speech_np = speech.cpu().numpy()
            audio_segments.append(speech_np)
            
            # Free up memory
            torch.cuda.empty_cache() if self.device == "cuda" else None
        
        # Combine all segments with small pauses between sentences
        if audio_segments:
            # Add small silence between sentences (0.3 seconds)
            sample_rate = 16000
            silence = np.zeros(int(0.3 * sample_rate))
            
            combined_audio = np.concatenate([
                segment for pair in zip(
                    audio_segments, 
                    [silence] * len(audio_segments)
                ) for segment in pair
            ][:-1])  # Remove the last silence
            
            # Apply audio enhancements
            combined_audio = self._enhance_audio(combined_audio)
            
            # Generate output path if not provided
            if output_path is None:
                timestamp = int(time.time())
                output_path = f"output/tts_output_{timestamp}.wav"
            
            # Save the audio file
            sf.write(output_path, combined_audio, sample_rate)
            print(f"Audio saved to {output_path}")
            
            return output_path
        else:
            print("No valid text to convert to speech")
            return None
    
    def speech_to_text(self, audio_path, language="english", task="transcribe"):
        """
        Convert speech to text using advanced neural STT
        
        Args:
            audio_path: Path to audio file
            language: Language of the audio
            task: Either 'transcribe' or 'translate' (to English)
            
        Returns:
            Transcribed or translated text
        """
        self._load_stt_model()
        
        print(f"Transcribing audio from {audio_path}...")
        start_time = time.time()
        
        # Load and preprocess the audio
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
        
        # Apply audio preprocessing for improved recognition
        speech_array = self._preprocess_audio(speech_array)
        
        # Convert audio to required format
        input_features = self.stt_processor(
            speech_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Generate token ids
        predicted_ids = self.stt_model.generate(
            input_features,
            language=language,
            task=task
        )
        
        # Decode token ids to text
        transcription = self.stt_processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        print(f"Transcription completed in {time.time() - start_time:.2f} seconds")
        return transcription
    
    def _enhance_audio(self, audio):
        """
        Apply audio enhancement techniques to improve quality
        
        Args:
            audio: Audio array
            
        Returns:
            Enhanced audio array
        """
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        # Apply subtle compression for better dynamics
        threshold = 0.3
        ratio = 4.0
        audio_compressed = np.copy(audio)
        mask = np.abs(audio) > threshold
        audio_compressed[mask] = threshold + (np.abs(audio[mask]) - threshold) / ratio * np.sign(audio[mask])
        
        return audio_compressed
    
    def _preprocess_audio(self, audio):
        """
        Preprocess audio for better speech recognition
        
        Args:
            audio: Audio array
            
        Returns:
            Preprocessed audio array
        """
        # Apply noise reduction
        # This is a simple high-pass filter to reduce low-frequency noise
        from scipy import signal
        
        # Design a highpass filter to remove frequencies below 80 Hz
        b, a = signal.butter(3, 80 / 8000, 'highpass')
        audio = signal.filtfilt(b, a, audio)
        
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        return audio
    
    def voice_conversion(self, source_audio_path, target_text=None, output_path=None):
        """
        Perform voice conversion - convert audio to text and then back to speech
        with potentially different voice characteristics
        
        Args:
            source_audio_path: Path to source audio
            target_text: Optional text override (if None, uses transcribed text)
            output_path: Path to save converted audio
            
        Returns:
            Path to converted audio file and transcribed text
        """
        # First transcribe the audio
        transcribed_text = self.speech_to_text(source_audio_path)
        
        # Use transcribed text or override with target text
        text_to_convert = target_text if target_text is not None else transcribed_text
        
        # Convert back to speech
        converted_audio_path = self.text_to_speech(text_to_convert, output_path=output_path)
        
        return converted_audio_path, transcribed_text

    def analyze_audio(self, audio_path):
        """
        Analyze audio file and display properties
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of audio properties
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        
        # Calculate audio properties
        rms = librosa.feature.rms(y=audio)[0]
        mean_rms = np.mean(rms)
        
        # Get tempo
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Get spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        mean_spectral_centroid = np.mean(spectral_centroids)
        
        return {
            "duration": duration,
            "sample_rate": sr,
            "mean_volume": mean_rms,
            "tempo_bpm": tempo,
            "spectral_centroid": mean_spectral_centroid,
            "num_samples": len(audio)
        }

    def cleanup(self):
        """Release memory by clearing models"""
        self.tts_model = None
        self.tts_processor = None
        self.vocoder = None
        self.speaker_embeddings = None
        
        self.stt_model = None
        self.stt_processor = None
        
        self.models_loaded = {"tts": False, "stt": False}
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if self.device == "cuda" else None
        print("Models unloaded and memory cleared")

def create_gradio_interface():
    """Create a Gradio interface for the speech system"""
    
    speech_system = AdvancedSpeechSystem()
    
    with gr.Blocks(title="Advanced Speech System") as app:
        gr.Markdown("# Advanced Speech System")
        gr.Markdown("### High-quality Text-to-Speech and Speech-to-Text")
        
        with gr.Tab("Text to Speech"):
            with gr.Row():
                tts_input = gr.Textbox(label="Text to Convert", lines=5, placeholder="Enter text to convert to speech...")
                tts_output = gr.Audio(label="Generated Speech", type="filepath")
            
            tts_button = gr.Button("Generate Speech")
            tts_button.click(speech_system.text_to_speech, inputs=tts_input, outputs=tts_output)
        
        with gr.Tab("Speech to Text"):
            with gr.Row():
                stt_input = gr.Audio(label="Audio Input", type="filepath")
                stt_output = gr.Textbox(label="Transcription", lines=5)
            
            language_input = gr.Dropdown(
                label="Language",
                choices=["english", "spanish", "french", "german", "chinese", "japanese"],
                value="english"
            )
            
            stt_button = gr.Button("Transcribe Audio")
            stt_button.click(
                speech_system.speech_to_text, 
                inputs=[stt_input, language_input], 
                outputs=stt_output
            )
        
        with gr.Tab("Voice Conversion"):
            with gr.Row():
                vc_input = gr.Audio(label="Source Audio", type="filepath")
                vc_text = gr.Textbox(label="Transcribed/Modified Text", lines=3)
                vc_output = gr.Audio(label="Converted Speech", type="filepath")
            
            vc_button = gr.Button("Convert Voice")
            
            def voice_conversion_handler(audio_path, text_override=None):
                if text_override and text_override.strip():
                    output_path, transcribed = speech_system.voice_conversion(audio_path, text_override)
                    return output_path, transcribed
                else:
                    output_path, transcribed = speech_system.voice_conversion(audio_path)
                    return output_path, transcribed
            
            vc_button.click(
                voice_conversion_handler,
                inputs=[vc_input, vc_text],
                outputs=[vc_output, vc_text]
            )

        with gr.Tab("Audio Analysis"):
            with gr.Row():
                analysis_input = gr.Audio(label="Audio to Analyze", type="filepath")
                analysis_output = gr.JSON(label="Audio Properties")
            
            analysis_button = gr.Button("Analyze Audio")
            analysis_button.click(speech_system.analyze_audio, inputs=analysis_input, outputs=analysis_output)
            
    return app

# Example usage
if __name__ == "__main__":
    # Launch Gradio interface
    app = create_gradio_interface()
    app.launch()
    
    # Alternative: Use the speech system directly
    """
    speech_system = AdvancedSpeechSystem()
    
    # Text to Speech example
    tts_output = speech_system.text_to_speech(
        "This is an example of high-quality text to speech synthesis "
        "that aims to sound more natural and expressive than existing systems. "
        "The voice should have appropriate intonation and rhythm."
    )
    
    # Speech to Text example (assuming you have an audio file)
    # transcription = speech_system.speech_to_text("path/to/your/audio.wav")
    # print(f"Transcription: {transcription}")
    
    # Cleanup
    speech_system.cleanup()
    """