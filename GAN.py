'''Voice Generation using GANs
A PyTorch implementation of a GAN-based voice generation system.
Designed and trained on the LJSpeech dataset.
Includes training and evaluation functionalities.
copyright: peterwahomemaina003@gmail.com
reach out for more info'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
class Config:
    # Audio parameters
    sample_rate = 22050
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    n_mels = 80
    f_min = 0
    f_max = 8000
    
    # Model parameters
    noise_dim = 128
    text_embed_dim = 256
    gen_hidden_dim = 512
    disc_hidden_dim = 512
    
    # Training parameters
    batch_size = 16
    learning_rate = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    n_epochs = 100
    lambda_gp = 10  # Gradient penalty weight
    lambda_fm = 10  # Feature matching weight
    n_critic = 5    # Train discriminator n times per generator
    
    # Text parameters
    max_text_len = 200
    vocab_size = 256  # ASCII characters

config = Config()

# Text preprocessing utilities
class TextProcessor:
    def __init__(self):
        self.char_to_idx = {chr(i): i for i in range(256)}
        self.idx_to_char = {i: chr(i) for i in range(256)}
    
    def text_to_sequence(self, text: str, max_len: int = config.max_text_len) -> torch.Tensor:
        """Convert text to sequence of indices"""
        sequence = [self.char_to_idx.get(c, 0) for c in text.lower()]
        # Pad or truncate
        if len(sequence) < max_len:
            sequence.extend([0] * (max_len - len(sequence)))
        else:
            sequence = sequence[:max_len]
        return torch.tensor(sequence, dtype=torch.long)

text_processor = TextProcessor()

# Audio preprocessing utilities
def load_audio(path: str) -> torch.Tensor:
    """Load audio file and resample to target sample rate"""
    waveform, sr = torchaudio.load(path)
    if sr != config.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
        waveform = resampler(waveform)
    return waveform[0]  # Take first channel

def audio_to_mel(waveform: torch.Tensor) -> torch.Tensor:
    """Convert waveform to mel spectrogram"""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        f_min=config.f_min,
        f_max=config.f_max
    )
    mel = mel_transform(waveform)
    mel = torch.log(torch.clamp(mel, min=1e-5))  # Log mel spectrogram
    return mel

def mel_to_audio(mel_spec: torch.Tensor) -> torch.Tensor:
    """Convert mel spectrogram back to audio using Griffin-Lim (simplified vocoder)"""
    # This is a simplified vocoder - in practice, you'd use HiFi-GAN or WaveGlow
    mel_spec = torch.exp(mel_spec)  # Convert from log scale
    
    # Griffin-Lim algorithm (simplified implementation)
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=config.n_fft // 2 + 1,
        n_mels=config.n_mels,
        sample_rate=config.sample_rate,
        f_min=config.f_min,
        f_max=config.f_max
    )
    
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_iter=32
    )
    
    spec = inverse_mel(mel_spec)
    waveform = griffin_lim(spec)
    return waveform

# Dataset class for LJSpeech
class LJSpeechDataset(Dataset):
    def __init__(self, data_dir: str, max_length: int = 400):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.csv"
        self.data = []
        
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 3:
                        filename = parts[0]
                        text = parts[2]
                        audio_path = self.data_dir / "wavs" / f"{filename}.wav"
                        if audio_path.exists():
                            self.data.append({
                                'audio_path': str(audio_path),
                                'text': text
                            })
        
        print(f"Loaded {len(self.data)} samples from LJSpeech dataset")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process audio
        waveform = load_audio(item['audio_path'])
        mel_spec = audio_to_mel(waveform)
        
        # Truncate or pad mel spectrogram
        if mel_spec.shape[-1] > self.max_length:
            start_idx = random.randint(0, mel_spec.shape[-1] - self.max_length)
            mel_spec = mel_spec[:, start_idx:start_idx + self.max_length]
        else:
            padding = self.max_length - mel_spec.shape[-1]
            mel_spec = F.pad(mel_spec, (0, padding), value=-11.5)  # Silence value
        
        # Process text
        text_seq = text_processor.text_to_sequence(item['text'])
        
        return {
            'mel_spec': mel_spec,
            'text': text_seq,
            'text_raw': item['text']
        }

# Text Encoder for conditioning
class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, text_seq: torch.Tensor) -> torch.Tensor:
        # text_seq: (batch_size, seq_len)
        embedded = self.embedding(text_seq)  # (batch_size, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)    # (batch_size, seq_len, hidden_dim * 2)
        projected = self.projection(lstm_out)  # (batch_size, seq_len, hidden_dim)
        
        # Global context vector (mean pooling)
        context = torch.mean(projected, dim=1)  # (batch_size, hidden_dim)
        return context, projected

# Generator Network
class Generator(nn.Module):
    def __init__(self, noise_dim: int, text_dim: int, mel_dim: int = 80):
        super().__init__()
        self.noise_dim = noise_dim
        self.text_dim = text_dim
        self.mel_dim = mel_dim
        
        # Text encoder
        self.text_encoder = TextEncoder(config.vocab_size, config.text_embed_dim, text_dim)
        
        # Input projection
        input_dim = noise_dim + text_dim
        self.input_proj = nn.Linear(input_dim, 256 * 8 * 8)
        
        # Upsampling blocks
        self.conv_blocks = nn.ModuleList([
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final layer to get mel_dim channels
            nn.Conv2d(32, mel_dim, 3, 1, 1),
            nn.Tanh()
        ])
        
        # Adaptive pooling to match target size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((mel_dim, 400))
        
    def forward(self, noise: torch.Tensor, text: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = noise.shape[0]
        
        # Text conditioning
        if text is not None:
            text_context, _ = self.text_encoder(text)
            # Combine noise and text
            combined = torch.cat([noise, text_context], dim=1)
        else:
            # Use zero text conditioning if no text provided
            text_context = torch.zeros(batch_size, self.text_dim, device=noise.device)
            combined = torch.cat([noise, text_context], dim=1)
        
        # Project to 2D feature map
        x = self.input_proj(combined)
        x = x.view(batch_size, 256, 8, 8)
        
        # Apply conv blocks
        for layer in self.conv_blocks:
            x = layer(x)
        
        # Reshape to mel spectrogram format
        x = self.adaptive_pool(x)
        x = x.squeeze(1) if x.shape[1] == 1 else x[:, :self.mel_dim]
        
        return x

# Multi-scale Discriminator
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class SingleDiscriminator(nn.Module):
    def __init__(self, mel_dim: int = 80):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Conv2d(mel_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            DiscriminatorBlock(256, 512),
            
            nn.Conv2d(512, 1, 4, 1, 1)
        ])
        
    def forward(self, x):
        features = []
        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(layer, DiscriminatorBlock):
                features.append(x)
        
        # Final layer
        output = self.layers[-1](x)
        return output, features

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, mel_dim: int = 80, num_scales: int = 3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            SingleDiscriminator(mel_dim) for _ in range(num_scales)
        ])
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        outputs = []
        features = []
        
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.pooling(x)
            
            out, feat = disc(x)
            outputs.append(out)
            features.append(feat)
        
        return outputs, features

# Loss functions
def gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Compute gradient penalty for WGAN-GP"""
    batch_size = real_samples.shape[0]
    
    # Random interpolation factor
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolated samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Discriminator output for interpolated samples
    d_interpolated, _ = discriminator(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return penalty

def feature_matching_loss(real_features, fake_features):
    """Compute feature matching loss"""
    loss = 0
    for real_feat_scale, fake_feat_scale in zip(real_features, fake_features):
        for real_feat, fake_feat in zip(real_feat_scale, fake_feat_scale):
            loss += F.l1_loss(fake_feat, real_feat.detach())
    return loss

# Training class
class VoiceGANTrainer:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
        # Initialize models
        self.generator = Generator(
            noise_dim=config.noise_dim,
            text_dim=config.gen_hidden_dim,
            mel_dim=config.n_mels
        ).to(device)
        
        self.discriminator = MultiScaleDiscriminator(
            mel_dim=config.n_mels
        ).to(device)
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        # Dataset and dataloader
        self.dataset = LJSpeechDataset(data_dir)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )
        
        # Training metrics
        self.losses = {
            'g_loss': [],
            'd_loss': [],
            'gp_loss': [],
            'fm_loss': []
        }
    
    def train_discriminator(self, real_mel, fake_mel):
        """Train discriminator for one step"""
        self.optimizer_d.zero_grad()
        
        # Real samples
        real_outputs, real_features = self.discriminator(real_mel)
        real_loss = sum(-torch.mean(out) for out in real_outputs)
        
        # Fake samples
        fake_outputs, fake_features = self.discriminator(fake_mel.detach())
        fake_loss = sum(torch.mean(out) for out in fake_outputs)
        
        # Gradient penalty
        gp_loss = 0
        for i in range(len(real_outputs)):
            gp = gradient_penalty(
                lambda x: self.discriminator(x)[0][i],
                real_mel, fake_mel, device
            )
            gp_loss += gp
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss + config.lambda_gp * gp_loss
        d_loss.backward()
        self.optimizer_d.step()
        
        return d_loss.item(), gp_loss.item()
    
    def train_generator(self, fake_mel, real_mel):
        """Train generator for one step"""
        self.optimizer_g.zero_grad()
        
        # Generator adversarial loss
        fake_outputs, fake_features = self.discriminator(fake_mel)
        g_adv_loss = sum(-torch.mean(out) for out in fake_outputs)
        
        # Feature matching loss
        _, real_features = self.discriminator(real_mel)
        fm_loss = feature_matching_loss(real_features, fake_features)
        
        # Total generator loss
        g_loss = g_adv_loss + config.lambda_fm * fm_loss
        g_loss.backward()
        self.optimizer_g.step()
        
        return g_loss.item(), fm_loss.item()
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {'g_loss': 0, 'd_loss': 0, 'gp_loss': 0, 'fm_loss': 0}
        
        for batch_idx, batch in enumerate(self.dataloader):
            real_mel = batch['mel_spec'].to(device)
            text = batch['text'].to(device)
            batch_size = real_mel.shape[0]
            
            # Generate noise
            noise = torch.randn(batch_size, config.noise_dim, device=device)
            
            # Generate fake mel spectrograms
            fake_mel = self.generator(noise, text)
            
            # Ensure shapes match
            if fake_mel.shape != real_mel.shape:
                fake_mel = F.interpolate(
                    fake_mel.unsqueeze(1), 
                    size=real_mel.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            
            # Train discriminator
            d_loss, gp_loss = self.train_discriminator(real_mel, fake_mel)
            
            # Train generator (every n_critic steps)
            if batch_idx % config.n_critic == 0:
                fake_mel = self.generator(noise, text)
                if fake_mel.shape != real_mel.shape:
                    fake_mel = F.interpolate(
                        fake_mel.unsqueeze(1), 
                        size=real_mel.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(1)
                
                g_loss, fm_loss = self.train_generator(fake_mel, real_mel)
                epoch_losses['g_loss'] += g_loss
                epoch_losses['fm_loss'] += fm_loss
            
            epoch_losses['d_loss'] += d_loss
            epoch_losses['gp_loss'] += gp_loss
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")
        
        # Average losses
        num_batches = len(self.dataloader)
        for key in epoch_losses:
            if key in ['g_loss', 'fm_loss']:
                epoch_losses[key] /= (num_batches // config.n_critic)
            else:
                epoch_losses[key] /= num_batches
            self.losses[key].append(epoch_losses[key])
        
        return epoch_losses
    
    def train(self, num_epochs: int = config.n_epochs):
        """Full training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            epoch_losses = self.train_epoch(epoch)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"D_loss: {epoch_losses['d_loss']:.4f}, "
                  f"G_loss: {epoch_losses['g_loss']:.4f}, "
                  f"GP_loss: {epoch_losses['gp_loss']:.4f}, "
                  f"FM_loss: {epoch_losses['fm_loss']:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
            
            # Generate samples every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.generate_samples(epoch + 1)
    
    def generate_samples(self, epoch: int, num_samples: int = 4):
        """Generate sample audio files"""
        self.generator.eval()
        
        with torch.no_grad():
            # Sample from dataset for text conditioning
            sample_batch = next(iter(self.dataloader))
            sample_texts = sample_batch['text'][:num_samples].to(device)
            sample_text_raw = sample_batch['text_raw'][:num_samples]
            
            # Generate noise
            noise = torch.randn(num_samples, config.noise_dim, device=device)
            
            # Generate mel spectrograms
            fake_mels = self.generator(noise, sample_texts)
            
            # Convert to audio and save
            for i in range(num_samples):
                mel_spec = fake_mels[i].cpu()
                audio = mel_to_audio(mel_spec)
                
                # Save audio file
                output_path = f"generated_epoch_{epoch}_sample_{i}.wav"
                torchaudio.save(output_path, audio.unsqueeze(0), config.sample_rate)
                
                print(f"Generated: {output_path}")
                print(f"Text: {sample_text_raw[i]}")
        
        self.generator.train()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'losses': self.losses
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.losses = checkpoint['losses']
        print(f"Checkpoint loaded: {filename}")

# Evaluation functions
class VoiceGANEvaluator:
    def __init__(self, generator_path: str):
        self.generator = Generator(
            noise_dim=config.noise_dim,
            text_dim=config.gen_hidden_dim,
            mel_dim=config.n_mels
        ).to(device)
        
        # Load trained generator
        checkpoint = torch.load(generator_path, map_location=device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.generator.eval()
    
    def text_to_speech(self, text: str, output_path: str = "generated_tts.wav"):
        """Generate speech from text"""
        with torch.no_grad():
            # Process text
            text_seq = text_processor.text_to_sequence(text).unsqueeze(0).to(device)
            
            # Generate noise
            noise = torch.randn(1, config.noise_dim, device=device)
            
            # Generate mel spectrogram
            mel_spec = self.generator(noise, text_seq)
            
            # Convert to audio
            audio = mel_to_audio(mel_spec[0].cpu())
            
            # Save audio
            torchaudio.save(output_path, audio.unsqueeze(0), config.sample_rate)
            print(f"Generated TTS audio: {output_path}")
            
            return audio
    
    def unconditional_generation(self, num_samples: int = 5):
        """Generate unconditional speech samples"""
        with torch.no_grad():
            for i in range(num_samples):
                # Generate noise
                noise = torch.randn(1, config.noise_dim, device=device)
                
                # Generate mel spectrogram (no text conditioning)
                mel_spec = self.generator(noise, None)
                
                # Convert to audio
                audio = mel_to_audio(mel_spec[0].cpu())
                
                # Save audio
                output_path = f"unconditional_sample_{i}.wav"
                torchaudio.save(output_path, audio.unsqueeze(0), config.sample_rate)
                print(f"Generated unconditional sample: {output_path}")
    
    def plot_mel_spectrogram(self, mel_spec: torch.Tensor, title: str = "Mel Spectrogram"):
        """Plot mel spectrogram"""
        plt.figure(figsize=(12, 6))
        plt.imshow(mel_spec.numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Mel Frequency Bins')
        plt.tight_layout()
        plt.show()
    
    def evaluate_sample_quality(self, num_samples: int = 10):
        """Generate samples and plot spectrograms for quality assessment"""
        with torch.no_grad():
            for i in range(num_samples):
                # Generate sample
                noise = torch.randn(1, config.noise_dim, device=device)
                mel_spec = self.generator(noise, None)
                
                # Plot spectrogram
                self.plot_mel_spectrogram(
                    mel_spec[0].cpu(), 
                    title=f"Generated Sample {i+1}"
                )

# Example usage and main execution
def main():
    # Set data directory (modify this path for your LJSpeech dataset)
    data_dir = "path/to/LJSpeech-1.1"  # Update this path
    
    print("Voice GAN Training System")
    print("=" * 50)
    
    # Check if dataset exists
    if not Path(data_dir).exists():
        print(f"Dataset not found at {data_dir}")
        print("Please download LJSpeech dataset and update the data_dir path")
        print("You can download it from: https://keithito.com/LJ-Speech-Dataset/")
        return
    
    # Initialize trainer
    trainer = VoiceGANTrainer(data_dir)
    
    # Start training
    try:
        trainer.train(num_epochs=5)  # Start with fewer epochs for testing
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    trainer.save_checkpoint("voice_gan_final.pt")
    
    # Demonstrate evaluation
    print("\nEvaluation Examples:")
    evaluator = VoiceGANEvaluator("voice_gan_final.pt")
    
    # Text-to-speech example
    sample_text = "Hello, this is a test of the voice generation system."
    evaluator.text_to_speech(sample_text, "example_tts.wav")
    
    # Unconditional generation
    evaluator.unconditional_generation(num_samples=3)
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()