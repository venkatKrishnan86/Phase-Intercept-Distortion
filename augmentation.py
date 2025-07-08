import numpy as np
import torch
import librosa
import os
from argparse import ArgumentParser
from scipy.signal import hilbert
from scipy.io import wavfile

def phase_intercept_distortion(audio: np.ndarray, theta: float):
    """
        Arguments:
        -----------
        audio: np.ndarray
            A numpy array representing the audio signal with shape (*, samples).

        theta: float
            The phase intercept distortion angle in radians.
    """
    assert len(audio.shape) >= 1, \
        "The audio array must be atleast 1 dimensional of the form (*, samples)"
    return np.real(hilbert(audio, axis=-1) * np.exp(1j*theta))

class PhaseInterceptDistortion(torch.nn.Module):
    """
    A PyTorch module to apply phase intercept distortion to audio signals by applying 
    a frequency-independent phase shift to the input audio tensor.
    """
    def __init__(self):
        super(PhaseInterceptDistortion, self).__init__()

    def forward(
            self, 
            audio: torch.Tensor, 
            theta: torch.Tensor = None
        ):
        """
            Arguments:
            -----------
            audio: torch.Tensor
                A 2D tensor representing the audio signal with shape (channels, samples).

            theta: torch.Tensor
                A 0D tensor or a 1D tensor of phase intercept distortion angles in radians.
                If 1D, it should have shape (batch_size,).
                If None, defaults to random angles in the range [-pi, pi] with a shape of (batch_size,).
                The phase intercept distortion angle in radians.
        """

        assert len(audio.shape) >= 2, \
            "The audio tensor must be 2 dimensional of the form (channels, samples) or 3 dimensional of the form (batch_size, channels, samples)"
        assert len(theta.shape) == 1 and theta.shape[0] == audio.shape[0], \
            "The theta tensor must be 1 dimensional of the form (batch_size,)"
        
        if theta is None:
            theta = torch.rand(audio.shape[0]) * 2 * np.pi - np.pi

        X = torch.fft.fft(audio, dim=-1, norm='ortho')
        num_samples = X.shape[-1]
        theta = theta.to(audio.device).unsqueeze(-1)

        if len(X.shape) == 2:
            X[:, :num_samples//2] *= torch.exp(theta * torch.tensor(1j).to(audio.device))
            X[:, num_samples//2:] *= torch.exp(-theta * torch.tensor(1j).to(audio.device))
        elif len(X.shape) == 3:
            X[:, :, :num_samples//2] *= torch.exp(theta * torch.tensor(1j).to(audio.device)).unsqueeze(1)
            X[:, :, num_samples//2:] *= torch.exp(-theta * torch.tensor(1j).to(audio.device)).unsqueeze(1)
        else:
            raise ValueError("The audio tensor must be 2 or 3 dimensional.")
        return torch.fft.ifft(X, dim=-1, norm='ortho').real

if __name__ == "__main__":
    parser = ArgumentParser(description="Apply phase intercept distortion to a single audio.")
    parser.add_argument("audio_path", type=str, help="Path to the audio file.")
    parser.add_argument("output_loc", type=str, help="Path to the output audio file.")
    parser.add_argument("--theta", type=float, default=-np.pi/2, help="Phase intercept distortion angle in radians (Default: -pi/2).")
    args = parser.parse_args()

    # Load audio file
    print(f"Augmenting audio from {args.audio_path} with theta={args.theta:.3f} radians")
    audio, sample_rate = librosa.load(args.audio_path, sr=None, mono=False)

    # Apply augmentation
    augmented_audio = phase_intercept_distortion(audio, args.theta)

    # Save the augmented audio
    output_loc = os.path.join(args.output_loc, os.path.basename(args.audio_path))
    print(f"Saving augmented audio to {output_loc}")
    os.makedirs(os.path.dirname(output_loc), exist_ok=True)
    wavfile.write(output_loc, sample_rate, augmented_audio.T.astype(np.float32))