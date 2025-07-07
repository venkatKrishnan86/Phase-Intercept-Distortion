import numpy as np
import torch
from scipy.signal import hilbert

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

def apply_augmentation_phase_intercept_distortion(
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