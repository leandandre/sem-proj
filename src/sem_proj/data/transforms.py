import torch
import numpy as np


class RandomTimeShift:
    """
    Randomly shift signal in time (circular shift).
    
    Useful for data augmentation during training to make model 
    invariant to small time offsets.
    
    Parameters
    ----------
    max_shift_ratio : float
        Maximum shift as fraction of signal length (e.g., 0.1 = ±10%).
    """
    
    def __init__(self, max_shift_ratio: float = 0.1):
        self.max_shift_ratio = max_shift_ratio
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random time shift.
        
        Parameters
        ----------
        x : torch.Tensor, shape (C, T)
            EEG signal with C channels and T timepoints.
        
        Returns
        -------
        torch.Tensor, shape (C, T)
            Time-shifted signal.
        """
        C, T = x.shape
        
        # Random shift amount (samples)
        max_shift = int(T * self.max_shift_ratio)
        shift = np.random.randint(-max_shift, max_shift + 1)
        
        if shift == 0:
            return x
        
        # Circular shift (roll along time axis)
        return torch.roll(x, shifts=shift, dims=1)


class RandomAmplitudeScale:
    """
    Randomly scale signal amplitude.
    
    Parameters
    ----------
    scale_range : tuple[float, float]
        (min_scale, max_scale). E.g., (0.8, 1.2) means ±20%.
    """
    
    def __init__(self, scale_range: tuple[float, float] = (0.8, 1.2)):
        self.scale_range = scale_range
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random amplitude scaling.
        
        Parameters
        ----------
        x : torch.Tensor, shape (C, T)
            EEG signal.
        
        Returns
        -------
        torch.Tensor, shape (C, T)
            Scaled signal.
        """
        scale = np.random.uniform(*self.scale_range)
        return x * scale


class RandomGaussianNoise:
    """
    Add random Gaussian noise to signal during training.
    
    Useful for data augmentation to improve robustness to noise.
    
    Parameters
    ----------
    noise_scale : tuple[float, float]
        (min_scale, max_scale) for noise magnitude as fraction of epoch std.
        E.g., (0.01, 0.05) means noise_std = uniform(0.01, 0.05) * epoch_std
    """
    
    def __init__(self, noise_scale: tuple[float, float] = (0.01, 0.05)):
        self.noise_scale = noise_scale
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add random Gaussian noise.
        
        Parameters
        ----------
        x : torch.Tensor, shape (C, T)
            EEG signal.
        
        Returns
        -------
        torch.Tensor, shape (C, T)
            Signal with added noise.
        """
        # Compute epoch standard deviation
        epoch_std = x.std()
        
        # Random noise scale factor
        scale_factor = np.random.uniform(*self.noise_scale)
        
        # Noise standard deviation
        noise_std = scale_factor * epoch_std
        
        # Add Gaussian noise
        noise = torch.randn_like(x) * noise_std
        return x + noise


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x