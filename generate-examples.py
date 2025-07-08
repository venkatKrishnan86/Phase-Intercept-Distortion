import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import glob
from augmentation import phase_intercept_distortion

# Seeding for reproducibility
np.random.seed(10)

def generate_examples(
        originals = "examples/originals",
        distorted = "examples/distorted",
        plots = "examples/plots",
    ):
    """
    Generate examples of phase intercept distortion.
    
    Arguments:
    ----------
    originals: str
        Path to the directory where original audio files exist
    
    distorted: str
        Path to the directory where distorted audio files will be saved.
    
    plots: int
        Path to the directory where the plot files will be saved.
    """
    if not os.path.exists(originals):
        raise ValueError(f"Originals directory {originals} does not exist.")
    if not os.path.exists(distorted):
        os.makedirs(distorted, exist_ok=True)
    else:
        files = glob.glob(distorted + "/*")
        for f in files:
            os.remove(f)
    if not os.path.exists(plots):
        os.makedirs(plots, exist_ok=True)
    else:
        files = glob.glob(plots + "/*")
        for f in files:
            os.remove(f)

    for original in os.listdir(originals):
        if not original.endswith('.wav'):
            continue
        original_path = os.path.join(originals, original)
        sr, audio = wavfile.read(original_path)
        curr_type = audio.dtype

        if curr_type != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(curr_type).max

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        elif audio.ndim == 2:
            audio = audio.T
        elif audio.ndim > 2:
            raise ValueError(f"Audio file {original} has more than 2 dimensions.")

        time_axis = np.arange(audio.shape[1]) / sr

        # Generating a random phase intercept distortion value
        theta = np.random.rand() * 2 * np.pi - np.pi
        theta_degrees = np.degrees(theta)

        # Applying phase intercept distortion
        distorted_audio = phase_intercept_distortion(audio, theta)

        # Saving the distorted audio
        distorted_path = os.path.join(distorted, original.replace('.wav', '') + f"_distorted_{theta_degrees:.2f}.wav")
        wavfile.write(distorted_path, sr, distorted_audio.T)

        # Plotting original and distorted audio
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.title(f"Original Audio: {original}")
        plt.plot(time_axis, audio[0])

        plt.subplot(2, 1, 2)
        plt.title(f"Distorted Audio (θ={theta_degrees:.2f} degrees): {original}")
        plt.plot(time_axis, distorted_audio[0])

        plt.tight_layout()

        plt.savefig(os.path.join(plots, original.replace('.wav', '') + f"_plot_{theta_degrees:.2f}.png"))
        plt.close()

if __name__ == "__main__":
    generate_examples()