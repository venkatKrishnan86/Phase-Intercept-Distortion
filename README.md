# Phase Intercept Distortion

## Introduction

Phase intercept distortion is a form of *phase distortion*, created by an operation called the frequency-independent phase shift.
The transfer function of a frequency-independent phase shift of $\theta$ is defined as:
$$
|H(\omega)|= 1
$$

$$
\angle H(\omega) = \theta \cdot sgn(\omega)
$$

The paper, [The Perception of Phase Intercept Distortion and its Application in Data Augmentation](https://www.arxiv.org/abs/2506.14571), presents evidence that the special case of *phase-intercept distortion* is not perceptible in real-world sounds, and shows how this fact can be leveraged for data augmentation in audio-based machine learning applications.
This repository provides a `numpy` and `torch` implementation of applying this phase intercept distortion to -
1. Single audio array in `numpy`
2. Batch of training samples in `torch` for data augmentation

## Example Usage
Prior to testing, we recommend creating a virtual environment, and then run -
```bash
pip install -r requirements.txt
```
### Phase Intercept Distortion examples
We have added some example audios from [AudioSet](https://research.google.com/audioset/) in the directory: `examples/originals`.
To test the effect of phase intercept distortion on these samples, run -
```bash
python generate-examples.py
```
- The modified/distorted audio will be saved inside the `examples/distorted` directory.
- The original and distorted audio plots will be saved inside `examples/plots` as `.png` files.


### Applying Phase Intercept Distortion to a single audio
To apply phase intercept distortion to your own audio files, use:
```bash
python augmentation.py /PATH/to/audio.wav /PATH/to/output/directory
```

This will apply the *Hilbert Transform* by default (phase intercept distortion angle of $-\pi/2$) to the audio file and save the augmented audio in the specified output directory.
If you want to use a different phase intercept distortion angle, you can specify it using the `--theta` flag:
```bash
python augmentation.py /PATH/to/audio.wav /PATH/to/output/directory --theta <angle_in_radians>
```

### Applying Phase Intercept Distortion for Data Augmentation (PyTorch)
