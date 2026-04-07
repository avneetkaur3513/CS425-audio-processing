# CS425 Audio Processing – Assignment 1

Automated experimental framework for **CS425: Time and Frequency Domain Audio
Analysis**.  Run a single command and get publication-quality plots, processed
WAV files, and CSV data tables for all 7 experiments.

---

## Experiments

| # | Title | Key Metrics |
|---|-------|-------------|
| 1 | Sampling Rate | SNR, Nyquist frequency |
| 2 | Quantization | SNR, Dynamic range |
| 3 | Time / Phase Shift | Phase angle, time-domain delay |
| 4 | STFT Parameter Study | Time & frequency resolution |
| 5 | Clipping (hard & soft) | Distortion dB, clipped fraction |
| 6 | Aliasing (downsampling) | Nyquist violations, SNR |
| 7 | DFT vs FFT Complexity | Execution time, speedup |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/avneetkaur3513/CS425-audio-processing.git
cd CS425-audio-processing
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get an audio file

Download any mono or stereo speech recording (WAV/MP3/FLAC) and save it as
`speech.wav` in the project folder.

**Option A – public domain sample (no sign-in required):**

```bash
curl -L https://www2.cs.uic.edu/~i101/SoundFiles/preamble.wav -o speech.wav
```

**Option B – librosa built-in example:**

```python
import librosa, soundfile as sf
y, sr = librosa.load(librosa.ex('trumpet'))
sf.write('speech.wav', y, sr)
```

**Option C – automatic synthetic fallback (no download needed):**

If `speech.wav` is missing or unreadable the runner automatically generates a
5-second synthetic speech-like signal and continues with all 7 experiments.
A warning is printed to stderr so you know which audio source was used.

> **Note:** The freesound.org *direct-download* links require a logged-in
> session; a bare `curl` of a freesound download URL will save an HTML login
> page instead of an audio file, causing a *"Format not recognised"* error.
> Use one of the options above to obtain a real audio file.

### 4. Run all experiments

```bash
python experimental_runner.py
```

Pass a custom file or output directory if needed:

```bash
python experimental_runner.py my_audio.wav --output-dir results
```

---

## Output Structure

```
outputs/
├── plots/
│   ├── 01_sampling_rate.png
│   ├── 02_quantization.png
│   ├── 03_time_phase_shift.png
│   ├── 04_stft_comparison.png
│   ├── 05_clipping.png
│   ├── 06_aliasing.png
│   └── 07_dft_vs_fft.png
├── audio/
│   ├── 01_sr_44100hz.wav
│   ├── 02_quantized_16bit.wav
│   └── … (20+ processed WAV files)
└── report_data/
    ├── 01_sampling_rate_results.csv
    ├── 02_quantization_results.csv
    ├── 03_time_phase_shift_results.csv
    ├── 04_stft_results.csv
    ├── 05_clipping_results.csv
    ├── 06_aliasing_results.csv
    ├── 07_dft_vs_fft_results.csv
    └── all_experiments_results.json
```

---

## Module Overview

| File | Purpose |
|------|---------|
| `experimental_runner.py` | Main orchestrator – runs all 7 experiments |
| `audio_io.py` | Audio loading, resampling, quantization, SNR |
| `fourier_analysis.py` | FFT, naïve DFT, time/phase shift helpers |
| `stft_analysis.py` | STFT computation and resolution metrics |
| `effects.py` | Hard/soft clipping, downsampling, aliasing |
| `main.py` | CLI entry-point, plotting & directory utilities |

---

## Dependencies

See [`requirements.txt`](requirements.txt):

* `numpy` – numerical computing
* `scipy` – signal processing
* `matplotlib` – plotting
* `librosa` – audio analysis
* `soundfile` – WAV I/O
* `pandas` – CSV output