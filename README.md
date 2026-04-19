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

## Viewing Outputs in GitHub Codespaces

After running `python experimental_runner.py`, inspect everything with:

```bash
python view_outputs.py
```

This prints the directory tree, all CSV tables, the JSON summary, and
lists every plot and audio file with file sizes and audio metadata.

You can also show only one section at a time:

```bash
python view_outputs.py --section tree    # directory structure
python view_outputs.py --section plots   # PNG plot list
python view_outputs.py --section audio   # WAV file list + metadata
python view_outputs.py --section csv     # all CSV tables
python view_outputs.py --section json    # full JSON summary
```

### Opening files in the Codespaces editor / browser

| File type | How to open |
|-----------|-------------|
| **PNG plots** | Explorer panel → `outputs/plots/` → right-click any `.png` → **Open Preview**; or run `code outputs/plots/01_sampling_rate.png` |
| **CSV tables** | Explorer panel → `outputs/report_data/` → click any `.csv` (renders as a spreadsheet) |
| **JSON summary** | Explorer panel → `outputs/report_data/all_experiments_results.json` → click to open |
| **WAV audio** | Explorer panel → `outputs/audio/` → click any `.wav` to play in the built-in audio player |

### Complete output file guide

```
outputs/
├── plots/                            ← PNG images at 300 dpi
│   ├── 01_sampling_rate.png          Waveforms/spectra at 44100/22050/8000 Hz
│   ├── 02_quantization.png           Quantisation noise at 16/8/4 bits
│   ├── 03_time_phase_shift.png       Time-domain delay and phase-shift effects
│   ├── 04_stft_comparison.png        STFT spectrograms (various window/hop sizes)
│   ├── 05_clipping.png               Hard and soft clipping distortion
│   ├── 06_aliasing.png               Aliasing artefacts from downsampling
│   └── 07_dft_vs_fft.png             DFT vs FFT execution-time comparison
│
├── audio/                            ← Processed WAV files
│   ├── 01_sr_44100hz.wav             Original sample rate
│   ├── 01_sr_22050hz.wav             Downsampled to 22 050 Hz
│   ├── 01_sr_8000hz.wav              Downsampled to 8 000 Hz (telephone quality)
│   ├── 02_quantized_16bit.wav        16-bit quantisation (CD quality)
│   ├── 02_quantized_8bit.wav         8-bit quantisation
│   ├── 02_quantized_4bit.wav         4-bit quantisation (clearly audible noise)
│   └── … (20+ additional files)
│
└── report_data/                      ← Data tables and summary
    ├── 01_sampling_rate_results.csv  SNR and Nyquist frequency per sample rate
    ├── 02_quantization_results.csv   SNR and dynamic range per bit depth
    ├── 03_time_phase_shift_results.csv Phase angle and SNR per time shift
    ├── 04_stft_results.csv           Time/frequency resolution per window config
    ├── 05_clipping_results.csv       Distortion dB and clipped fraction
    ├── 06_aliasing_results.csv       Nyquist violations and SNR
    ├── 07_dft_vs_fft_results.csv     Execution times and FFT speed-up
    └── all_experiments_results.json  Full JSON archive of all experiment results
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