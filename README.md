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

## Module Overview – Assignment 1 (Time & Frequency Domain)

| File | Purpose |
|------|---------|
| `experimental_runner.py` | Main orchestrator – runs all 7 experiments |
| `audio_io.py` | Audio loading, resampling, quantization, SNR |
| `fourier_analysis.py` | FFT, naïve DFT, time/phase shift helpers |
| `stft_analysis.py` | STFT computation and resolution metrics |
| `effects.py` | Hard/soft clipping, downsampling, aliasing |
| `main.py` | CLI entry-point, plotting & directory utilities |

---

## Assignment 2 – Speech-to-Text & Text-to-Speech

This section covers **Part 1 (STT)** and **Part 2 (TTS)** of Assignment 2.
All new scripts are self-contained and do **not** modify the Assignment 1 code.

### Audio file

Place `Speaking_Female.wav` at `Audio files/Speaking_Female.wav` before running the STT
scripts.  If the file is absent the scripts automatically use a built-in
synthetic speech-like signal so all plots are still produced.

### Quick-run – regenerate ALL Assignment 2 results

```bash
# Part 1 – STT: A/B experiments for every parameter (Table 1, Figures 1–8)
python run_stt_experiments.py "Audio files/Speaking_Female.wav"

# Part 1 – Multi-parameter best/worst experiment (Table 2, Figures 9–11)
python run_multi_param_experiment.py "Audio files/Speaking_Female.wav"

# Part 2 – TTS parameter sweep for all 10 parameters (Table 3, Figures 12–31)
python run_tts_experiments.py

# Generate an auto-filled report with Tables 1–3 + ordered figure embeds
python generate_assignment2_report.py
```

Generated report path:
`outputs/assignment2/CS425_Assignment2_Report.md`

For TTS synthesis with real speech (instead of the synthetic fallback) install
`pyttsx3` and, on Linux, the `espeak-ng` system engine:

```bash
pip install SpeechRecognition pyttsx3
sudo apt-get install espeak-ng   # Linux only
```

### Assignment 2 output structure

```
outputs/
├── stt/
│   ├── pre_emphasis_0.0.png          ← Figure 1
│   ├── pre_emphasis_0.97.png         ← Figure 2
│   ├── noise_level_0.0.png           ← Figure 3
│   ├── noise_level_0.01.png          ← Figure 4
│   ├── speed_factor_1.0.png          ← Figure 5
│   ├── speed_factor_1.25.png         ← Figure 6
│   ├── pitch_steps_0.png             ← Figure 7
│   ├── pitch_steps_2.png             ← Figure 8
│   ├── *_transcription.txt           ← STT output text
│   └── multi_param/
│       ├── multi_param_Original.png         ← Figure 9
│       ├── multi_param_Config_A_Best.png    ← Figure 10
│       └── multi_param_Config_B_Worst.png   ← Figure 11
└── tts/
    ├── tts_speech_rate_120.png       ← Figure 12
    ├── tts_speech_rate_220.png       ← Figure 13
    ├── … (Figures 14–31)
    └── *.wav                         ← processed TTS audio files
```

### Report template

Open [`report_template.md`](report_template.md) to find:
- Pre-filled table structures matching Tables 1, 2, and 3 from the assignment
  appendix.
- Inline figure references for every plot (Figures 1–31).
- `[YOUR …]` placeholder cells for your own observations and analysis.
- Instructions for exporting to PDF.

### Assignment 2 module overview

| File | Purpose |
|------|---------|
| `stt_template.py` | STT processing functions + single-run demo |
| `tts_template.py` | TTS synthesis functions + single-run demo |
| `run_stt_experiments.py` | Automated A/B experiments (Table 1) |
| `run_multi_param_experiment.py` | Best/worst multi-param experiment (Table 2) |
| `run_tts_experiments.py` | TTS parameter sweep (Table 3) |
| `report_template.md` | Markdown report template with tables & figure callouts |

---

## Dependencies

See [`requirements.txt`](requirements.txt):

* `numpy` – numerical computing
* `scipy` – signal processing
* `matplotlib` – plotting
* `librosa` – audio analysis
* `soundfile` – WAV I/O
* `pandas` – CSV output
* `SpeechRecognition` *(optional)* – STT via Google API
* `pyttsx3` *(optional)* – local TTS engine (requires `espeak-ng` on Linux)
