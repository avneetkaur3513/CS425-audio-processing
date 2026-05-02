# CS425 – Speech Technology Assignment Report

> **Instructions for use**
>
> 1. Run the experiment scripts (see [How to regenerate all results](#how-to-regenerate-all-results)).
> 2. Copy each spectrogram figure (from `outputs/stt/` and `outputs/tts/`) into
>    the corresponding position below.
> 3. Fill in every `[YOUR …]` placeholder with your own observations and results.
> 4. Delete or replace the *example* entries in italics once you have real data.
> 5. Export the completed file to PDF for submission (e.g. with Pandoc or your
>    word processor).

---

## Part 1 – Speech-to-Text (STT)

### Table 1 – Parameter Analysis

> Place your spectrogram figures **immediately after this table** (Figures 1–8).

| Parameter | Value A | Value B | Effect of A<br>(Audio / Recognition behaviour) | Effect of B<br>(Audio / Recognition behaviour) | Better Value (A/B) + Justification |
|---|---|---|---|---|---|
| **Pre-emphasis** | 0.0 | 0.97 | *[YOUR observation for Value A — e.g. "Baseline signal; no high-freq boost; consonants may be underrepresented in the spectrogram."]* | *[YOUR observation for Value B — e.g. "High-frequency components (s, t, k) visibly boosted in the spectrogram; slight improvement in recognition accuracy."]* | *[YOUR choice — e.g. "B (0.97): standard speech enhancement; improves consonant clarity."]* |
| **Noise Level** | 0.0 | 0.01 | *[YOUR observation for Value A — e.g. "Clean signal; clearest recognition result."]* | *[YOUR observation for Value B — e.g. "Visible broadband noise floor in spectrogram; some recognition errors introduced."]* | *[YOUR choice — e.g. "A (0.0): clean signal yields best STT accuracy."]* |
| **Speed Factor** | 1.0 | 1.25 | *[YOUR observation for Value A — e.g. "Normal pace; spectrogram time-axis matches original duration."]* | *[YOUR observation for Value B — e.g. "25% faster; time-axis compressed; some phonemes reduced; occasional recognition errors."]* | *[YOUR choice — e.g. "A (1.0): natural speed is easiest for the recogniser."]* |
| **Pitch Shift** | 0 | +2 | *[YOUR observation for Value A — e.g. "Natural pitch; recogniser performs normally."]* | *[YOUR observation for Value B — e.g. "Harmonics visibly shifted upward; some confusion in recogniser for voiced phonemes."]* | *[YOUR choice — e.g. "A (0): no pitch distortion preserves original voice characteristics."]* |

---

### Figures for Table 1

> Insert your plots here. Example captions are shown; replace the `![…](…)`
> image paths with your actual plot paths after running the scripts.

**Figure 1: Spectrogram – Pre-emphasis 0.0**
*(File: `outputs/stt/pre_emphasis_0.0.png`)*

![Figure 1 – Pre-emphasis 0.0](outputs/stt/pre_emphasis_0.0.png)

---

**Figure 2: Spectrogram – Pre-emphasis 0.97**
*(File: `outputs/stt/pre_emphasis_0.97.png`)*

![Figure 2 – Pre-emphasis 0.97](outputs/stt/pre_emphasis_0.97.png)

---

**Figure 3: Spectrogram – Noise Level 0.0**
*(File: `outputs/stt/noise_level_0.0.png`)*

![Figure 3 – Noise Level 0.0](outputs/stt/noise_level_0.0.png)

---

**Figure 4: Spectrogram – Noise Level 0.01**
*(File: `outputs/stt/noise_level_0.01.png`)*

![Figure 4 – Noise Level 0.01](outputs/stt/noise_level_0.01.png)

---

**Figure 5: Spectrogram – Speed Factor 1.0**
*(File: `outputs/stt/speed_factor_1.0.png`)*

![Figure 5 – Speed Factor 1.0](outputs/stt/speed_factor_1.0.png)

---

**Figure 6: Spectrogram – Speed Factor 1.25**
*(File: `outputs/stt/speed_factor_1.25.png`)*

![Figure 6 – Speed Factor 1.25](outputs/stt/speed_factor_1.25.png)

---

**Figure 7: Spectrogram – Pitch Shift 0**
*(File: `outputs/stt/pitch_steps_0.png`)*

![Figure 7 – Pitch Shift 0](outputs/stt/pitch_steps_0.png)

---

**Figure 8: Spectrogram – Pitch Shift +2**
*(File: `outputs/stt/pitch_steps_2.png`)*

![Figure 8 – Pitch Shift +2](outputs/stt/pitch_steps_2.png)

---

### Table 2 – Multiple Parameters Settings Experiment

> For each configuration you must set **at least 3 parameters**.
> Place the spectrogram figures (Figures 9–11) **immediately after this table**.

| System Condition | Configuration Used | Sample Output Text | Spectrogram Observation | Explanation of Performance |
|---|---|---|---|---|
| **Original Audio** | No modifications | *[YOUR transcription]* | *[YOUR observation — e.g. "Full frequency range visible; clear harmonic structure; no artefacts."]* | *[YOUR explanation — e.g. "Cleanest condition; baseline for comparison."]* |
| **Processed – Configuration A (best)** | `pre_emphasis=0.97`, `noise=0.0`, `speed=1.0`, `pitch=0` | *[YOUR transcription]* | *[YOUR observation — e.g. "High frequencies boosted; harmonics more visible in upper bands; no added noise."]* | *[YOUR explanation — e.g. "Pre-emphasis enhances consonant clarity; no noise or speed distortion gives best recognition."]* |
| **Processed – Configuration B (worst)** | `pre_emphasis=0.0`, `noise=0.02`, `speed=1.3`, `pitch=+3` | *[YOUR transcription]* | *[YOUR observation — e.g. "Broadband noise floor visible; time-axis compressed; harmonic grid shifted upward."]* | *[YOUR explanation — e.g. "Combination of noise, speed and pitch change degrades all aspects; recognition quality drops significantly."]* |

---

### Figures for Table 2

**Figure 9: Spectrogram – Original Audio (no modifications)**
*(File: `outputs/stt/multi_param/multi_param_Original.png`)*

![Figure 9 – Original](outputs/stt/multi_param/multi_param_Original.png)

---

**Figure 10: Spectrogram – Configuration A (Best)**
*(File: `outputs/stt/multi_param/multi_param_Config_A_Best.png`)*

![Figure 10 – Config A Best](outputs/stt/multi_param/multi_param_Config_A_Best.png)

---

**Figure 11: Spectrogram – Configuration B (Worst)**
*(File: `outputs/stt/multi_param/multi_param_Config_B_Worst.png`)*

![Figure 11 – Config B Worst](outputs/stt/multi_param/multi_param_Config_B_Worst.png)

---

## Part 2 – Text-to-Speech (TTS)

### Table 3 – Parameter Exploration

> Modify one parameter at a time. Place spectrogram figures (Figures 12–31)
> **immediately after this table**, in the same row-order.

| Parameter Changed | Values Tested | Perceptual Observations | Visual Differences in Spectrogram | Explanation (Link to Concepts) |
|---|---|---|---|---|
| **Speech rate** | 120, 220 | *[YOUR observation — e.g. "120 wpm: clear, measured pace; 220 wpm: fast, words may run together."]* | *[YOUR observation — e.g. "220 wpm shows compressed time axis; phoneme segments shorter."]* | *[YOUR explanation — e.g. "Higher rate shortens vowel duration; increases likelihood of mis-segmentation by recogniser."]* |
| **`VOICE_INDEX`** | 0, 1 | *[YOUR observation — e.g. "Voice 0: typical male; Voice 1: female or different accent (system-dependent)."]* | *[YOUR observation — e.g. "Fundamental frequency and formant structure differ between voices."]* | *[YOUR explanation — e.g. "Different vocal tract characteristics produce distinct pitch and formant patterns."]* |
| **Volume** | 0.5, 1.0 | *[YOUR observation — e.g. "0.5: noticeably quieter; 1.0: full loudness."]* | *[YOUR observation — e.g. "Lower amplitude in waveform; spectrogram intensity reduced uniformly."]* | *[YOUR explanation — e.g. "Reducing volume scales amplitude without affecting spectral shape or intelligibility."]* |
| **Pre-emphasis** | 0.0, 0.95 | *[YOUR observation — e.g. "0.95: brighter, sibilants sharper; 0.0: natural, warmer tone."]* | *[YOUR observation — e.g. "0.95: visible high-frequency energy boost above 2 kHz in spectrogram."]* | *[YOUR explanation — e.g. "Pre-emphasis applies a first-order filter y[n]=x[n]-α·x[n-1]; boosts consonant energy."]* |
| **Noise level** | 0.0, 0.02 | *[YOUR observation — e.g. "0.02: audible background hiss; speech harder to hear in quiet sections."]* | *[YOUR observation — e.g. "Broadband noise floor visible across entire spectrogram."]* | *[YOUR explanation — e.g. "Additive Gaussian noise uniformly raises the noise floor, masking low-energy speech details."]* |
| **Pitch shift** | −2, +2 | *[YOUR observation — e.g. "−2: lower, more masculine; +2: higher, more feminine."]* | *[YOUR observation — e.g. "Harmonic bands shift down/up by ~2 semitones (≈12.2%)."]* | *[YOUR explanation — e.g. "Pitch shifting resamples the signal in frequency; preserves duration via interpolation."]* |
| **Time-stretch** | 0.8, 1.3 | *[YOUR observation — e.g. "0.8: slightly slow/drawn out; 1.3: fast and clipped."]* | *[YOUR observation — e.g. "Time axis expands (0.8) or contracts (1.3); harmonic pitch unchanged."]* | *[YOUR explanation — e.g. "Phase vocoder time-stretching modifies duration without altering frequency content."]* |
| **`LOW_CUT`** | 200 Hz, 500 Hz | *[YOUR observation — e.g. "200 Hz: slight bass roll-off; 500 Hz: noticeably thin, telephone-like."]* | *[YOUR observation — e.g. "Energy below cut frequency removed; energy above preserved."]* | *[YOUR explanation — e.g. "High-pass filter removes low-frequency energy; simulates band-limited transmission channel."]* |
| **`HIGH_CUT`** | 3000 Hz, 4000 Hz | *[YOUR observation — e.g. "3000 Hz: muffled, lacks sibilance; 4000 Hz: slightly warmer."]* | *[YOUR observation — e.g. "Energy above cut frequency removed; visible dark band in upper spectrogram."]* | *[YOUR explanation — e.g. "Low-pass filter removes high-frequency consonant energy; simulates telephone bandwidth (300–3400 Hz)."]* |
| **Gain** | 0.7, 2.0 | *[YOUR observation — e.g. "0.7: quieter, comfortable; 2.0: loud with audible distortion."]* | *[YOUR observation — e.g. "2.0: waveform clips at ±1.0; spectrogram shows harmonic smearing from clipping."]* | *[YOUR explanation — e.g. "Gain > 1.0 scales amplitude; values causing peaks > 1.0 are hard-clipped, introducing non-linear distortion."]* |

---

### Figures for Table 3

**Figure 12: Spectrogram – Speech Rate 120 (slow/clear)**
*(File: `outputs/tts/tts_speech_rate_120.png`)*

![Figure 12 – Speech Rate 120](outputs/tts/tts_speech_rate_120.png)

---

**Figure 13: Spectrogram – Speech Rate 220 (fast/rushed)**
*(File: `outputs/tts/tts_speech_rate_220.png`)*

![Figure 13 – Speech Rate 220](outputs/tts/tts_speech_rate_220.png)

---

**Figure 14: Spectrogram – Voice Index 0 (default voice)**
*(File: `outputs/tts/tts_voice_index_0.png`)*

![Figure 14 – Voice Index 0](outputs/tts/tts_voice_index_0.png)

---

**Figure 15: Spectrogram – Voice Index 1 (alternate voice)**
*(File: `outputs/tts/tts_voice_index_1.png`)*

![Figure 15 – Voice Index 1](outputs/tts/tts_voice_index_1.png)

---

**Figure 16: Spectrogram – Volume 0.5 (quiet)**
*(File: `outputs/tts/tts_volume_0.5.png`)*

![Figure 16 – Volume 0.5](outputs/tts/tts_volume_0.5.png)

---

**Figure 17: Spectrogram – Volume 1.0 (full)**
*(File: `outputs/tts/tts_volume_1.0.png`)*

![Figure 17 – Volume 1.0](outputs/tts/tts_volume_1.0.png)

---

**Figure 18: Spectrogram – Pre-emphasis 0.0 (none)**
*(File: `outputs/tts/tts_pre_emphasis_0.0.png`)*

![Figure 18 – Pre-emphasis 0.0](outputs/tts/tts_pre_emphasis_0.0.png)

---

**Figure 19: Spectrogram – Pre-emphasis 0.95 (boosted high-freq)**
*(File: `outputs/tts/tts_pre_emphasis_0.95.png`)*

![Figure 19 – Pre-emphasis 0.95](outputs/tts/tts_pre_emphasis_0.95.png)

---

**Figure 20: Spectrogram – Noise Level 0.0 (clean)**
*(File: `outputs/tts/tts_noise_level_0.0.png`)*

![Figure 20 – Noise Level 0.0](outputs/tts/tts_noise_level_0.0.png)

---

**Figure 21: Spectrogram – Noise Level 0.02 (moderate noise)**
*(File: `outputs/tts/tts_noise_level_0.02.png`)*

![Figure 21 – Noise Level 0.02](outputs/tts/tts_noise_level_0.02.png)

---

**Figure 22: Spectrogram – Pitch Shift −2 semitones (lower voice)**
*(File: `outputs/tts/tts_pitch_steps_-2.png`)*

![Figure 22 – Pitch Shift -2](outputs/tts/tts_pitch_steps_-2.png)

---

**Figure 23: Spectrogram – Pitch Shift +2 semitones (higher voice)**
*(File: `outputs/tts/tts_pitch_steps_2.png`)*

![Figure 23 – Pitch Shift +2](outputs/tts/tts_pitch_steps_2.png)

---

**Figure 24: Spectrogram – Time-stretch 0.8 (slower speech)**
*(File: `outputs/tts/tts_speed_factor_0.8.png`)*

![Figure 24 – Time-stretch 0.8](outputs/tts/tts_speed_factor_0.8.png)

---

**Figure 25: Spectrogram – Time-stretch 1.3 (faster speech)**
*(File: `outputs/tts/tts_speed_factor_1.3.png`)*

![Figure 25 – Time-stretch 1.3](outputs/tts/tts_speed_factor_1.3.png)

---

**Figure 26: Spectrogram – Low-cut 200 Hz (slight bass removal)**
*(File: `outputs/tts/tts_low_cut_200.png`)*

![Figure 26 – Low-cut 200 Hz](outputs/tts/tts_low_cut_200.png)

---

**Figure 27: Spectrogram – Low-cut 500 Hz (telephone-like)**
*(File: `outputs/tts/tts_low_cut_500.png`)*

![Figure 27 – Low-cut 500 Hz](outputs/tts/tts_low_cut_500.png)

---

**Figure 28: Spectrogram – High-cut 3000 Hz (narrowband / muffled)**
*(File: `outputs/tts/tts_high_cut_3000.png`)*

![Figure 28 – High-cut 3000 Hz](outputs/tts/tts_high_cut_3000.png)

---

**Figure 29: Spectrogram – High-cut 4000 Hz (slight treble roll-off)**
*(File: `outputs/tts/tts_high_cut_4000.png`)*

![Figure 29 – High-cut 4000 Hz](outputs/tts/tts_high_cut_4000.png)

---

**Figure 30: Spectrogram – Gain 0.7 (reduced amplitude)**
*(File: `outputs/tts/tts_gain_0.7.png`)*

![Figure 30 – Gain 0.7](outputs/tts/tts_gain_0.7.png)

---

**Figure 31: Spectrogram – Gain 2.0 (amplified / may clip)**
*(File: `outputs/tts/tts_gain_2.0.png`)*

![Figure 31 – Gain 2.0](outputs/tts/tts_gain_2.0.png)

---

## General Reflections

*Replace this section with your own ~250 word reflection on the assignment.*

*Example structure (replace with your own words):*

> This assignment provided hands-on experience with two fundamental areas of
> speech technology: Speech-to-Text (STT) and Text-to-Speech (TTS). Through
> systematic parameter modification I observed how individual signal-processing
> choices affect both the auditory and visual (spectrogram) representation of
> speech.
>
> In Part 1, the pre-emphasis experiment demonstrated clearly how boosting high
> frequencies (coefficient 0.97) enhances consonant energy in the spectrogram,
> making sibilants like /s/ and /t/ more prominent above 3 kHz. The noise
> experiment confirmed that even moderate additive Gaussian noise (level 0.01)
> visibly raises the noise floor across all frequency bands and degrades
> recognition accuracy. The speed and pitch experiments showed that the
> recogniser is fairly robust to mild changes but degrades with combinations of
> distortions, as the multi-parameter experiment illustrated.
>
> In Part 2, the TTS parameter sweep revealed that synthesis/voice parameters
> (rate, voice, volume) primarily affect the temporal and amplitude properties
> of the signal, while signal-processing parameters introduce more complex
> spectral changes. The bandwidth limitation experiments (LOW_CUT and HIGH_CUT)
> were particularly informative: simulating telephone bandwidth by restricting
> to ~300–3400 Hz removed much harmonic richness but kept speech intelligible,
> mirroring real-world telephony trade-offs.
>
> Overall, the assignment reinforced the importance of matching preprocessing
> choices to the recognition pipeline, and highlighted how TTS output quality
> degrades with extreme parameter values. Future work could explore adaptive
> noise cancellation or neural-network-based TTS to improve robustness.

---

## AI Usage Declaration

*[Insert the AI usage declaration from Appendix B of the CS425 AI Usage Policy
(available on Moodle) here.]*

---

## References

*[Add your references here in the citation style required by your module.]*

*Example:*

1. McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg,
   E., & Nieto, O. (2015). librosa: Audio and music signal analysis in Python.
   *Proceedings of the 14th Python in Science Conference*, 18–25.

2. Boll, S. F. (1979). Suppression of acoustic noise in speech using spectral
   subtraction. *IEEE Transactions on Acoustics, Speech, and Signal Processing*,
   27(2), 113–120.

3. Klatt, D. H. (1987). Review of text-to-speech conversion for English.
   *Journal of the Acoustical Society of America*, 82(3), 737–793.

---

## How to Regenerate All Results

Run the following commands from the repository root directory to regenerate
every plot and output file used in this report:

```bash
# Part 1 – STT A/B experiments (generates Figures 1–8, Table 1 data)
python run_stt_experiments.py Speaking_Female.wav

# Part 1 – Multi-parameter experiment (generates Figures 9–11, Table 2 data)
python run_multi_param_experiment.py Speaking_Female.wav

# Part 2 – TTS parameter sweep (generates Figures 12–31, Table 3 data)
python run_tts_experiments.py
```

If `Speaking_Female.wav` is not present, the STT scripts automatically use a
synthetic speech-like signal so all plots are still produced.

### Output locations

| Content | Directory |
|---|---|
| STT A/B spectrogram plots (Figs 1–8) | `outputs/stt/` |
| STT multi-param plots (Figs 9–11) | `outputs/stt/multi_param/` |
| STT transcription text files | `outputs/stt/` and `outputs/stt/multi_param/` |
| TTS spectrogram plots (Figs 12–31) | `outputs/tts/` |
| TTS processed audio WAV files | `outputs/tts/` |

### Individual templates

You can also run the original templates manually to experiment with a single
parameter combination:

```bash
# STT template – set PRE_EMPHASIS, NOISE_LEVEL, SPEED_FACTOR, PITCH_STEPS at
# the top of the file, then run:
python stt_template.py

# TTS template – set RATE, VOICE_INDEX, VOLUME, PRE_EMPHASIS, … at the top,
# then run:
python tts_template.py
```
