<div align="center">

# TRIBE v2

**A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

📄 [Paper](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) ▶️ [Demo](https://aidemos.atmeta.com/tribev2/) | 🤗 [Weights](https://huggingface.co/facebook/tribev2)

</div>

TRIBE v2 is a deep multimodal brain encoding model that predicts fMRI brain responses to naturalistic stimuli (video, audio, text). It combines state-of-the-art text, audio and video models into a unified Transformer architecture that maps multimodal representations onto the cortical surface.

## Quick start

Load a pretrained model from HuggingFace and predict brain responses to a video:

```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

df = model.get_events_dataframe(video_path="path/to/video.mp4")
preds, segments = model.predict(events=df)
print(preds.shape)  # (n_timesteps, n_vertices)
```

Predictions are for the "average" subject (see paper for details) and live on the **fsaverage5** cortical mesh (~20k vertices).
They are offset by 5 seconds in the past, in order to compensate for the hemodynamic lag.

You can also pass `text_path` or `audio_path` to `model.get_events_dataframe` — text is automatically converted to speech and transcribed to obtain word-level timings.

For a full walkthrough with brain visualizations, see the [Colab demo notebook](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb).

### Inference input formats

`TribeModel.get_events_dataframe(...)` accepts exactly one input source per call:

| Input type | Argument | Supported file formats | Example | Notes |
| --- | --- | --- | --- | --- |
| Video | `video_path` | `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm` | `model.get_events_dataframe(video_path="sample.mp4")` | Uses the video stream and extracts/transcribes audio when present to obtain word-level timings. |
| Audio | `audio_path` | `.wav`, `.mp3`, `.flac`, `.ogg` | `model.get_events_dataframe(audio_path="speech.wav")` | Transcribes the audio to obtain word-level timings. |
| Text | `text_path` | `.txt` | `model.get_events_dataframe(text_path="prompt.txt")` | Plain UTF-8 text. No timestamps are required; text is converted to speech and transcribed to create synthetic word-level timings. |

Example plain-text input (`prompt.txt`):

```text
A person walks through a busy city street while music plays in the background.
```

Then run inference:

```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

# Choose exactly one of these:
df = model.get_events_dataframe(video_path="sample.mp4")
# df = model.get_events_dataframe(audio_path="speech.wav")
# df = model.get_events_dataframe(text_path="prompt.txt")

preds, segments = model.predict(df)
```

If you already have real word-level timestamps aligned to a stimulus, construct an events DataFrame directly instead of using `text_path`, because `text_path` intentionally creates synthetic timings through text-to-speech.

## Installation

**Basic** (inference only):
```bash
pip install -e .
```

**With brain visualization**:
```bash
pip install -e ".[plotting]"
```

**With FreeSurfer surface-to-volume / cluster-table export**:

The vectorized analysis pipeline can write fsaverage surface maps directly.  To
run the optional FreeSurfer projection and clustering stage, install
FreeSurfer separately because it is a large system package, not a Python wheel.
For Apple Silicon Macs, the helper below downloads the official 8.2.0 arm64
package and installs it under `/Applications/freesurfer/8.2.0`:

```bash
bash tools/setup_freesurfer_macos.sh --download-only
bash tools/setup_freesurfer_macos.sh
```

FreeSurfer commands require a license file.  After receiving `license.txt` from
the FreeSurfer registration form, make it discoverable before running the
pipeline:

```bash
export FS_LICENSE="$HOME/license.txt"
export FREESURFER_HOME="/Applications/freesurfer/8.2.0"
export NO_FSFAST=1
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
```

Then a saved TRIBE prediction matrix can be converted into vectorized outputs,
surface maps, and FreeSurfer command outputs:

```bash
python tools/run_vectorized_analysis.py \
  --predictions derivatives/tribe/predictions.npy \
  --segments derivatives/tribe/segments.csv \
  --events derivatives/tribe/events.csv \
  --condition-column type \
  --contrast A-B \
  --write-rdm \
  --write-surface \
  --run-freesurfer \
  --output-root derivatives/vectorized/run_001
```

Without a license or local FreeSurfer install, use `--dry-run-freesurfer` to
validate the generated `mri_surf2vol` and `mri_surfcluster` commands.

**With training dependencies** (PyTorch Lightning, W&B, etc.):
```bash
pip install -e ".[training]"
```

## Training a model from scratch

### 1. Set environment variables

Configure data/output paths and Slurm partition (or edit `tribev2/grids/defaults.py` directly):

```bash
export DATAPATH="/path/to/studies"
export SAVEPATH="/path/to/output"
```


### 2. Run training

**Local test run:**
```bash
python -m tribev2.grids.test_run
```

**Grid search on Slurm:**
```bash
python -m tribev2.grids.run_cortical
python -m tribev2.grids.run_subcortical
```

## Project structure

```
tribev2/
├── main.py              # Experiment pipeline: Data, TribeExperiment
├── model.py             # FmriEncoder: Transformer-based multimodal→fMRI model
├── pl_module.py         # PyTorch Lightning training module
├── demo_utils.py        # TribeModel and helpers for inference from text/audio/video
├── analysis/            # Vectorized predictions, ROI/RDM, surface and FreeSurfer export
├── eventstransforms.py  # Custom event transforms (word extraction, chunking, …)
├── utils.py             # Multi-study loading, splitting, subject weighting
├── utils_fmri.py        # Surface projection (MNI / fsaverage) and ROI analysis
├── grids/
│   ├── defaults.py      # Full default experiment configuration
│   └── test_run.py      # Quick local test entry point
├── plotting/            # Brain visualization (PyVista & Nilearn backends)
└── studies/             # Dataset definitions (Algonauts2025, Lahner2024, …)
```

## Contributing to open science

If you use this software, please share your results with the broader research community using the following citation:

```bibtex
@article{dascoli2026foundation,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and Benchetrit, Yohann and Brooks, Teon and Begany, Katelyn and Raugel, Jos{\'e}phine and Banville, Hubert and King, Jean-R{\'e}mi},
  journal={arXiv preprint arXiv:2605.04326},
  year={2026}
}
```

## License

This project is licensed under CC-BY-NC-4.0. See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.
