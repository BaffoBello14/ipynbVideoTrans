# ipynbVideoTrans

> A **Jupyter Notebook** adaptation of [pyVideoTrans](https://github.com/jianchang512/pyvideotrans) – translate videos from one language to another with speech recognition, subtitle translation, AI dubbing and subtitle embedding, all from a notebook without any GUI.

---

## Features

- **ASR** – Speech-to-text via Faster-Whisper (local), OpenAI Whisper, Gemini, and many other providers
- **Translation** – Google, Microsoft, DeepSeek, OpenAI ChatGPT, Gemini, DeepL, and more
- **TTS** – Edge-TTS (free), OpenAI TTS, Google TTS, Gemini TTS, and many others
- **Video assembly** – merge dubbed audio + subtitles (hard or soft) back into the video via FFmpeg
- **Standalone tasks** – run only ASR, only translation, or only TTS independently

---

## Quick start

### Prerequisites

| Tool | Notes |
|------|-------|
| Python 3.10.x | required (3.10 exactly, < 3.11) |
| FFmpeg | `sudo apt install ffmpeg` / `brew install ffmpeg` |
| Jupyter | `pip install jupyter` |

### Install

```bash
git clone <this-repo>
cd ipynbVideoTrans
pip install -r requirements.txt
```

### Run

```bash
jupyter notebook pyvideotrans_notebook.ipynb
```

---

## Notebook structure

| Cell | Purpose |
|------|---------|
| 1 | Install dependencies |
| 2 | **Configuration** – set input/output paths, languages, ASR/TTS/translation providers |
| 3 | Initialise the pyVideoTrans backend |
| 4 | **ASR** – transcribe video to source-language SRT |
| 5 | **Translate** – source SRT → target SRT |
| 6 | **TTS** – target SRT → dubbed audio |
| 7 | **Full VTV pipeline** – run all steps in one shot → final MP4 |
| 8 | Standalone SRT translation |
| 9 | Standalone TTS from SRT |
| 10 | Utilities (list voices, providers, inspect SRT, check GPU) |

---

## Configuration (Cell 2)

```python
INPUT_VIDEO    = "/path/to/your/video.mp4"
OUTPUT_DIR     = "/path/to/output"

SOURCE_LANG    = "zh"          # spoken language of the video
TARGET_LANG    = "en"          # translation target

RECOGN_TYPE    = 0             # 0 = Faster-Whisper (local)
WHISPER_MODEL  = "large-v3-turbo"
USE_CUDA       = False

TRANSLATE_TYPE = 0             # 0 = Google Translate (no key needed)
TTS_TYPE       = 0             # 0 = Edge-TTS (free)
VOICE_ROLE     = "en-US-JennyNeural"

SUBTITLE_TYPE  = 1             # 1 = hard (burned-in) subtitles
```

### API keys (optional, for cloud providers)

Set environment variables before starting Jupyter, or paste directly into Cell 2:

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export DEEPL_API_KEY="..."
```

---

## Project structure

```
ipynbVideoTrans/
├── pyvideotrans_notebook.ipynb   ← main notebook
├── requirements.txt
├── README.md
└── videotrans/                   ← backend library (no GUI)
    ├── configure/                 config, settings, logging
    ├── recognition/               ASR providers
    ├── translator/                translation providers
    ├── tts/                       TTS providers
    ├── task/                      pipeline tasks
    ├── util/                      helpers (ffmpeg, srt, tools)
    ├── codes/                     language code mappings
    ├── language/                  UI string translations
    ├── prompts/                   LLM prompt templates
    └── voicejson/                 voice lists per provider
```

> **GUI components removed:** `winform/`, `mainwin/`, `ui/`, `styles/`, `component/` and all PySide6/Qt dependencies have been stripped. The backend is 100% headless.

---

## Credits

Built on top of [pyVideoTrans](https://github.com/jianchang512/pyvideotrans) by jianchang512, licensed under GPL-3.0.
