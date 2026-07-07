# Multimodal Fake News Detection

A local, privacy-preserving system that assesses the credibility of news content across **text** and **images**. It combines fast keyword heuristics with local LLM reasoning (via Ollama) to flag sensationalism, unreliable attribution, and visual manipulation — all without sending data to external APIs.

## Overview

Misinformation rarely lives in text alone — a real headline can be paired with a manipulated photo, and a fabricated story can be dressed up with legitimate-looking imagery. This project tackles both angles:

- **Text Detector** — three complementary approaches (keyword-based, local LLM, hybrid) for scoring the credibility of an article.
- **Image Detector** — forensic analysis (metadata, compression artifacts, noise patterns, cloning, AI-generation heuristics) plus optional vision-LLM analysis to catch manipulated or synthetic images.

Both run entirely on local models through [Ollama](https://ollama.ai/), so no content ever leaves your machine.

## Features

### Text Analysis (`text_detector.py`)

| Approach | Description | Speed | Dependencies |
|---|---|---|---|
| **Keyword-based** | Pattern matching against sensational, emotional, absolutist, and vague-attribution language, plus clickbait regex patterns | Very fast | None |
| **Local LLM** | Sends article text to a local model (e.g. `llama3`) for semantic credibility analysis with reasoning | Slow | Ollama |
| **Hybrid** | Weighted combination — 35% keyword, 65% LLM — with automatic fallback to keyword-only if the LLM is unavailable | Medium | Ollama (optional) |

Each method returns a `fake_probability` (0–100), a verdict (**Likely Credible / Questionable / Likely Fake**), and a list of specific red flags.

### Image Analysis (`img_analysis.py`)

Six complementary forensic techniques, combined into a single weighted authenticity score:

1. **Metadata Analysis** — EXIF tampering, missing camera data, editing-software signatures
2. **Error Level Analysis (ELA)** — detects inconsistent JPEG compression across regions
3. **Noise Pattern Analysis** — flags blocks with anomalous noise variance (splicing indicator)
4. **Clone Detection** — block-hash matching to catch copy-paste manipulation
5. **AI Generation Heuristics** — color distribution, symmetry, and gradient smoothness checks
6. **Vision LLM Analysis** *(optional)* — semantic manipulation assessment via `llava` or `moondream`

Output includes an authenticity score, a verdict (**Likely Authentic / Questionable / Likely Manipulated**), and an annotated image with color-coded suspicious regions (red/orange/yellow by confidence).

Full scoring formulas, weight breakdowns, and data flow diagrams are documented in [`fake_news.md`](./fake_news.md).

## Architecture

```
Article
├─ Text Analysis (text_detector.py)
│  └─ Keyword / LLM-based credibility assessment
│
├─ Image Analysis (img_analysis.py)
│  └─ Multi-technique visual authenticity verification
│
└─ Combined Verdict
   └─ Flagged as suspicious if EITHER text OR images raise concerns
```

## Tech Stack

- **Python** — detection logic (keyword matching, forensic image analysis, LLM orchestration)
- **HTML/JS frontend** — served locally for interactive text and image analysis
- **Ollama** — local LLM runtime (`llama3` for text, `llava`/`moondream` for vision)
- **Pillow, NumPy, SciPy** — image processing and forensic computation

## Project Structure

```
├── code/            # Core detection logic and Flask apps
├── docs/             # Documentation
├── images/           # Sample / test images
├── text/              # Sample / test articles
├── uploads/           # Runtime upload directory
├── fake_news.md    # Full architecture documentation
└── requirements.txt   # Setup notes and dependencies
```

## Setup

### 1. Install Ollama and pull the required models

```bash
# Install Ollama: https://ollama.ai/

ollama pull llama3:latest   # for text detection
ollama pull llava:latest    # for image detection (or moondream/bakllava)
```

Verify the models are available:

```bash
ollama list
```

### 2. Install Python dependencies

```bash
pip install ollama Pillow numpy scipy
```

### 3. Run the servers

From the `code/` directory, start the backend apps:

```bash
python app.py          # text detector server
python app_image.py    # image detector server
```

Then serve the frontend:

```bash
cd code
python3 -m http.server
```

### 4. Open in browser

| Detector | URL |
|---|---|
| Text | `http://localhost:8000/` |
| Image | `http://localhost:8000/image_analyzer.html` |

Select text/image mode, and choose keyword, LLM, or hybrid analysis as needed.

## Performance Characteristics

**Text**

| Approach | Speed | Accuracy | Explainability |
|---|---|---|---|
| Keyword | Very Fast | Moderate | High |
| LLM | Slow | High | Medium |
| Hybrid | Medium | High | High |

**Image**

| Technique | Speed | Accuracy |
|---|---|---|
| Metadata | <10ms | Medium |
| ELA | 100–500ms | High |
| Noise | 200ms–1s | Medium |
| Clone | 1–5s | Medium |
| AI Heuristics | <50ms | Low |
| Vision LLM | 5–30s | High |

## Limitations

- Keyword-based detection is context-blind and can miss sophisticated misinformation.
- LLM outputs can be inconsistent or occasionally hallucinated; malformed responses default to a neutral score.
- AI-generation detection relies on heuristics, not a trained classifier — a dedicated model would improve reliability.
- ELA is most effective on JPEGs and less reliable on PNG/BMP sources.
- Vision LLM analysis adds significant latency (5–30s per image).

## Roadmap

- Replace AI-generation heuristics with a trained classifier
- Add source credibility / fact-check API cross-referencing
- Expand the combined verdict logic beyond an OR-based flagging rule
- Package the pipeline into a single unified web app

## License

No license file is currently included — please contact the repository owner before reuse or distribution.
