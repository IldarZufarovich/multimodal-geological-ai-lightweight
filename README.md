# Multimodal Geological AI

> **v0.3 — True Multimodal Document Understanding:** document/screenshots are routed to OCR first; only non-text geological figure candidates are sent to Rock Vision. Runtime Tesseract discovery and optional `eng+rus` OCR are included.


**From Legacy Geological Data to Digital Rock Intelligence**

A traceable multimodal AI portfolio prototype connecting legacy-document understanding with quantitative object-level rock-image analysis.

> **Live demo:** add your Hugging Face Space URL here after deployment.  
> **Research notebook:** `notebooks/Multimodal_Geological_AI_Pipeline.ipynb`

## Why this project

Subsurface archives combine scanned reports, searchable PDFs, tables, core photographs, thin sections, SEM imagery and CT-derived images. This project demonstrates the complete lifecycle from heterogeneous geological assets to structured, traceable outputs.

```text
Legacy document ─→ text-layer/OCR ─→ geological entities ─┐
                                                          ├─→ traceable geological intelligence
Rock image ─→ semantic mask ─→ instances ─→ measurements ┘
```

## Live application

### Rock Vision
Upload an image or use a phone camera. The CPU-safe demo performs QC, semantic segmentation, watershed-based instance separation, object measurements, an annotated overlay, confidence/provenance fields, and downloadable CSV/JSON outputs.

### Legacy Document AI
Upload a searchable PDF or scanned page. Searchable PDFs use the text layer; scans use preprocessing + Tesseract when available. Geological entities retain source file, page, extraction method and confidence.

### Multimodal traceability
The demo connects `source_document → page → image → object → class → confidence → model_version`.

## Semantic vs instance segmentation

- **Semantic:** all pore pixels belong to the `pore` class.
- **Instance:** Pore #1, Pore #2 and Pore #3 are distinct objects with individual measurements.

## Research vs deployment

The notebook contains synthetic-data generation, leakage-safe source-level splits, U-Net training, CNN classification, a simplified CNN–Transformer concept, metrics and explainability. The public app intentionally uses lightweight deterministic fallbacks unless validated weights are present. This keeps free-CPU deployment reliable and prevents the demo from overstating scientific validation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Tesseract improves OCR of raster scans. Ubuntu/Colab:

```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

The searchable-PDF demo works without Tesseract.

## Repository

```text
app.py                  Gradio application
config.py               central runtime settings
inference.py            public inference entry points
src/document_ai/        ingestion, OCR, entities, pipeline
src/rock_vision/        preprocessing, segmentation, instances, measurements
notebooks/              research/training notebook
assets/examples/        reproducible public demo inputs
models/                  optional validated model weights
outputs/                 generated outputs (git-ignored)
tests/                   smoke/unit tests
```

## Example outputs

The repository ships with reproducible demo samples: carbonate thin section, fractured rock, vuggy carbonate, searchable geological PDF, and scanned geological page.

## Metrics and scientific honesty

The research notebook reports accuracy-related metrics only when a valid source-level held-out split exists. The public fallback does **not** report field accuracy because it has not been validated on independent real geological data.

**Important limitations**

- Synthetic/demo performance does not establish field performance.
- 2D visible porosity is not automatically laboratory or 3D effective porosity.
- Color/texture heuristics are demonstration fallbacks, not a geological interpretation standard.
- Real deployment requires expert labels, domain-shift testing, calibration and geological QC.

## Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for Hugging Face Spaces, GitHub Pages buttons, mobile testing and QR generation.

## Roadmap

1. Fine-tune U-Net/instance model on expert-labelled thin sections.
2. Add calibrated object classifier with real pore/grain classes.
3. Add robust table/layout model for legacy scans.
4. Add optional multimodal RAG after OCR/document understanding.
5. Extend 2D workflow to 3D micro-CT volumes.

## Privacy

**Do not upload confidential or proprietary geological information to a public demo.** Enterprise deployment can use on-prem inference, private VPC, internal OCR, local/open-weight LLMs and internal vector databases.

## Author

**Ildar Z. Farkhutdinov** — Computational geoscience, reservoir engineering and applied AI.  
Add: [Portfolio](#) · [LinkedIn](#) · [Google Scholar](#)
