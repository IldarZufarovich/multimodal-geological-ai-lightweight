# Local test — v0.3

From the project root:

```bash
python app.py
```

Open `http://127.0.0.1:7860`.

Recommended acceptance test:
1. Rock Vision → built-in carbonate sample → Analyze.
2. Rock Vision → upload a screenshot/document image. It should refuse forced rock segmentation.
3. Legacy Document AI → scanned demo → verify OCR text and entities.
4. Multimodal Pipeline → scanned demo report → verify OCR plus separate figure routing.
5. Multimodal Pipeline → upload a text-heavy screenshot with no geological image. It should show `NO ROCK SEGMENTATION` rather than labeling letters as grains.

For Russian OCR, install the Tesseract `rus` traineddata pack. The app will automatically select `eng+rus` when both are available.
