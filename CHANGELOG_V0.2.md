# v0.2 Local Test Edition

## Improvements
- Modern dark Gradio UI with corrected text contrast and compact prototype/privacy notices.
- Rock fallback changed from aggressive thresholds/Hough lines to material clustering + morphology.
- Class-specific instance separation: watershed for grains/pores, connected components for fractures/cement.
- Significant reduction of synthetic-demo over-segmentation (carbonate demo: ~3400 objects in v0.1 -> ~140 in v0.2 during validation).
- Clean semantic overlay with boundaries and labels only on the largest objects.
- Browser object table limited to 100 largest objects; full CSV remains downloadable.
- Multi-variant OCR: original/upscaled/CLAHE+sharpen/Otsu/adaptive, with automatic selection by OCR confidence.
- Automatic Windows Tesseract discovery in LocalAppData and Program Files.
- Entity regexes bounded by subsequent field labels so one-line OCR does not swallow neighboring fields.
- Model version updated to `geo-ai-demo-0.2.0`.

## Scientific limitation
Rock Vision remains a deterministic demo baseline, not a trained/validated geological segmentation model. The research notebook contains the path to replace it with validated U-Net/CNN/Transformer weights.
