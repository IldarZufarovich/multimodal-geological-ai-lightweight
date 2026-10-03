# v0.3 — True Multimodal Document Understanding

## Fixed from local acceptance testing
- Runtime Tesseract discovery now happens on every OCR call, including Windows LocalAppData installs.
- OCR automatically uses `eng+rus` when both language packs are installed; otherwise it falls back to available languages.
- Multimodal analysis no longer sends an entire document screenshot into rock segmentation.
- Added a CPU-safe content router: document-like input is processed with OCR first; only detected large image-like regions are routed to Rock Vision.
- If a document contains no credible geological figure candidate, rock segmentation is deliberately skipped instead of inventing grains from text.
- Rock Vision itself rejects obvious document/screenshots and directs the user to Document AI / Multimodal Pipeline.
- Added visible routing decisions, OCR text, region table, source page, detected figure crop and object provenance to the Multimodal tab.
- Updated dark-theme contrast and Gradio compatibility through 6.x.
- Replaced deprecated `fitz` import style with `pymupdf as fitz`.

## Scientific limitation
The content router and figure detector are deterministic CPU-safe fallbacks, not trained layout models. Real deployment should replace them with validated layout/document models and expert-labelled geological image classifiers.
