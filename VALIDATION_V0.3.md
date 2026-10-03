# Validation — v0.3

## Tested in build environment
- Python syntax compilation: PASS.
- Existing automated tests: 5/5 PASS when repository root is on `PYTHONPATH`.
- Rock demo inference (`carbonate_thin_section.png`): PASS; 140 retained objects in this build test.
- Document-vs-rock router: carbonate demo routed as rock; legacy scan routed as document.
- OCR on built-in degraded legacy scan: PASS in build environment with Tesseract; ~0.75 mean OCR confidence in the observed run.
- New multimodal demo page: PASS.
  - OCR/entity extraction: 12 entities in build run.
  - Figure regions detected: 1.
  - Routing decisions: PAGE-TEXT → OCR, FIG-001 → Rock Vision.
  - Object traceability rows returned: 40 preview rows.
- `app.py` import: PASS.

## Requires user/local verification
- Windows Gradio rendering under the user's exact Gradio 6.29.1 environment.
- Russian OCR quality. The app selects `eng+rus` only when the Tesseract `rus` traineddata pack is installed.
- Smartphone camera behavior on a future HTTPS public deployment.
- Arbitrary real geological documents and real petrographic images are not validated field data.

## Scientific status
The page router, figure detector and live rock segmentation are deterministic CPU-safe baselines. They demonstrate architecture and traceability; they are not operational geological models.
