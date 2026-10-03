# v1.1 Lightweight Final

- Renamed Model Pipeline to **Multimodal Pipeline**.
- Added **EXECUTED IN THIS RUN** tables to Rock Vision, Legacy Document AI, and Multimodal Pipeline.
- Added dynamic execution graphs showing the actual branch used for each input.
- Shows actual algorithms only: OpenCV, K-means + morphology, watershed/connected components, fracture-candidate detector, scikit-image regionprops, PyMuPDF, Tesseract OCR, regex/unit normalization.
- SAM 2, U-Net, YOLO-Seg, ResNet/ViT are explicitly not claimed as executed in this CPU edition.
- Added per-run inference timing and CPU device labels.
- Fixed Gradio 6.x theme incompatibility by removing string font lists.
- Final local port: 7871.
- Version: `geo-ai-lightweight-1.1.0`.
