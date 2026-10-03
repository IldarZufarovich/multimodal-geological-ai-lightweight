# Validation v0.2

Tested in the build environment:
- Python compileall: PASS
- `python -m pytest -q`: 5/5 PASS
- Rock demo, carbonate thin section: PASS; 140 retained objects (91 grains, 35 pores, 14 cement/clay) in validation run
- Rock demo, fractured rock: PASS; 2 elongated fracture instances retained in validation run
- Rock demo, vuggy carbonate: PASS; 50 pore/vug instances retained in validation run
- Scanned legacy report OCR: PASS with Tesseract 5.5 in build environment
- OCR confidence on included degraded scan improved from ~0.36 in the user's v0.1 test to ~0.75 in v0.2 build validation
- Entity extraction recovered well, field, formation, depth interval, lithology, porosity, permeability, water saturation, sample ID, core number, pore and fracture descriptions from the included scan; OCR still reads `Arab-D` imperfectly as `ArabO`, which is intentionally not silently corrected.
- Gradio module import: PASS

Not claimed as tested here:
- Final Windows rendering on the user's exact Gradio 6.29.1 installation
- Smartphone camera permissions
- Public cloud deployment
- Performance on real field petrography or proprietary legacy scans
