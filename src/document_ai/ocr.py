from __future__ import annotations
import os
import shutil
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_TESSDATA = Path(__file__).resolve().parents[2] / 'assets' / 'tessdata'


def configure_tesseract() -> tuple[bool, str | None]:
    """Resolve Tesseract at runtime, including common Windows/Jupyter installs."""
    if importlib.util.find_spec('pytesseract') is None:
        return False, None
    import pytesseract
    exe = shutil.which('tesseract')
    if not exe and os.name == 'nt':
        local = os.environ.get('LOCALAPPDATA', '')
        candidates = [
            Path(local) / 'Programs' / 'Tesseract-OCR' / 'tesseract.exe' if local else None,
            Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe'),
            Path(r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'),
        ]
        for p in candidates:
            if p is not None and p.exists():
                exe = str(p)
                break
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe
        exe_dir = str(Path(exe).parent)
        if exe_dir not in os.environ.get('PATH', ''):
            os.environ['PATH'] = exe_dir + os.pathsep + os.environ.get('PATH', '')
        return True, exe
    return False, None


def _tessdata_dir(exe: str | None = None) -> Path | None:
    # Prefer bundled language packs when present; otherwise use the Tesseract install folder.
    if PROJECT_TESSDATA.exists() and any(PROJECT_TESSDATA.glob('*.traineddata')):
        return PROJECT_TESSDATA
    if exe:
        p = Path(exe).parent / 'tessdata'
        if p.exists():
            return p
    return None


def available_languages() -> list[str]:
    ok, exe = configure_tesseract()
    if not ok:
        return []
    import pytesseract
    td = _tessdata_dir(exe)
    if td:
        langs = sorted(p.stem for p in td.glob('*.traineddata'))
        if langs:
            return langs
    try:
        return sorted(pytesseract.get_languages(config=''))
    except Exception:
        return []


def preferred_language() -> str:
    langs = set(available_languages())
    if {'eng', 'rus'} <= langs:
        return 'eng+rus'
    if 'rus' in langs:
        return 'rus'
    if 'eng' in langs:
        return 'eng'
    return ''


def russian_ocr_ready() -> bool:
    return 'rus' in set(available_languages())


def ocr_rgb(rgb: np.ndarray, psm: int = 6, lang: str | None = None):
    ok, exe = configure_tesseract()
    if not ok:
        return {'text': '', 'mean_confidence': 0.0, 'tokens': pd.DataFrame(),
                'warning': 'Tesseract executable was not found', 'language': '', 'tesseract_path': None}
    import pytesseract
    lang = preferred_language() if lang is None else lang
    cfg = f'--oem 3 --psm {int(psm)}'
    td = _tessdata_dir(exe)
    if td:
        cfg += f' --tessdata-dir "{td}"'
    kwargs = {'output_type': pytesseract.Output.DATAFRAME, 'config': cfg}
    if lang:
        kwargs['lang'] = lang
    try:
        d = pytesseract.image_to_data(rgb, **kwargs)
    except Exception as exc:
        return {'text': '', 'mean_confidence': 0.0, 'tokens': pd.DataFrame(),
                'warning': f'OCR failed: {exc}', 'language': lang, 'tesseract_path': exe}
    d = d.dropna(subset=['text'])
    d = d[d.text.astype(str).str.strip() != '']
    conf = pd.to_numeric(d.conf, errors='coerce')
    good = conf[conf >= 0]
    text = ' '.join(d.text.astype(str).tolist())
    warning = None
    if lang == 'eng' and len(text) > 40:
        warning = 'Russian OCR language pack is not installed. Cyrillic pages may be transliterated or misread.'
    return {'text': text, 'mean_confidence': float(good.mean()/100 if len(good) else 0),
            'tokens': d, 'warning': warning, 'language': lang, 'tesseract_path': exe}


def best_ocr(variants: dict[str, np.ndarray]):
    candidates = []
    for name, image in variants.items():
        best = None
        for psm in (6, 11):
            out = ocr_rgb(image, psm=psm)
            useful = sum(ch.isalnum() for ch in out['text'])
            score = out['mean_confidence'] + min(useful, 800) / 8000.0
            rec = {**out, 'variant': name, 'psm': psm, 'selection_score': score}
            if best is None or rec['selection_score'] > best['selection_score']:
                best = rec
        candidates.append(best)
    winner = max(candidates, key=lambda x: x['selection_score']) if candidates else ocr_rgb(np.zeros((10,10,3),np.uint8))
    winner['candidates'] = [
        {'variant': x['variant'], 'psm': x['psm'], 'mean_confidence': round(x['mean_confidence'],3),
         'text_chars': len(x['text']), 'language': x.get('language','')} for x in candidates
    ]
    return winner
