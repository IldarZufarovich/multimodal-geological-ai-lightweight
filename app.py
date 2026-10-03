import os
from __future__ import annotations
from pathlib import Path
import html, time, platform
import cv2
import pandas as pd
import gradio as gr
from config import SETTINGS
from inference import analyze_rock, analyze_document
from utils import ensure_path
from src.document_ai.ingestion import ingest
from src.document_ai.router import classify_input, detect_figure_regions, detect_text_regions, draw_routing_map
from src.document_ai.ocr import configure_tesseract, available_languages, russian_ocr_ready

EX = SETTINGS.examples_dir
DISCLAIMER = 'Portfolio prototype · lightweight CPU baseline · not validated for operational geological interpretation.'
CSS = r"""
:root{--bg:#f4f7fb;--card:#fff;--text:#111827;--muted:#526174;--accent:#0f9f91;--border:#d8e1eb;--code:#eef3f8;--ok:#087f5b}
html[data-geo-theme='dark']{--bg:#08111f;--card:#101c2d;--text:#f7fafc;--muted:#c1ccda;--accent:#3bd4c0;--border:#2b4058;--code:#0b1626;--ok:#6ee7b7}
html,body,.gradio-container{background:var(--bg)!important;color:var(--text)!important;font-family:Inter,ui-sans-serif,system-ui,-apple-system,'Segoe UI',Arial,sans-serif!important}
.gradio-container{max-width:1500px!important;margin:auto!important;padding:18px!important}.hero{padding:24px 28px;border-radius:20px;background:linear-gradient(135deg,#102d49,#176d70);margin-bottom:10px}.hero h1{color:white!important;margin:0!important;font-size:2rem!important}.hero p{color:#eafffb!important;margin:7px 0 0!important}.themebar{justify-content:flex-end}.themebar button{min-width:auto!important;border-radius:999px!important;background:var(--card)!important;color:var(--text)!important;border:1px solid var(--border)!important}.card,.exec-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:15px 18px;color:var(--text)!important}.badge{display:inline-block;background:#d8f7f1;color:#07695f;border-radius:999px;padding:4px 9px;font-size:.76rem;font-weight:850}.executed{display:inline-block;background:#d1fae5;color:#065f46;border:1px solid #6ee7b7;border-radius:999px;padding:4px 9px;font-size:.75rem;font-weight:900}.muted{color:var(--muted)!important}.legend{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}.lg{padding:5px 9px;border-radius:8px;color:#111827;font-weight:850}.g{background:#EEBE46}.p{background:#2887E6;color:#fff}.v{background:#1ECDCF}.f{background:#EB4141;color:#fff}.c{background:#9B5ACD;color:#fff}.flow{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--code);border:1px solid var(--border);border-radius:14px;padding:16px;white-space:pre-wrap;line-height:1.65;color:var(--text)!important}.gradio-container label,.gradio-container p,.gradio-container span,.gradio-container h1,.gradio-container h2,.gradio-container h3,.gradio-container h4{color:var(--text)!important}.gradio-container input,.gradio-container textarea,.gradio-container .block,.gradio-container .form{background:var(--card)!important;color:var(--text)!important;border-color:var(--border)!important}.gradio-container table{color:var(--text)!important;background:var(--card)!important}.gradio-container th{background:var(--code)!important;color:var(--text)!important}.gradio-container td{background:var(--card)!important;color:var(--text)!important}.gradio-container pre,.gradio-container code{background:var(--code)!important;color:var(--text)!important}.gradio-container .tab-nav button{color:var(--text)!important;font-weight:850!important}.gradio-container .tab-nav button.selected{color:var(--accent)!important;border-color:var(--accent)!important}footer{display:none!important}
"""
LEGEND="""<div class='legend'><span class='lg g'>G · Grain</span><span class='lg p'>P · Pore</span><span class='lg v'>V · Vug</span><span class='lg f'>F · Fracture candidate</span><span class='lg c'>C · Cement / clay</span></div>"""

def _safe_df(df,n=30): return df.head(n).copy() if isinstance(df,pd.DataFrame) else pd.DataFrame()
def _runtime_ms(t0): return round((time.perf_counter()-t0)*1000,1)
def _exec_df(rows): return pd.DataFrame(rows, columns=['stage','model / algorithm','version','device','status','output'])
def _graph(title, lines): return f"<div class='exec-card'><span class='executed'>✓ EXECUTED IN THIS RUN</span><h3>{html.escape(title)}</h3><div class='flow'>{html.escape(chr(10).join(lines))}</div></div>"

def rock_summary(s):
    return f"""<div class='card'><span class='badge'>ROCK VISION · {html.escape(str(s.get('image_type','image')))}</span><br><b>Visible 2D porosity:</b> {s.get('visible_2d_porosity_pct',0)}% · <b>Objects:</b> {s.get('objects',0)} · <b>Grains:</b> {s.get('grains',0)} · <b>Pores/vugs:</b> {s.get('pores_vugs',0)} · <b>Fracture candidates:</b> {s.get('fractures',0)}<br><b>Version:</b> {s.get('model_version')}<br><span class='muted'>{DISCLAIMER}</span></div>"""

def rock_execution(ms):
    rows=[
      ['Input QC & preprocessing','OpenCV','runtime','CPU','EXECUTED','normalized RGB / quality metrics'],
      ['Semantic segmentation','K-means material clustering + morphology','lightweight-baseline','CPU','EXECUTED','grain / pore / vug / cement-clay pixel classes'],
      ['Instance segmentation','Distance transform + watershed + connected components','lightweight-baseline','CPU','EXECUTED','individual object IDs'],
      ['Fracture candidates','thin-ridge / color / skeleton network detector','lightweight-baseline','CPU','EXECUTED','candidate fracture networks'],
      ['Quantification','scikit-image regionprops','runtime','CPU','EXECUTED','area / diameter / shape / orientation'],
      ['Total inference',f'{ms} ms','this run','CPU','EXECUTED','annotated image + object table']]
    graph=_graph('Rock Vision · execution graph',['Geological image','  ↓  OpenCV QC / normalization','Semantic segmentation · K-means + morphology','  ↓','Instance segmentation · watershed + connected components','  ↓','Fracture candidate detector · ridge/color/skeleton','  ↓','Object quantification · scikit-image regionprops'])
    return _exec_df(rows),graph

def run_rock(file):
    if not file: raise gr.Error('Upload a geological image or choose a demo sample.')
    path=ensure_path(file); t0=time.perf_counter()
    try:
        r=analyze_rock(path); ms=_runtime_ms(t0); edf,graph=rock_execution(ms)
        return r['original'],r['annotated'],rock_summary(r['summary']),_safe_df(r['objects'].sort_values('area_px',ascending=False),35),edf,graph,r['files']['annotated'],r['files']['objects_csv'],r['files']['json']
    except Exception as exc:
        try: rgb=ingest(path,1)[0][0]['rgb']
        except Exception: rgb=None
        msg=f"<div class='card'><span class='badge'>ANALYSIS NOT COMPLETED</span><br>{html.escape(str(exc))}<br><span class='muted'>Input preserved; no geological claim is made.</span></div>"
        return rgb,rgb,msg,pd.DataFrame(),pd.DataFrame(),_graph('Rock Vision · stopped',['Input received','  ↓','Analysis stopped safely · no geological claim']),None,None,None

def demo_rock(kind):
    m={'Carbonate thin section':'carbonate_thin_section.png','Fractured rock':'fractured_rock.png','Vuggy carbonate':'vuggy_carbonate.png'}; return str(EX/m[kind])
def preview_document(file):
    if not file:return None
    try:return ingest(ensure_path(file),1)[0][0]['rgb']
    except Exception:return None

def doc_summary(s):
    conf=float(s.get('mean_confidence',0)); langs=available_languages(); rus='READY' if russian_ocr_ready() else 'NOT INSTALLED'
    return f"""<div class='card'><span class='badge'>{'TEXT / OCR READY' if conf>=.60 else 'LOW OCR CONFIDENCE'}</span><br><b>Document type:</b> {html.escape(str(s.get('document_type','unknown')))} · <b>Text/OCR confidence:</b> {conf:.0%}<br><b>OCR languages available:</b> {html.escape(', '.join(langs) or 'none')} · <b>Russian OCR:</b> {rus} · <b>Version:</b> {SETTINGS.model_version}<br><span class='muted'>{DISCLAIMER}</span></div>"""

def doc_execution(result, ms):
    pages=result.get('pages',pd.DataFrame()); methods=[]
    if isinstance(pages,pd.DataFrame) and 'extraction_method' in pages:
        methods=[str(x) for x in pages['extraction_method'].dropna().unique()]
    searchable=any('text' in x.lower() and 'tesseract' not in x.lower() for x in methods)
    ocr=any('tesseract' in x.lower() for x in methods)
    rows=[['File/page ingestion','PyMuPDF + Pillow/OpenCV','runtime','CPU','EXECUTED','page image + metadata']]
    if searchable: rows.append(['Text extraction','PyMuPDF active text layer','runtime','CPU','EXECUTED','native PDF text'])
    if ocr:
        langs='+'.join([x for x in available_languages() if x in ('eng','rus')]) or 'eng'
        rows += [['Scan preprocessing','OpenCV multi-variant preprocessing','runtime','CPU','EXECUTED','OCR-ready page variants'],['OCR',f'Tesseract OCR ({langs})','local installation','CPU','EXECUTED','text + OCR confidence']]
    rows += [['Geological entity extraction','Regex dictionaries + unit normalization','lightweight-baseline','CPU','EXECUTED','structured entities + provenance'],['Total document inference',f'{ms} ms','this run','CPU','EXECUTED','text / entities / page traceability']]
    branch='PyMuPDF active text extraction' if searchable and not ocr else 'OpenCV preprocessing → Tesseract OCR'
    graph=_graph('Legacy Document AI · execution graph',['PDF / scan / photographed page','  ↓  file & page inspection',branch,'  ↓','Geological entity extraction · regex + normalization','  ↓','Traceability · source file → page → extracted value'])
    return _exec_df(rows),graph

def run_doc(file):
    if not file: raise gr.Error('Upload a PDF or scanned/photographed page.')
    t0=time.perf_counter(); r=analyze_document(ensure_path(file)); ms=_runtime_ms(t0); edf,graph=doc_execution(r,ms)
    return preview_document(file),doc_summary(r['summary']),r['text'],r['entities'],r['pages'],edf,graph,r['files']['entities_csv'],r['files']['text'],r['files']['json']

def demo_doc(kind):
    p=str(EX/('legacy_report_searchable.pdf' if kind=='Searchable geological PDF' else 'legacy_report_scan.png')); return p,preview_document(p)

def mm_exec(rows): return _exec_df(rows)

def multimodal_pipeline(file):
    if not file: raise gr.Error('Upload a PDF/page or geological image.')
    t0=time.perf_counter(); path=ensure_path(file); rgb=ingest(path,1)[0][0]['rgb']; top=classify_input(rgb)
    doc_text=''; entities=pd.DataFrame(); crop=None; interp=None; trace=pd.DataFrame(); routes=[]; exec_rows=[]
    exec_rows.append(['Input router','OCR evidence + page/image statistics','lightweight-router','CPU','EXECUTED',top['input_kind']])
    if top['is_document']:
        doc=analyze_document(path,max_pages=1); doc_text=doc['text'][:7000]; entities=doc['entities']; regions,crops=detect_figure_regions(rgb); texts=detect_text_regions(rgb); routed=draw_routing_map(rgb,regions,texts)
        routes.append({'region':'PAGE-TEXT','type':'TEXT','route':'OCR / active text → geological entities','confidence':round(top['ocr_confidence'],3)})
        exec_rows += [['Layout approximation','OpenCV contours + OCR evidence','lightweight-layout','CPU','EXECUTED','TEXT / FIGURE candidate regions']]
        ddf,_=doc_execution(doc,0); exec_rows += ddf.iloc[:-1].values.tolist()[1:]
        chosen=None
        for i,c in enumerate(crops):
            if not classify_input(c)['is_document']: chosen=(i,c); break
        if chosen:
            i,crop=chosen; tmp=SETTINGS.output_dir/'_routed_figure.png'; cv2.imwrite(str(tmp),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)); rr=analyze_rock(tmp); interp=rr['annotated']; trace=_safe_df(rr['objects'],40); trace['source_document']=Path(path).name; trace['page_number']=1; trace['image_id']=str(regions.iloc[i].region_id); routes.append({'region':str(regions.iloc[i].region_id),'type':'FIGURE','route':'semantic → instance → measurements','confidence':float(regions.iloc[i].confidence)}); rdf,_=rock_execution(0); exec_rows += rdf.iloc[:-1].values.tolist()[1:]
        note="<div class='card'><span class='badge'>DOCUMENT ROUTE</span> Green = TEXT → text extraction/OCR. Teal = FIGURE candidate → Rock Vision only after the content gate accepts it.</div>"
        graph_lines=['SOURCE DOCUMENT / PAGE','  ↓','Input router · document route','  ↓','Layout approximation · TEXT / FIGURE candidates','  ├── TEXT → active text or Tesseract OCR → geological entities','  └── FIGURE → Rock Vision','                 ↓ semantic segmentation','                 ↓ instance segmentation','                 ↓ fracture candidates / measurements','  ↓','Unified traceability · document → page → figure → object → measurement']
    else:
        routed=rgb.copy(); rr=analyze_rock(path); crop=rr['original']; interp=rr['annotated']; trace=_safe_df(rr['objects'],40); trace['source_document']=Path(path).name; trace['page_number']=1; routes.append({'region':'WHOLE IMAGE','type':'GEOLOGICAL IMAGE / COLLAGE','route':'Rock Vision directly · OCR skipped','confidence':top['routing_confidence']}); note="<div class='card'><span class='badge'>GEOLOGICAL IMAGE ROUTE</span> Input treated as geological visual/collage. OCR skipped as the primary route.</div>"; rdf,_=rock_execution(0); exec_rows += rdf.iloc[:-1].values.tolist()[1:]
        graph_lines=['SOURCE GEOLOGICAL IMAGE / COLLAGE','  ↓','Input router · geological-image route','  ↓','Rock Vision directly · OCR skipped','  ↓ semantic segmentation','  ↓ instance segmentation','  ↓ fracture candidates','  ↓ object quantification','  ↓','Traceability · source → image → object → measurement']
    ms=_runtime_ms(t0); exec_rows.append(['Total multimodal inference',f'{ms} ms','this run','CPU','EXECUTED','routed structured result'])
    graph=_graph('Multimodal Pipeline · execution graph',graph_lines)
    return rgb,routed,note,doc_text,entities,pd.DataFrame(routes),crop,interp,trace,mm_exec(exec_rows),graph

def demo_mm(): return str(EX/'multimodal_legacy_report.png')
def info():
    ok,_=configure_tesseract(); langs=available_languages() if ok else []
    return f"""<div class='card'><b>Project information</b><br>Version: <code>{SETTINGS.model_version}</code> · Device: CPU lightweight edition · Tesseract: {'READY' if ok else 'NOT FOUND'} · OCR languages: {html.escape(', '.join(langs) or 'none')}<br><span class='muted'>Only algorithms actually executed are labeled EXECUTED IN THIS RUN. SAM 2, U-Net, YOLO-Seg and transformer models belong to the planned GPU/research edition and are not claimed here.</span></div>"""

theme=gr.themes.Base(primary_hue='teal',secondary_hue='slate',neutral_hue='gray')
with gr.Blocks(title=SETTINGS.app_name) as demo:
    gr.HTML("""<div class='hero'><h1>MULTIMODAL GEOLOGICAL AI</h1><p>Legacy documents → structured geology · Geological images → traceable object-level interpretation</p></div>""")
    with gr.Row(elem_classes=['themebar']): light=gr.Button('☀ Light',size='sm'); dark=gr.Button('🌙 Dark',size='sm')
    light.click(None,None,None,js="() => {document.documentElement.setAttribute('data-geo-theme','light');}")
    dark.click(None,None,None,js="() => {document.documentElement.setAttribute('data-geo-theme','dark');}")
    with gr.Accordion('ⓘ Project info · privacy · OCR status',open=False): gr.HTML(info())
    with gr.Tabs():
        with gr.Tab('Rock Vision'):
            gr.Markdown('### Geological image → class-colored interpretation → instances → measurements')
            with gr.Row(): rock_in=gr.Image(type='filepath',sources=['upload','webcam'],label='Geological image / collage'); original=gr.Image(label='Original'); rock_out=gr.Image(label='Class-colored interpretation')
            gr.HTML(LEGEND)
            with gr.Row(): choice=gr.Dropdown(['Carbonate thin section','Fractured rock','Vuggy carbonate'],value='Fractured rock',label='Demo'); use=gr.Button('USE DEMO SAMPLE'); analyze=gr.Button('ANALYZE',variant='primary')
            stats=gr.HTML(); objects=gr.Dataframe(label='Largest objects / candidates',interactive=False)
            gr.Markdown('### Executed in this run'); rock_exec=gr.Dataframe(interactive=False); rock_graph=gr.HTML()
            with gr.Row(): ann=gr.File(label='Annotated image'); objcsv=gr.File(label='Objects CSV'); rockjson=gr.File(label='JSON result')
            use.click(demo_rock,choice,rock_in); analyze.click(run_rock,rock_in,[original,rock_out,stats,objects,rock_exec,rock_graph,ann,objcsv,rockjson])
        with gr.Tab('Legacy Document AI'):
            gr.Markdown('### Upload → preview → OCR / active text → geological entities → traceability')
            with gr.Row():
                with gr.Column(): doc_in=gr.File(file_types=['.pdf','.png','.jpg','.jpeg','.tif','.tiff'],label='PDF / scanned / photographed page'); dc=gr.Dropdown(['Searchable geological PDF','Scanned geological page'],value='Scanned geological page',label='Demo'); du=gr.Button('USE DEMO SAMPLE'); da=gr.Button('ANALYZE DOCUMENT',variant='primary')
                doc_preview=gr.Image(label='Preview · page 1',interactive=False)
            ds=gr.HTML(); text=gr.Textbox(label='OCR / active text',lines=10); entities=gr.Dataframe(label='Geological entities',interactive=False); pages=gr.Dataframe(label='Page traceability',interactive=False)
            gr.Markdown('### Executed in this run'); doc_exec=gr.Dataframe(interactive=False); doc_graph=gr.HTML()
            with gr.Row(): ecsv=gr.File(label='Entities CSV'); otxt=gr.File(label='OCR text'); djson=gr.File(label='JSON result')
            doc_in.change(preview_document,doc_in,doc_preview); du.click(demo_doc,dc,[doc_in,doc_preview]); da.click(run_doc,doc_in,[doc_preview,ds,text,entities,pages,doc_exec,doc_graph,ecsv,otxt,djson])
        with gr.Tab('Multimodal Pipeline'):
            gr.Markdown('### Input routing first: DOCUMENT → OCR/layout · GEOLOGICAL IMAGE/COLLAGE → Rock Vision')
            mm_in=gr.File(file_types=['.pdf','.png','.jpg','.jpeg','.tif','.tiff'],label='Upload document, page, or geological image')
            with gr.Row(): mm_demo=gr.Button('USE DEMO REPORT'); mm_run=gr.Button('RUN MULTIMODAL PIPELINE',variant='primary')
            mm_note=gr.HTML()
            with gr.Row(): mm_source=gr.Image(label='1 · Source'); mm_regions=gr.Image(label='2 · Routing map')
            mm_text=gr.Textbox(label='3A · Text branch → OCR / active text',lines=6); mm_entities=gr.Dataframe(label='3B · Geological entities'); mm_routes=gr.Dataframe(label='Routing decisions')
            with gr.Row(): mm_crop=gr.Image(label='4A · Geological image / routed figure'); mm_interp=gr.Image(label='4B · Class-colored object interpretation')
            gr.HTML(LEGEND); mm_trace=gr.Dataframe(label='5 · Provenance / object traceability')
            gr.Markdown('### Executed in this run'); mm_exec_table=gr.Dataframe(interactive=False,label='Models / algorithms actually executed'); mm_graph=gr.HTML()
            mm_demo.click(demo_mm,None,mm_in); mm_run.click(multimodal_pipeline,mm_in,[mm_source,mm_regions,mm_note,mm_text,mm_entities,mm_routes,mm_crop,mm_interp,mm_trace,mm_exec_table,mm_graph])

if __name__=='__main__':
    demo.queue(default_concurrency_limit=2).launch(theme=theme,css=CSS,server_name='0.0.0.0',server_port=int(os.environ.get('PORT', 7871)),inbrowser=False)
