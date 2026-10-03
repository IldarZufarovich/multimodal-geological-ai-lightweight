from __future__ import annotations
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from config import SETTINGS
from src.common.io import load_rgb, save_json, utc_now
from .preprocessing import quality_metrics, normalize_rgb
from .segmentation import demo_semantic_segmentation, overlay
from .instances import semantic_to_instances
from .measurements import measure_objects, summarize

OBJECT_COLORS = {
    'grain': (238, 190, 70),       # gold
    'pore': (40, 135, 230),        # blue
    'vug': (30, 205, 205),         # cyan
    'fracture': (235, 65, 65),     # red
    'cement/clay': (155, 90, 205), # purple
}
PREFIX = {'grain':'G','pore':'P','vug':'V','fracture':'F','cement/clay':'C'}


def annotate_instances(rgb: np.ndarray, sem: np.ndarray, inst: np.ndarray, objects: pd.DataFrame, max_labels: int = 24) -> np.ndarray:
    """Class-colored object overlay with clear boundaries and compact labels."""
    out = rgb.copy()
    if objects.empty:
        return out
    # Transparent class fill per object, using object-level class (including vugs).
    tint = out.copy()
    for _, row in objects.iterrows():
        oid=int(row.object_id); cls=str(row.predicted_class); color=OBJECT_COLORS.get(cls,(255,255,255))
        m=inst==oid
        if np.any(m): tint[m]=color
    out=cv2.addWeighted(out,0.72,tint,0.28,0)
    for _, row in objects.iterrows():
        oid=int(row.object_id); cls=str(row.predicted_class); color=OBJECT_COLORS.get(cls,(255,255,255))
        mask=(inst==oid).astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out,contours,-1,color,2,cv2.LINE_AA)
    # Label largest objects only to avoid covering the geology.
    largest=objects.nlargest(min(max_labels,len(objects)),'area_px')
    for _,r in largest.iterrows():
        x,y=int(r.centroid_x),int(r.centroid_y); cls=str(r.predicted_class)
        label=f"{PREFIX.get(cls,'O')}{int(r.object_id)}"
        color=OBJECT_COLORS.get(cls,(255,255,255))
        cv2.putText(out,label,(max(2,x-10),max(14,y)),cv2.FONT_HERSHEY_SIMPLEX,.42,(10,18,28),3,cv2.LINE_AA)
        cv2.putText(out,label,(max(2,x-10),max(14,y)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)
    return out


def analyze_rock(path, output_dir=None):
    outdir=Path(output_dir or SETTINGS.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    rgb=load_rgb(path,SETTINGS.max_image_side); qc=quality_metrics(rgb); norm=normalize_rgb(rgb)
    sem,conf=demo_semantic_segmentation(norm); inst=semantic_to_instances(sem); image_id=Path(path).stem
    objects=measure_objects(rgb,sem,inst,str(path),None,image_id,SETTINGS.model_version)
    summary=summarize(sem,objects); summary.update(qc)
    blue_ratio=float(((rgb[...,2]>rgb[...,0]*1.15)&(rgb[...,2]>rgb[...,1]*1.05)).mean())
    summary['image_type']='thin-section / petrographic image' if blue_ratio>0.01 else 'core/rock image'
    summary['model_version']=SETTINGS.model_version; summary['timestamp']=utc_now(); summary['prototype_confidence']=conf['mean_confidence']; summary['segmentation_method']=conf['method']; summary['recognized_fraction']=round(conf.get('recognized_fraction',0.0),3); summary['fracture_candidate_method']=conf.get('fracture_candidates',{}).get('method'); summary['fracture_candidate_networks']=conf.get('fracture_candidates',{}).get('candidate_networks',0)
    annotated=annotate_instances(rgb,sem,inst,objects)
    ann=outdir/f'{image_id}_annotated.png'; mask=outdir/f'{image_id}_semantic_mask.png'; csv=outdir/f'{image_id}_objects.csv'; js=outdir/f'{image_id}_result.json'
    cv2.imwrite(str(ann),cv2.cvtColor(annotated,cv2.COLOR_RGB2BGR)); cv2.imwrite(str(mask),sem); objects.to_csv(csv,index=False); save_json({'summary':summary,'objects':objects.to_dict('records')},js)
    report=outdir/f'{image_id}_report.html'; report.write_text('<html><body><h1>Rock Vision Report</h1><pre>'+str(summary)+'</pre><p>Prototype output; not field validated.</p></body></html>',encoding='utf-8')
    return {'original':rgb,'annotated':annotated,'semantic_mask':sem,'instance_mask':inst,'objects':objects,'summary':summary,'files':{'annotated':str(ann),'mask':str(mask),'objects_csv':str(csv),'json':str(js),'report':str(report)}}
