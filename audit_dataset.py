"""Validate unresized masks and all split manifests without modifying data."""
import argparse
import json
import numpy as np
from PIL import Image
from train import VOC

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True)
    p.add_argument('--dataset-config',required=True)
    p.add_argument('--splits',nargs='+',default=['train','val','test'])
    a=p.parse_args(); cfg=json.load(open(a.dataset_config))
    c=cfg['num_classes']; ignore=cfg.get('ignore_index',255)
    used=set()
    for split in a.splits:
        ds=VOC(a.data_root,split,512,c,ignore_index=ignore)
        overlap=used.intersection(ds.ids)
        if overlap: raise ValueError(f'Split overlap: {split}, {sorted(overlap)[:10]}')
        used.update(ds.ids); counts=np.zeros(c,dtype=np.int64); ignored=0
        for name in ds.ids:
            with Image.open(ds.images[name]) as image, Image.open(ds.root/'SegmentationClass'/f'{name}.png') as mask:
                raw=np.asarray(mask)
                if image.size!=mask.size or raw.ndim!=2: raise ValueError(f'Bad shape: {name}')
                valid=raw!=ignore; vals=raw[valid]
                if ((vals<0)|(vals>=c)).any(): raise ValueError(f'Invalid IDs: {name}: {np.unique(raw)}')
                counts+=np.bincount(vals.astype(np.int64),minlength=c)
                ignored+=int((~valid).sum())
        if not counts.sum(): raise ValueError(f'No valid pixels: {split}')
        print(json.dumps(dict(split=split,images=len(ds),class_names=cfg['class_names'],pixels=counts.tolist(),ignored=ignored),ensure_ascii=False))
        if (counts==0).any(): print(f'WARNING: {split} missing GT classes {np.flatnonzero(counts==0).tolist()}')
if __name__=='__main__': main()
