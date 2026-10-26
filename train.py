"""VOC multiclass segmentation: indexed masks, CE + Dice, dataset-level metrics."""
import argparse, csv, json, random, time
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from metrics import MulticlassSegmentationMetrics, export_metrics
from segmentation_metrics import write_run_results, append_history
from benchmark import benchmark_model
from local_weights import load_local_encoder

class VOC(Dataset):
    def __init__(self, root, split, size, num_classes, augment=False, ignore_index=255):
        self.root, self.size, self.augment = Path(root), size, augment
        self.num_classes, self.ignore_index = num_classes, ignore_index
        paths = [self.root/'ImageSets'/f'{split}.txt', self.root/'ImageSets'/'Main'/f'{split}.txt', self.root/'ImageSets'/'Segmentation'/f'{split}.txt']
        manifest = next((p for p in paths if p.exists()), None)
        if manifest is None: raise FileNotFoundError(f'Missing split: {paths}')
        self.ids = [x.strip() for x in manifest.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
        if not self.ids: raise ValueError(f'Empty split: {manifest}')
        if len(set(self.ids)) != len(self.ids): raise ValueError('Duplicate image IDs')
        self.images = {}
        for p in (self.root/'JPEGImages').iterdir():
            if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}:
                if p.stem in self.images: raise ValueError(f'Ambiguous image stem: {p.stem}')
                self.images[p.stem] = p
        for name in self.ids:
            if name not in self.images or not (self.root/'SegmentationClass'/f'{name}.png').exists():
                raise FileNotFoundError(f'Missing image/mask: {name}')
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        name = self.ids[i]
        image = Image.open(self.images[name]).convert('RGB')
        mask = Image.open(self.root/'SegmentationClass'/f'{name}.png')
        if image.size != mask.size: raise ValueError(f'Image/mask size mismatch: {name}')
        raw = np.asarray(mask)
        if raw.ndim != 2: raise ValueError(f'Mask must be single-channel or palette: {name}')
        allowed = set(range(self.num_classes)) | {self.ignore_index}
        if not set(np.unique(raw)).issubset(allowed): raise ValueError(f'Unexpected mask labels: {name}')
        image = image.resize((self.size,self.size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.size,self.size), Image.Resampling.NEAREST)
        if self.augment:
            if random.random()<.5: image,mask = ImageOps.mirror(image),ImageOps.mirror(mask)
            if random.random()<.5: image,mask = ImageOps.flip(image),ImageOps.flip(mask)
        raw = np.asarray(mask).copy()
        target = raw.astype(np.int64)
        x = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
        x = (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        return x,torch.from_numpy(target)

def loss_fn(logits, target, ignore_index=255):
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0
    ce = torch.nn.functional.cross_entropy(logits, target, ignore_index=ignore_index)
    safe = target.masked_fill(~valid, 0)
    y = torch.nn.functional.one_hot(safe, logits.shape[1]).permute(0,3,1,2).float()
    v = valid[:,None].float()
    prob = logits.float().softmax(1) * v
    y = y * v
    dims = (0,2,3)
    intersection = (prob*y).sum(dims)
    denominator = prob.sum(dims)+y.sum(dims)
    present = y.sum(dims)>0
    dice = (2*intersection+1e-6)/(denominator+1e-6)
    return ce + (1-dice[present]).mean()

@torch.no_grad()
def evaluate(model, loader, device, num_classes, class_names, ignore_index=255, desc='Validation', background_index=0):
    model.eval()
    meter = MulticlassSegmentationMetrics(num_classes, class_names, ignore_index, background_index)
    for x,y in tqdm(loader,desc=desc,dynamic_ncols=True,mininterval=1.0):
        prediction = model(x.to(device)).argmax(1).cpu()
        meter.update(prediction, y)
    result = meter.compute()
    if not result['valid_pixels']: raise ValueError(f'No valid pixels in {desc}')
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True); p.add_argument('--output',default='outputs/unet')
    p.add_argument('--arch',default='Unet'); p.add_argument('--encoder',default='resnet34')
    p.add_argument('--weights',choices=['none','imagenet','local'],default='local',help='none: random; imagenet: online; local: offline ImageNet encoder')
    p.add_argument('--pretrained-dir',default='pretrained',help='Folder containing <encoder>.safetensors for --weights local')
    p.add_argument('--dataset-config',help='JSON with num_classes, class_names and optional ignore_index/background_index')
    p.add_argument('--num-classes',type=int)
    p.add_argument('--class-names',nargs='+')
    p.add_argument('--ignore-index',type=int,default=None)
    p.add_argument('--background-index',type=int,default=None)
    p.add_argument('--train-split',default='train'); p.add_argument('--val-split',default='val'); p.add_argument('--test-split',default='test')
    p.add_argument('--size',type=int,default=512); p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=32); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--benchmark-warmup',type=int,default=50)
    p.add_argument('--benchmark-iterations',type=int,default=200)
    p.add_argument('--resume',action='store_true'); p.add_argument('--eval-only',action='store_true')
    a=p.parse_args()
    cfg=json.loads(Path(a.dataset_config).read_text()) if a.dataset_config else {}
    a.num_classes=a.num_classes or cfg.get('num_classes')
    if a.num_classes is None or a.num_classes<2: p.error('Set --dataset-config or --num-classes >= 2')
    a.class_names=a.class_names or cfg.get('class_names') or [f'class_{i}' for i in range(a.num_classes)]
    if len(a.class_names)!=a.num_classes or len(set(a.class_names))!=a.num_classes: p.error('class-names must be unique and match num-classes')
    a.ignore_index=a.ignore_index if a.ignore_index is not None else cfg.get('ignore_index',255)
    a.background_index=a.background_index if a.background_index is not None else cfg.get('background_index',0)
    if a.ignore_index in range(a.num_classes): p.error('ignore-index cannot be a valid class ID')
    if not 0<=a.background_index<a.num_classes: p.error('Invalid background-index')
    if a.epochs<1 or a.size<32 or a.batch_size<1: p.error('Invalid epochs/size/batch-size')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def dataset(split,aug=False): return VOC(a.data_root,split,a.size,a.num_classes,aug,a.ignore_index)
    train,val=dataset(a.train_split,True),dataset(a.val_split)
    if set(train.ids)&set(val.ids): raise ValueError('Train/val overlap')
    test=None
    if a.test_split.lower()!='none':
        test=dataset(a.test_split)
        if set(test.ids)&(set(train.ids)|set(val.ids)): raise ValueError('Test overlaps train/val')
    def loader(ds,shuffle=False): return DataLoader(ds,batch_size=a.batch_size,shuffle=shuffle,num_workers=a.workers,pin_memory=device.type=='cuda')
    print(f'Device: {device}; train={len(train)}, val={len(val)}, test={len(test) if test is not None else 0}; model={a.arch}/{a.encoder}',flush=True)
    print('Initializing model...',flush=True)
    restored=a.resume or a.eval_only
    model=smp.create_model(a.arch,encoder_name=a.encoder,encoder_weights=None if restored or a.weights in {'none','local'} else a.weights,in_channels=3,classes=a.num_classes).to(device)
    if a.weights=='local' and not restored:
        load_local_encoder(model,a.encoder,a.pretrained_dir)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start,best=0,-1
    if restored:
        ck=torch.load(out/('best.pth' if a.eval_only else 'last.pth'),map_location=device,weights_only=False)
        for key in ['arch','encoder','size','num_classes','class_names','ignore_index','background_index','seed','epochs','lr','batch_size','data_root','train_split','val_split','test_split','weights']:
            if ck['config'][key]!=vars(a)[key]: raise ValueError(f'Checkpoint config mismatch: {key}')
        model.load_state_dict(ck['model'])
        if a.resume and not a.eval_only:
            opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
            start,best=ck['epoch']+1,ck['best']
            random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng'].cpu())
            if device.type=='cuda' and ck['cuda_rng'] is not None: torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    elif (out/'last.pth').exists(): raise FileExistsError('Output exists: use --resume or a different --output')
    (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    trainloader,valloader=loader(train,True),loader(val)
    for epoch in range(start,a.epochs) if not a.eval_only else []:
        model.train(); total=0; begin=time.time()
        progress=tqdm(trainloader,desc=f'Epoch {epoch+1}/{a.epochs} Train',dynamic_ncols=True,mininterval=1.0)
        seen=0
        for x,y in progress:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                loss=loss_fn(model(x),y,a.ignore_index)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=loss.item()*len(x)
            seen+=len(x); progress.set_postfix(loss=f'{total/seen:.4f}',refresh=False)
        result=evaluate(model,valloader,device,a.num_classes,a.class_names,a.ignore_index,desc=f'Epoch {epoch+1}/{a.epochs} Val',background_index=a.background_index); scheduler.step()
        improved=result['miou']>best
        if improved: best=result['miou']
        ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch,best=best,config=vars(a),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        torch.save(ck,out/'last.pth')
        if improved: torch.save(ck,out/'best.pth')
        row=dict(metric_protocol=result["metric_protocol"],epoch=epoch+1,loss=total/len(train),seconds=time.time()-begin,**{k:v for k,v in result.items() if isinstance(v,(int,float)) or v is None})
        append_history(row,out/'history.csv')
        print(f"Epoch {epoch+1}/{a.epochs} | loss={row['loss']:.4f} | val mIoU={result['miou']:.4f} | Dice={result['dice']:.4f} | best mIoU={best:.4f} | time={row['seconds']:.1f}s",flush=True)
    ck=torch.load(out/'best.pth',map_location=device,weights_only=False); model.load_state_dict(ck['model'])
    result=dict(arch=a.arch,encoder=a.encoder,seed=a.seed,best_epoch=ck['epoch']+1,parameters=sum(p.numel() for p in model.parameters()),val=evaluate(model,valloader,device,a.num_classes,a.class_names,a.ignore_index,background_index=a.background_index))
    if test is not None: result['test']=evaluate(model,loader(test),device,a.num_classes,a.class_names,a.ignore_index,desc='Test',background_index=a.background_index)
    write_run_results(result,out)
    print('Measuring parameters, FLOPs and inference speed...',flush=True)
    result['benchmark']=benchmark_model(model,a.size,a.benchmark_warmup,a.benchmark_iterations)
    (out/'benchmark.json').write_text(json.dumps(result['benchmark'],indent=2))
    for split in ['val','test']:
        if split in result: export_metrics(result[split], out, split, a.background_index)
    write_run_results(result,out); print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__': main()

