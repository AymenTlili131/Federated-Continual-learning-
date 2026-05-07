#!/usr/bin/env python3
"""checkpoint_deep_analysis.py — Load checkpoint, produce 3 dataframes:
  DF1: weight vectors (i1,i2,pred,gt,FN1..FN5) + CNN losses per epoch
  DF2: eigenvalues per layer + Marchenko-Pastur
  DF3: persistent homology + multipers distances (ALL weight types)

Optimizations:
  - CNN batch_size=256, num_workers=4, pin_memory=True
  - Zoo CSV cached per overlap (loaded once)
  - DataLoader cached per task_classes combo
  - tqdm progress bars on all loops
  - CNN train/val losses saved per epoch in DF1

Usage:
  conda run -n FCL python3 scripts/checkpoint_deep_analysis.py \
      --loss MSE --overlap 0 --n-samples 100
"""
import sys, os, json, pickle, time, warnings
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import wasserstein_distance
from tqdm import tqdm
warnings.filterwarnings('ignore')

NB = Path(__file__).resolve().parent.parent
PROJ = NB.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(NB / 'core_modules'))

from Double_input_transformer import TransformerAE
from cnn_reconstruction import (reconstruct_cnn_from_weights, compute_eigenvalues,
    ClassSpecificImageFolder, train_cnn_epoch, validate_cnn)
try:
    from weight_normalization import WeightNormalizer
    HAS_WN = True
except ImportError:
    HAS_WN = False

LAYER_DELIMS = [208, 1414, 1514, 2254, 2464]
LAYER_NAMES = ['conv1','conv2','conv3','fc1','fc2']
MNIST = str(PROJ / 'data' / 'SplitMnist')
ZOO_CSV = PROJ / 'data' / 'Merged zoo.csv'

_zoo_cache = {}

import argparse
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp-dir', type=str, default=None)
    p.add_argument('--loss', type=str, default=None)
    p.add_argument('--overlap', type=int, default=0)
    p.add_argument('--model-size', type=str, default='tiny')
    p.add_argument('--n-samples', type=int, default=100)
    p.add_argument('--finetune-epochs', type=int, default=5)
    p.add_argument('--cnn-batch-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--output-dir', type=Path, default=NB/'paper_results'/'deep_analysis')
    p.add_argument('--topo-subsample', type=int, nargs='+', default=[60,120])
    p.add_argument('--topo-n-lines', type=int, nargs='+', default=[10,20])
    p.add_argument('--resume', action='store_true',
                   help='Skip experiments that already have output df1_weights.pkl')
    return p.parse_args()

def build_weight_index(df, wc):
    wm = df[wc].values.astype(np.float32)
    la, ea = df['label'].values, df['epoch'].values
    lei = {}
    for ac in ['leakyrelu','relu','tanh','sigmoid']:
        if ac not in df.columns: continue
        im = {}
        for ri in np.where(df[ac].values==1.0)[0]:
            im.setdefault(la[ri],{})[int(ea[ri])] = ri
        lei[ac] = im
    def lookup(l, a='leakyrelu', e=21):
        m = lei.get(a,{}).get(l,{})
        for t in range(e,10,-5):
            if t in m: return wm[m[t]]
        return None
    return lookup

def load_test_data(overlap, n_samples, seed=42):
    sd = PROJ/'data'/'Scenario'/f'overlapping_m{overlap}'
    tp = np.load(sd/'test_pairs.npy', allow_pickle=True)
    if overlap not in _zoo_cache:
        df = pd.read_csv(ZOO_CSV)
        wc = list(df.columns[17:-2])
        lk = build_weight_index(df, wc)
        _zoo_cache[overlap] = {'df': df, 'wc': wc, 'lookup': lk}
    lk = _zoo_cache[overlap]['lookup']
    x1l,x2l,yl,ml = [],[],[],[]
    for p in tp:
        t1,t2 = p; tc = sorted(set(t1)|set(t2))
        w1,w2,wy = lk(str(t1)),lk(str(t2)),lk(str(tc))
        if w1 is not None and w2 is not None and wy is not None:
            x1l.append(w1); x2l.append(w2); yl.append(wy)
            ml.append({'task1':t1,'task2':t2,'task_combined':tc})
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x1l), min(n_samples,len(x1l)), replace=False)
    return {'x1':np.array(x1l,dtype=np.float32)[idx],
            'x2':np.array(x2l,dtype=np.float32)[idx],
            'y':np.array(yl,dtype=np.float32)[idx],
            'meta':[ml[i] for i in idx]}

def load_normalizer(exp_dir):
    norm_path = Path(exp_dir) / 'weight_normalizer.pkl'
    if not norm_path.exists(): return None
    with open(norm_path, 'rb') as f: obj = pickle.load(f)
    if isinstance(obj, dict) and HAS_WN:
        wn = WeightNormalizer(method=obj.get('method','standard'))
        wn.scalers = obj['scalers']; wn.fitted = obj.get('fitted',True)
        return wn
    if hasattr(obj, 'transform'): return obj
    return None

def load_model(exp_dir, device):
    ckp = Path(exp_dir)/'checkpoints'/'best_model.pth'
    if not ckp.exists(): ckp = Path(exp_dir)/'checkpoints'/'final_model.pth'
    ck = torch.load(ckp, map_location='cpu', weights_only=False)
    cfg = ck['config']
    m = TransformerAE(max_seq_len=cfg.max_seq_len, N=cfg.N, heads=cfg.heads,
                      d_model=cfg.d_model, d_ff=cfg.d_ff, neck=cfg.neck, dropout=cfg.dropout)
    m.load_state_dict(ck['model_state_dict']); m.to(device).eval()
    return m, ck

def snapshot_weights(cnn_model):
    sd = cnn_model.state_dict()
    keys = ['module_list.0.weight','module_list.0.bias',
            'module_list.3.weight','module_list.3.bias',
            'module_list.6.weight','module_list.6.bias',
            'module_list.9.weight','module_list.9.bias',
            'module_list.11.weight','module_list.11.bias']
    return np.concatenate([sd[k].cpu().numpy().ravel() for k in keys]).astype(np.float32)

_loader_cache = {}

def get_loaders(task_classes, batch_size, num_workers):
    key = (tuple(sorted(task_classes)), batch_size)
    if key in _loader_cache:
        return _loader_cache[key]
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Grayscale(1)])
    ood = [c for c in range(10) if c not in task_classes]
    tid = ClassSpecificImageFolder(f"{MNIST}/test/", dropped_classes=[str(c) for c in ood], transform=tf)
    trd = ClassSpecificImageFolder(f"{MNIST}/train/", dropped_classes=[str(c) for c in ood], transform=tf)
    pin = torch.cuda.is_available()
    tidl = torch.utils.data.DataLoader(tid, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers, pin_memory=pin)
    trdl = torch.utils.data.DataLoader(trd, batch_size=batch_size, shuffle=True,
                                        num_workers=num_workers, pin_memory=pin,
                                        drop_last=True)
    _loader_cache[key] = (trdl, tidl)
    return trdl, tidl

def finetune_with_snapshots(pred_w, task_classes, n_epochs=5, device='cuda',
                             batch_size=256, num_workers=4):
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = reconstruct_cnn_from_weights(pred_w, 'leakyrelu').to(dev)
    trdl, tidl = get_loaders(task_classes, batch_size, num_workers)
    crit = nn.CrossEntropyLoss()
    _, acc0 = validate_cnn(model, tidl, crit, dev)
    fn_w, acc_h, loss_h = [], [acc0], []
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    sch = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=1e-3, max_lr=0.1,
        step_size_up=400, mode='triangular2', cycle_momentum=False)
    for ep in range(1, n_epochs+1):
        eloss, eacc = train_cnn_epoch(model, trdl, opt, crit, dev)
        _, vacc = validate_cnn(model, tidl, crit, dev)
        fn_w.append(snapshot_weights(model))
        acc_h.append(vacc)
        loss_h.append(eloss)
        sch.step()
    return {'fn_weights': fn_w, 'acc_history': acc_h, 'loss_history': loss_h,
            'acc_init': acc0, 'acc_final': acc_h[-1]}

WEIGHT_BOUNDS = [(0,200),(208,1408),(1414,1510),(1514,2234),(2254,2454)]
WEIGHT_SHAPES = [(8,25),(6,200),(4,24),(20,36),(10,20)]

def layer_eigenvalues(wv):
    eigs = {}
    for (s,e), nm, sh in zip(WEIGHT_BOUNDS, LAYER_NAMES, WEIGHT_SHAPES):
        W = wv[s:e].reshape(sh[0], -1)
        G = W @ W.T
        eigs[nm] = np.sort(np.linalg.eigvalsh(G))[::-1]
    return eigs

def mp_bounds(q, sigma=1.0):
    lp = sigma**2*(1+np.sqrt(1/q))**2
    lm = sigma**2*(1-np.sqrt(1/q))**2
    return lm, lp, sigma**2

def compute_ph(wv):
    try:
        import gudhi
        n = len(wv)
        if n > 500:
            idx = np.round(np.linspace(0,n-1,500)).astype(int)
            ws = wv[idx].astype(np.float64)
        else: ws = wv.astype(np.float64)
        st = gudhi.SimplexTree()
        for i in range(len(ws)): st.insert([i], filtration=float(ws[i]))
        for i in range(len(ws)-1): st.insert([i,i+1], filtration=float(max(ws[i],ws[i+1])))
        st.compute_persistence()
        ph0 = st.persistence_intervals_in_dimension(0)
        ph1 = st.persistence_intervals_in_dimension(1) if st.dimension()>=1 else []
        def pstats(pairs):
            if len(pairs)==0: return dict(n_features=0,total_persistence=0.0,persistence_entropy=0.0,max_lifetime=0.0,mean_lifetime=0.0)
            f = np.array(pairs); f = f[f[:,1]!=np.inf]
            if len(f)==0: return dict(n_features=0,total_persistence=0.0,persistence_entropy=0.0,max_lifetime=0.0,mean_lifetime=0.0)
            lt = f[:,1]-f[:,0]; pr = lt/lt.sum()
            return dict(n_features=len(f),total_persistence=float(lt.sum()),
                        persistence_entropy=float(-np.sum(pr*np.log(pr+1e-12))),
                        max_lifetime=float(lt.max()),mean_lifetime=float(lt.mean()))
        s0 = pstats(np.array(ph0)); s1 = pstats(np.array(ph1)) if len(ph1)>0 else pstats([])
        return {'betti_0':len(ph0),'betti_1':len(ph1),
                **{f'h0_{k}':v for k,v in s0.items()},**{f'h1_{k}':v for k,v in s1.items()}}
    except ImportError:
        s = np.sort(wv); gaps = np.sort(s[1:]-s[:-1])[::-1]
        return {'betti_0':int((gaps>0).sum()),'betti_1':0,
                'h0_n_features':int((gaps>0).sum()),'h0_total_persistence':float(gaps.sum()),
                'h0_persistence_entropy':0.0,'h0_max_lifetime':float(gaps[0]) if len(gaps)>0 else 0.0,
                'h0_mean_lifetime':float(gaps.mean()) if len(gaps)>0 else 0.0,
                'h1_n_features':0,'h1_total_persistence':0.0,'h1_persistence_entropy':0.0,
                'h1_max_lifetime':0.0,'h1_mean_lifetime':0.0}

def multipers_grid(wv, subs, nls):
    try:
        from multipers_analysis import compute_multipers_features as cmpf
        HAS = True
    except: HAS = False
    res = {}
    for s in subs:
        for nl in nls:
            k = f'sub{s}_nl{nl}'
            if HAS:
                try:
                    f = cmpf(wv, subsample_per_layer=s, n_lines=nl)
                    res[k] = np.concatenate([f[n] for n in LAYER_NAMES])
                except: res[k] = None
            else:
                parts = []; prev = 0
                for end,nm in zip(LAYER_DELIMS, LAYER_NAMES):
                    seg = wv[prev:end]; gs = np.sort(np.sort(seg)[1:]-np.sort(seg)[:-1])[::-1][:s]
                    if len(gs)<s: gs=np.pad(gs,(0,s-len(gs)))
                    parts.append(gs); prev=end
                res[k] = np.concatenate(parts).astype(np.float32)
    return res

def mpdist(f1, f2):
    if f1 is None or f2 is None: return float('nan')
    return float(wasserstein_distance(f1, f2))

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.exp_dir: exp_dir = Path(args.exp_dir)
    elif args.loss:
        ls = args.loss.replace('+','_').replace('*','x')
        exp_dir = NB/'experiments'/f'{args.model_size}_overlap{args.overlap}_{ls}'
    else: raise ValueError("Provide --exp-dir or --loss")
    assert exp_dir.exists(), f"Not found: {exp_dir}"
    loss_name = exp_dir.name.split('_',2)[-1]
    print(f"Experiment: {exp_dir.name}  |  Device: {device}  |  CNN bs={args.cnn_batch_size}")

    out_sub = args.output_dir / exp_dir.name
    if args.resume and (out_sub / 'df1_weights.pkl').exists():
        print(f"  SKIP (already done): {exp_dir.name}")
        return None, None, None

    t0 = time.time()
    model, ck = load_model(exp_dir, device)
    data = load_test_data(args.overlap, args.n_samples)
    normalizer = load_normalizer(exp_dir)

    N = len(data['y'])
    n_fn = args.finetune_epochs
    print(f"Samples: {N}  |  Finetune epochs: {n_fn}")

    # ── Inference (batch, GPU) ───────────────────────────────────────────────
    x1n = data['x1'].copy(); x2n = data['x2'].copy()
    if normalizer:
        x1n = normalizer.transform(x1n); x2n = normalizer.transform(x2n)
    else:
        mu = data['x1'].mean(0); sig = data['x1'].std(0)+1e-8
        x1n = (x1n-mu)/sig; x2n = (x2n-mu)/sig

    with torch.no_grad():
        out = model(torch.from_numpy(x1n).float().to(device),
                    torch.from_numpy(x2n).float().to(device))
        pred_norm = out[0].cpu().numpy() if isinstance(out, tuple) else out.cpu().numpy()

    if normalizer: pred_w = normalizer.inverse_transform(pred_norm)
    else: pred_w = pred_norm

    # ── DF1: Weight vectors + CNN finetune ────────────────────────────────────
    rows_df1 = []
    all_weight_types = ['i1','i2','pred','gt'] + [f'fn{e}' for e in range(1,n_fn+1)]
    n_wt = len(all_weight_types)  # 4 + n_fn

    # ── ETA estimator: time first CNN finetune, extrapolate ────────────────────
    tc0 = data['meta'][0].get('task_combined', list(range(10)))
    if isinstance(tc0, str): import ast; tc0 = ast.literal_eval(tc0)
    t_ft0 = time.time()
    fn0 = finetune_with_snapshots(pred_w[0], tc0, n_epochs=n_fn, device=str(device),
                                 batch_size=args.cnn_batch_size, num_workers=args.num_workers)
    sec_per_sample = time.time() - t_ft0
    # Eigenvalues: ~0.01s/sample*9wt, Topology: ~0.3s/sample*9wt*4scales (rough)
    sec_eig_est = N * n_wt * 0.01
    sec_topo_est = N * n_wt * 0.3 * len(args.topo_subsample) * len(args.topo_n_lines)
    eta_total = sec_per_sample * N + sec_eig_est + sec_topo_est
    print(f"\n  ⏱  1 CNN = {sec_per_sample:.1f}s  |  ETA finetune={sec_per_sample*N/60:.0f}min  eig={sec_eig_est/60:.0f}min  topo={sec_topo_est/60:.0f}min  TOTAL={eta_total/60:.0f}min")

    for i in tqdm(range(N), desc=f'{loss_name} finetune', ncols=100):
        row = {'sample': i, 'loss': loss_name, 'overlap': args.overlap}
        row['i1'] = data['x1'][i]
        row['i2'] = data['x2'][i]
        row['pred'] = pred_w[i]
        row['gt'] = data['y'][i]
        tc = data['meta'][i].get('task_combined', list(range(10)))
        if isinstance(tc, str): import ast; tc = ast.literal_eval(tc)
        # Reuse first finetune result instead of recomputing
        if i == 0:
            fn_result = fn0
        else:
            fn_result = finetune_with_snapshots(
                pred_w[i], tc, n_epochs=n_fn, device=str(device),
                batch_size=args.cnn_batch_size, num_workers=args.num_workers)
        for e in range(n_fn):
            row[f'fn{e+1}'] = fn_result['fn_weights'][e]
        row['cnn_acc_init'] = fn_result['acc_init']
        row['cnn_acc_final'] = fn_result['acc_final']
        row['cnn_acc_per_epoch'] = np.array(fn_result['acc_history'])
        row['cnn_loss_per_epoch'] = np.array(fn_result['loss_history'])
        rows_df1.append(row)

    df1 = pd.DataFrame(rows_df1)
    print(f"DF1: {df1.shape}  [{time.time()-t0:.1f}s]")

    # ── DF2: Eigenvalues (ALL weight types) ──────────────────────────────────
    rows_df2 = []
    for i in tqdm(range(N), desc=f'{loss_name} eigenvalues', ncols=100):
        for wt in all_weight_types:
            wv = rows_df1[i][wt]
            eigs = layer_eigenvalues(wv)
            row = {'sample':i, 'weight_type':wt, 'loss':loss_name, 'overlap':args.overlap}
            for ln in LAYER_NAMES:
                row[f'eig_{ln}'] = eigs[ln]
                shp = WEIGHT_SHAPES[LAYER_NAMES.index(ln)]
                q = shp[0] / shp[1]
                lm, lp, lpk = mp_bounds(q, sigma=float(np.std(wv)))
                row[f'mp_min_{ln}'] = lm
                row[f'mp_max_{ln}'] = lp
                row[f'mp_peak_{ln}'] = lpk
                ev = eigs[ln]
                row[f'frac_outside_mp_{ln}'] = float(np.mean((ev < lm) | (ev > lp)))
            rows_df2.append(row)
    df2 = pd.DataFrame(rows_df2)
    print(f"DF2: {df2.shape}  [{time.time()-t0:.1f}s]")

    # ── DF3: Topology (ALL weight types) ─────────────────────────────────────
    rows_df3 = []

    # Grid search on sample 0 to find best scale
    print("Topology grid search on sample 0...")
    grid_pred = multipers_grid(pred_w[0], args.topo_subsample, args.topo_n_lines)
    grid_gt = multipers_grid(data['y'][0], args.topo_subsample, args.topo_n_lines)
    best_key, best_dist = None, float('inf')
    dist_map = {}
    for k in grid_pred:
        d = mpdist(grid_pred[k], grid_gt[k])
        dist_map[k] = d
        if d < best_dist: best_dist = d; best_key = k
    sorted_keys = sorted(dist_map.keys(), key=lambda k: dist_map[k])
    breakpoint_key = sorted_keys[len(sorted_keys)//2] if len(sorted_keys)>2 else sorted_keys[0]
    meta_key = 'sub60_nl10'
    if meta_key not in grid_pred: meta_key = sorted_keys[0]
    selected_scales = {'optimal': best_key, 'breakpoint': breakpoint_key, 'meta': meta_key}
    print(f"Selected scales: {selected_scales}")

    for i in tqdm(range(N), desc=f'{loss_name} topology', ncols=100):
        for wt in all_weight_types:
            wv = rows_df1[i][wt]
            ph = compute_ph(wv)
            row = {'sample':i, 'weight_type':wt, 'loss':loss_name, 'overlap':args.overlap}
            row.update(ph)
            gt_wv = rows_df1[i]['gt']
            for scale_name, scale_key in selected_scales.items():
                sub_s = int(scale_key.split('_')[0].replace('sub',''))
                nl_s = int(scale_key.split('_')[1].replace('nl',''))
                feat_w = multipers_grid(wv, [sub_s], [nl_s])
                feat_gt = multipers_grid(gt_wv, [sub_s], [nl_s])
                fk = list(feat_w.keys())[0] if feat_w else None
                row[f'mpdist_{scale_name}'] = mpdist(feat_w.get(fk), feat_gt.get(fk))
            rows_df3.append(row)
    df3 = pd.DataFrame(rows_df3)
    print(f"DF3: {df3.shape}  [{time.time()-t0:.1f}s]")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = args.output_dir / exp_dir.name
    out.mkdir(parents=True, exist_ok=True)
    df1.to_pickle(out / 'df1_weights.pkl')
    df2.to_pickle(out / 'df2_eigenvalues.pkl')
    df3.to_pickle(out / 'df3_topology.pkl')
    with open(out / 'selected_scales.json','w') as f:
        json.dump(selected_scales, f, indent=2)
    elapsed = time.time() - t0
    print(f"\nSaved 3 dataframes to {out}  [{elapsed/60:.1f}min]")
    return df1, df2, df3

if __name__ == '__main__':
    main()
