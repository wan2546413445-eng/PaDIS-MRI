"""Frozen E0/OPA single-patch diagnostic. Launch from PaDIS-MRI project root."""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter

HERE = Path(__file__).resolve().parent
DEFAULT_GT = "/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB/sample_26.pt"
DEFAULT_OUT = "/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/创新/R7_frozen_denoiser/sample_26"


def geometry(j, shape, unit, samples):
    pts = np.asarray(j["xy"], float).reshape(2, 2)
    if int(j["sample"]) != 26 or j["kind"] != "line":
        raise ValueError("Expected sample 26 line selection")
    length = np.linalg.norm(pts[1]-pts[0])
    ends = np.asarray(j["interval_ends"], float).reshape(6)
    if unit == "auto":
        unit = "fraction" if ends[-1] <= 1 else "px"
    if unit == "index":
        if samples is None or samples < 2:
            raise ValueError("Pass --source-samples (number of points on the original profile)")
        ends *= length/(samples-1)
    if unit == "fraction":
        ends *= length
    if not (np.all(np.diff(ends)>0) and ends[0]>=0 and ends[-1]<=length+1e-6):
        raise ValueError(f"Interval endpoints {ends} incompatible with length {length}; check original units.")
    top = np.clip(np.rint(pts.mean(0)).astype(int)-32, 0, np.array(shape[::-1])-64)
    intervals = ends.reshape(3,2)
    distances = np.array([np.linspace(a,b,65) for a,b in intervals])
    xy = (pts[0]-top)[:,None,None]+(pts[1]-pts[0])[:,None,None]*distances/length
    if np.any(xy<0) or np.any(xy>63):
        raise ValueError("Selected intervals extend outside the centered patch")
    return top, xy[::-1], intervals


def depth(image, coords):
    left, core, right = map_coordinates(image, coords, order=1).mean(axis=1)
    return (left+right)/2-core


def flat_roi(gt, coords):
    mean = uniform_filter(gt,7)
    var = np.maximum(uniform_filter(gt**2,7)-mean**2,0)
    y,x = np.indices(gt.shape)
    cy,cx = coords.mean(axis=(1,2))
    radius = np.hypot(x-cx,y-cy)
    eligible = (x>=3)&(x<61)&(y>=3)&(y<61)&(radius>=10)&(radius<=25)&(mean>0.5*np.median(map_coordinates(gt,coords,order=1)))
    if not eligible.any():
        raise ValueError("No eligible nearby foreground window")
    iy,ix = np.unravel_index(np.argmin(np.where(eligible,var,np.inf)),gt.shape)
    return np.s_[iy-3:iy+4,ix-3:ix+4]


def save_csv(path, rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection",type=Path,default=Path("s26_line.json"))
    p.add_argument("--gt",default=DEFAULT_GT)
    p.add_argument("--out",type=Path,default=Path(DEFAULT_OUT))
    p.add_argument("--device",default="cuda")
    p.add_argument("--interval-unit",choices=["auto","fraction","px","index"],default="auto")
    p.add_argument("--source-samples",type=int)
    args=p.parse_args()
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path.cwd()
    sys.path[:0]=[str(root),str(root/"train/padis-mri"),str(root/"eval")]
    cfg=[json.loads((HERE/f).read_text()) for f in ("summaryE.json","summaryOPA.json")]
    for key in ("steps","sigma_max","sigma_min","rho","seed"):
        if cfg[0][key]!=cfg[1][key]:
            raise ValueError(f"Unmatched {key}")
    raw=torch.load(args.gt,map_location="cpu",weights_only=False)["gt"].squeeze()
    if not raw.is_complex() or raw.ndim!=2:
        raise ValueError("Expected complex H x W GT")
    j=json.loads(args.selection.read_text())
    (x,y),coords,intervals=geometry(j,raw.shape,args.interval_unit,args.source_samples)
    gt=raw[y:y+64,x:x+64].to(torch.complex64)
    mag=gt.abs().numpy()
    ref=depth(mag,coords)
    if ref<=0:
        raise ValueError("Selected GT intervals do not describe a dark line; check interval units")
    roi=flat_roi(mag,coords)
    args.out.mkdir(parents=True,exist_ok=True)
    plt.imshow(mag,cmap="gray")
    for c in coords.transpose(1,0,2):
        plt.plot(c[1],c[0],lw=3)
    from matplotlib.patches import Rectangle
    plt.gca().add_patch(Rectangle((roi[1].start-.5,roi[0].start-.5),7,7,fill=False,color="cyan"))
    plt.title("GT: left / core / right; cyan = flat ROI")
    plt.savefig(args.out/"gt_regions.png",dpi=180)
    plt.close()
    print("GT depth:",ref,"intervals in pixels:",intervals.tolist(),flush=True)
    h,w=raw.shape
    gx=torch.linspace(-1,1,w+128)[64+x:128+x]
    gy=torch.linspace(-1,1,h+128)[64+y:128+y]
    pos=torch.stack([gx.expand(64,64),gy[:,None].expand(64,64)])[None].to(args.device)
    c=cfg[0]
    schedule=np.linspace(c["sigma_max"]**(1/c["rho"]),c["sigma_min"]**(1/c["rho"]),c["steps"])**c["rho"]
    rows=[]
    for model in cfg:
        print(model["model"],model["checkpoint"],flush=True)
        with open(model["checkpoint"],"rb") as f:
            net=pickle.load(f)["ema"].to(args.device).eval().requires_grad_(False)
        with torch.inference_mode():
            for level in (0,19,39):
                sigma=net.round_sigma(torch.tensor(schedule[level],device=args.device,dtype=torch.float64))
                for seed in range(c["seed"],c["seed"]+8):
                    gen=torch.Generator().manual_seed(seed)
                    noisy=gt+sigma.item()*torch.randn(gt.shape,dtype=gt.dtype,generator=gen)
                    channels=torch.view_as_real(noisy).permute(2,0,1)[None].contiguous().to(args.device)
                    out=net(channels,sigma,pos,None)[0].float().norm(dim=0).cpu().numpy()
                    d=depth(out,coords)
                    rows.append(dict(model=model["model"],level=level+1,sigma=sigma.item(),seed=seed,
                        gt_depth=ref,depth=d,retention=d/ref,abs_depth_error=abs(d-ref),
                        flat_rmse=np.mean((out[roi]-mag[roi])**2)**.5))
        del net
    save_csv(args.out/"per_seed.csv",rows)
    pairs=[]
    for level in (1,20,40):
        a,b=[[r for r in rows if r["model"]==m["model"] and r["level"]==level] for m in cfg]
        summary=dict(level=level,sigma=a[0]["sigma"])
        for k in ("retention","abs_depth_error","flat_rmse"):
            av=np.array([r[k] for r in a]); bv=np.array([r[k] for r in b])
            summary.update({f"E0_{k}":av.mean(),f"OPA_{k}":bv.mean(),f"delta_{k}":(bv-av).mean(),
                            f"delta_sd_{k}":(bv-av).std(ddof=1)})
        pairs.append(summary)
        print(f"level={level} sigma={a[0]['sigma']:.5g} depth ratio E0/OPA "
              f"{summary['E0_retention']:.3f}/{summary['OPA_retention']:.3f} "
              f"flat RMSE {summary['E0_flat_rmse']:.5f}/{summary['OPA_flat_rmse']:.5f}")
    save_csv(args.out/"paired_summary.csv",pairs)
    print(args.out)


if __name__=="__main__":
    main()
