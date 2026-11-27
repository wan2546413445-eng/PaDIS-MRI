"""Measure an existing GT line on R7 reconstructions. Requires r7_wavelet_detail.py beside this file."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

import r7_wavelet_detail as w


def pair(value):
    if isinstance(value, dict):
        for a, b in (("x", "y"), ("start", "end"), ("lo", "hi"), ("min", "max")):
            if a in value and b in value:
                value = [value[a], value[b]]
                break
    a = np.asarray(value, dtype=float)
    if a.shape != (2,) or not np.all(np.isfinite(a)):
        raise ValueError(f"Expected a pair, got {value}")
    return a


def selection(j):
    # Coordinates are always image x,y, with the origin at the top left.
    for obj in (j, j.get("line", {}), j.get("selection", {})):
        if not isinstance(obj, dict):
            continue
        for key in ("points", "endpoints", "line_points", "line_xy", "endpoints_xy", "points_xy", "line"):
            v = obj.get(key)
            if isinstance(v, list) and len(v) == 2:
                return np.array([pair(p) for p in v])
        for a, b in (("p0","p1"), ("p1","p2"), ("p0_xy","p1_xy"), ("start","end"), ("start_xy","end_xy"), ("a","b"), ("A","B")):
            if a in obj and b in obj:
                return np.array([pair(obj[a]), pair(obj[b])])
        for keys in (("x0","y0","x1","y1"), ("x1","y1","x2","y2")):
            if all(k in obj for k in keys):
                return np.array([obj[k] for k in keys], dtype=float).reshape(2,2)
    raise ValueError(f"Unknown endpoint fields in JSON: {list(j)}. See --xy.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json", type=Path)
    p.add_argument("--xy", nargs=4, type=float, metavar=("X0","Y0","X1","Y1"),
                   help="Original GT endpoint coordinates in the full uncropped image")
    p.add_argument("--out", type=Path, default=w.ROOT/"R7_wavelet_E0_vs_OPA1p25_10k/sample_26/same_line")
    args = p.parse_args()
    j = json.loads(args.json.read_text(encoding="utf-8-sig"))
    points = np.reshape(args.xy,(2,2)) if args.xy else selection(j)
    paths = [w.VAL, w.recon_path("E0_10k_pure_DPS_smax1","E0_10k"),
             w.recon_path("OPA_10k_pure_DPS_smax1","OPA1p25_10k")]
    imgs = [w.magnitude(f) for f in paths]
    h, width = imgs[0].shape
    if any(im.shape != (h,width) for im in imgs) or np.any(points<0) or np.any(points>[width-1,h-1]):
        raise ValueError("Image shapes or endpoint coordinates do not match full GT.")
    scale = np.percentile(imgs[0],99)
    length = np.linalg.norm(points[1]-points[0])
    if length < 2 or scale <= 0:
        raise ValueError("Invalid line length or GT scale")
    d = np.linspace(0,length,int(np.ceil(length*16))+1)
    xy = points[0,:,None]+(points[1]-points[0])[:,None]*d/length
    curves = [map_coordinates(im/scale,xy[::-1],order=1) for im in imgs]
    # This diagnostic uses the whole selected isolated dark trough and its two endpoints.
    # Depth relative to the straight baseline through the endpoints handles a sloping background.
    names = ["GT","E0","OPA1.25"]
    fig,(ax,pr) = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    ax.imshow(imgs[0]/scale,cmap="gray",vmin=0,vmax=1)
    ax.plot(*points.T,"r.-",lw=1)
    margin = max(10,length)
    ax.set_xlim(max(0,points[:,0].min()-margin),min(width-1,points[:,0].max()+margin))
    ax.set_ylim(min(h-1,points[:,1].max()+margin),max(0,points[:,1].min()-margin))
    ax.set_title("GT / original line"); ax.set_aspect("equal")
    rows=[]
    for name,v in zip(names,curves):
        trough = int(np.argmin(v))
        baseline = np.interp(d[trough],[0,length],[v[0],v[-1]])
        depth = baseline-v[trough]
        if not rows:
            gt_pos,gt_depth = d[trough],depth
        rows.append([name,d[trough],d[trough]-gt_pos,depth,depth/gt_depth if gt_depth>0 else np.nan,
                     trough in (0,len(d)-1)])
        line, = pr.plot(d,v,label=name)
        pr.plot(d[trough],v[trough],"o",color=line.get_color(),ms=4)
    pr.set(xlabel="Distance from first endpoint (px)",ylabel="Magnitude / GT p99")
    pr.legend()
    args.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out/"r7_line.png",dpi=200)
    np.savetxt(args.out/"profiles.csv",np.column_stack([d,*curves]),delimiter=",",header="distance_px,GT,E0,OPA1p25",comments="")
    with (args.out/"metrics.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["model","minimum_px","shift_vs_GT_px","depth_vs_endpoints","depth_ratio","minimum_at_endpoint"])
        writer.writerows(rows)
    for name,pos,shift,depth,ratio,edge in rows:
        print(f"{name:8} shift={shift:+.2f} px   depth={depth:.5f}   depth_ratio={ratio:.3f}   endpoint={edge}")
    print("GT endpoints:",points.tolist()," Output:",args.out)


if __name__=="__main__":
    main()
