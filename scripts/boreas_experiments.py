"""Real-weather experiments on Boreas (same route, many conditions, GNSS/INS GT).

  --mode weather   E1-real: per-condition SfM on the same 4 places of the route.
  --mode synth     E1-gap : synthetic weather applied to the *sunny* runs at the same
                            places, to compare synthetic vs real degradation.
  --mode xseason   E7     : cross-condition mapping/localisation. Map frames from a sunny
                            run + query frames from another condition at the same place,
                            reconstructed jointly; GT for both lives in one ENU frame.

All runs: 40 frames spaced 1.5 m (60 m of road), car hood cropped, 1024 px, known
(fixed) intrinsics, SIFT + exhaustive matching + COLMAP incremental mapping.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sfmrob import data as D
from sfmrob.metrics import evaluate, poses_from_reconstruction, pose_metrics
from sfmrob.sfm import SfMConfig, run_sfm
from sfmrob.perturb.weather import WEATHER, ground_plane_depth

# Conditions labelled by visual inspection of a frame from each run (see docs/boreas_contact_sheet.jpg)
STRATA = {
    "sunny": ["boreas-2020-12-18-13-44", "boreas-2021-04-08-12-44", "boreas-2021-06-03-16-00",
              "boreas-2021-08-05-13-34", "boreas-2021-11-23-14-27"],
    "overcast": ["boreas-2020-11-26-13-58", "boreas-2021-01-19-15-08", "boreas-2021-11-16-14-10"],
    "snow_ground": ["boreas-2020-12-01-13-26", "boreas-2021-01-15-12-17", "boreas-2021-02-02-14-07"],
    "snowfall": ["boreas-2021-01-26-11-22"],
    "rain": ["boreas-2021-04-29-15-55", "boreas-2021-07-20-17-33"],
    "night": ["boreas-2021-09-14-20-00"],
}
REF_SEQ = "boreas-2021-04-08-12-44"
ANCHOR_DIST = [500, 1500, 2600, 3500]   # metres along the reference run (outbound leg)


def anchors():
    P = D.boreas_camera_poses(REF_SEQ)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P[:, 1:4], axis=0), axis=1))])
    out = []
    for a in ANCHOR_DIST:
        i = int(np.searchsorted(d, a))
        out.append((a, P[i, 1:3].copy(), float(np.arctan2(P[i, 5], P[i, 4]))))
    return out


def segment(seq, anchor, work, n=40):
    a, xy, hd = anchor
    idx = D.boreas_select(seq, xy, hd, n_frames=n)
    if idx is None:
        return None
    return D.boreas_fetch(seq, idx, 1024, work / "boreas_frames" / seq / f"a{a}")


SHARD = (0, 1)
THREADS = -1
_k = [0]


def _mine():
    """Round-robin sharding of work items across Slurm array tasks."""
    _k[0] += 1
    return (_k[0] - 1) % SHARD[1] == SHARD[0]


def _cfg(K, seed=0):
    return SfMConfig(camera_params=f"{K['f']},{K['cx']},{K['cy']},0", seed=seed, num_threads=THREADS)


def _append(row, csv):
    pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)


def _done(csv, keys):
    out = set()
    for f in csv.parent.glob(csv.name.split("__")[0].replace(".csv", "") + "*.csv"):
        df = pd.read_csv(f)
        out |= set(tuple(r) for r in df[keys].astype(str).values)
    return out


def run_weather(work, csv, seeds):
    done = _done(csv, ["seq", "anchor", "seed"])
    for stratum, seqs in STRATA.items():
        for seq in seqs:
            for anc in anchors():
                if not _mine():
                    continue
                seg = segment(seq, anc, work)
                if seg is None:
                    print("no pass", seq, anc[0]); continue
                names, ref, K = seg
                for seed in seeds:
                    if (seq, str(anc[0]), str(seed)) in done:
                        continue
                    t0 = time.time()
                    res = run_sfm(work / "boreas_frames" / seq / f"a{anc[0]}", work / ("boreas_run" + f"_{SHARD[0]}"), _cfg(K, seed), image_names=names)
                    m = evaluate(res, ref, names)
                    row = dict(exp="e1_real_weather", stratum=stratum, seq=seq, anchor=anc[0], seed=seed, **m,
                               mean_intensity=float(np.mean([np.asarray(Image.open(work / "boreas_frames" / seq / f"a{anc[0]}" / n)).mean() for n in names[::8]])),
                               wall_s=round(time.time() - t0, 1))
                    _append(row, csv)
                    print(f"[weather] {stratum:11s} {seq} a{anc[0]} reg={m['n_reg']}/{m['n_input']} auc@10={m['auc@10']:.3f}", flush=True)


def run_synth(work, csv, seeds, severities=(1, 3, 5)):
    done = _done(csv, ["seq", "anchor", "perturb", "severity", "seed"])
    for seq in STRATA["sunny"]:
        for anc in anchors():
            if not _mine():
                continue
            seg = segment(seq, anc, work)
            if seg is None:
                continue
            names, ref, K = seg
            src = work / "boreas_frames" / seq / f"a{anc[0]}"
            h, w = np.asarray(Image.open(src / names[0])).shape[:2]
            depth = ground_plane_depth(h, w, horizon=K["cy"] / h - 0.02, f=K["f"])
            for p in ["fog", "rain", "snow", "night"]:
                for s in severities:
                    for seed in seeds:
                        if (seq, str(anc[0]), p, str(s), str(seed)) in done:
                            continue
                        out = work / ("boreas_synth" + f"_{SHARD[0]}")
                        shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
                        for n in names:
                            rng = np.random.default_rng(int(hashlib.md5(f'{n}|{p}|{s}|{seed}'.encode()).hexdigest()[:12], 16))
                            img = np.asarray(Image.open(src / n).convert("RGB"))
                            Image.fromarray(WEATHER[p](img, s, rng, depth)).save(out / n, quality=95)
                        res = run_sfm(out, work / ("boreas_run" + f"_{SHARD[0]}"), _cfg(K, seed), image_names=names)
                        m = evaluate(res, ref, names)
                        _append(dict(exp="e1_synth_on_boreas", seq=seq, anchor=anc[0], perturb=p, severity=s, seed=seed, **m), csv)
                        print(f"[synth] {seq} a{anc[0]} {p}{s} reg={m['n_reg']} auc@10={m['auc@10']:.3f}", flush=True)


def run_xseason(work, csv, seeds, map_seq=REF_SEQ):
    done = _done(csv, ["query_seq", "anchor", "seed"])
    for anc in anchors():
        mseg = segment(map_seq, anc, work)
        if mseg is None:
            continue
        mnames, mref, K = mseg
        for stratum, seqs in STRATA.items():
            for qseq in seqs:
                if qseq == map_seq:
                    continue
                for seed in seeds:
                    if not _mine() or (qseq, str(anc[0]), str(seed)) in done:
                        continue
                    qseg = segment(qseq, anc, work)
                    if qseg is None:
                        continue
                    qnames, qref, _ = qseg
                    qnames = qnames[::2]  # 20 query frames
                    joint = work / ("boreas_joint" + f"_{SHARD[0]}")
                    shutil.rmtree(joint, ignore_errors=True); joint.mkdir(parents=True)
                    for n in mnames:
                        os.symlink(work / "boreas_frames" / map_seq / f"a{anc[0]}" / n, joint / n)
                    for n in qnames:
                        os.symlink(work / "boreas_frames" / qseq / f"a{anc[0]}" / n, joint / n)
                    names = mnames + qnames
                    ref = {**mref, **{k: qref[k] for k in qnames}}
                    res = run_sfm(joint, work / ("boreas_run" + f"_{SHARD[0]}"), _cfg(K, seed), image_names=names)
                    rec = res.best
                    est = poses_from_reconstruction(rec) if rec else {}
                    # map coverage of the main model, query registration *into the same model*
                    m_in = sum(n in est for n in mnames) / len(mnames)
                    q_in = sum(n in est for n in qnames) / len(qnames) if m_in > 0.5 else 0.0
                    # cross pairs only (map, query): this is the localisation accuracy
                    cross = []
                    from sfmrob.metrics import _rotmat_angle, _vec_angle
                    for a in mnames[::4]:
                        for b in qnames:
                            Ra, ta = ref[a]; Rb, tb = ref[b]
                            Rr = Rb @ Ra.T; tr = tb - Rr @ ta
                            if a in est and b in est and m_in > 0.5:
                                Ea, ea = est[a]; Eb, eb = est[b]
                                Re = Eb @ Ea.T; te = eb - Re @ ea
                                cross.append(max(_rotmat_angle(Re.T @ Rr), _vec_angle(te, tr)))
                            else:
                                cross.append(np.inf)
                    cross = np.array(cross)
                    auc = {f"cross_auc@{t}": float(np.mean([(cross <= x).mean() for x in np.linspace(0, t, 101)[1:]])) for t in (5, 10, 20)}
                    row = dict(exp="e7_xseason", map_seq=map_seq, query_seq=qseq, stratum=stratum, anchor=anc[0],
                               seed=seed, map_reg=m_in, query_reg=q_in, n_models=len(res.models), **auc)
                    _append(row, csv)
                    print(f"[xseason] a{anc[0]} {stratum:11s} {qseq} map={m_in:.2f} query={q_in:.2f} cross_auc@10={auc['cross_auc@10']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["weather", "synth", "xseason"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--work", default=os.environ.get("SFMROB_WORK", "/tmp/sfmrob_work"))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=-1)
    a = ap.parse_args()
    global SHARD, THREADS
    SHARD = tuple(map(int, a.shard.split("/")))
    THREADS = a.threads
    work = Path(a.work)
    csv = Path(a.out) / (f"boreas_{a.mode}.csv" if SHARD[1] == 1 else f"boreas_{a.mode}__{SHARD[0]:03d}.csv")
    csv.parent.mkdir(parents=True, exist_ok=True)
    {"weather": run_weather, "synth": run_synth, "xseason": run_xseason}[a.mode](work, csv, a.seeds)


if __name__ == "__main__":
    main()
