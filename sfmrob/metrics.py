"""Metrics shared by all experiments.

Structure-only metrics (no ground truth needed):
  reg_frac, n_models, n_points3D, mean_track_len, mean_reproj_err, obs_per_image

Pose metrics against a reference (GT or clean-run reconstruction):
  * pairwise relative-pose errors for *all* input image pairs (IMC-style). A pair
    with an unregistered image (or images in different models) counts as failed
    (error = inf), so the AUC penalises both inaccuracy and failure to register.
      - rot_err  = angle(R_rel_est^T R_rel_ref)
      - trans_err = angle between relative translation directions
      - AUC@{5,10,20} deg of max(rot_err, trans_err)
  * ATE: RMSE of camera centres after robust Sim(3) alignment, normalised by the
    reference scene extent (median distance of centres to their centroid).
"""
from __future__ import annotations

import numpy as np


def _rotmat_angle(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _vec_angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 180.0
    c = abs(np.dot(a, b) / (na * nb))   # sign ambiguity: use the unsigned angle like IMC
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def poses_from_reconstruction(rec) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (R_cw, t_cw) world-to-camera."""
    out = {}
    for _, im in rec.images.items():
        if not im.has_pose:
            continue
        T = im.cam_from_world()
        M = T.matrix()
        out[im.name] = (np.asarray(M[:3, :3]), np.asarray(M[:3, 3]))
    return out


def structure_metrics(rec, n_input: int, n_models: int) -> dict:
    if rec is None:
        return dict(n_input=n_input, n_reg=0, reg_frac=0.0, n_models=0, n_points3D=0,
                    mean_track_len=np.nan, mean_reproj_err=np.nan, obs_per_image=np.nan)
    n_reg = rec.num_reg_images()
    return dict(
        n_input=n_input, n_reg=n_reg, reg_frac=n_reg / max(n_input, 1), n_models=n_models,
        n_points3D=rec.num_points3D(),
        mean_track_len=float(rec.compute_mean_track_length()) if rec.num_points3D() else np.nan,
        mean_reproj_err=float(rec.compute_mean_reprojection_error()) if rec.num_points3D() else np.nan,
        obs_per_image=float(rec.compute_mean_observations_per_reg_image()) if n_reg else np.nan,
    )


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale=True):
    """Find s,R,t minimising ||dst - (s R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    C = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = (D * np.diag(S)).sum() / var_s if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def robust_sim3(src, dst, iters=500, thresh=None, seed=0):
    rng = np.random.default_rng(seed)
    n = len(src)
    if n < 3:
        return None
    if thresh is None:
        thresh = 0.05 * np.median(np.linalg.norm(dst - dst.mean(0), axis=1)) + 1e-9
    best, best_inl = None, -1
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            s, R, t = umeyama(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        err = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1)
        inl = (err < thresh).sum()
        if inl > best_inl:
            best_inl, best = inl, err < thresh
    if best is None or best.sum() < 3:
        best = np.ones(n, bool)
    return umeyama(src[best], dst[best])


def pose_metrics(est: dict, ref: dict, names: list[str], thresholds=(5, 10, 20)) -> dict:
    """est/ref: name -> (R_cw, t_cw). names: all *input* images of the run."""
    names = [n for n in names if n in ref]
    errs = []
    rot_errs, tr_errs = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            Ra, ta = ref[a]; Rb, tb = ref[b]
            Rrel_ref = Rb @ Ra.T
            trel_ref = tb - Rrel_ref @ ta
            if a in est and b in est:
                Ea, ea = est[a]; Eb, eb = est[b]
                Rrel = Eb @ Ea.T
                trel = eb - Rrel @ ea
                er = _rotmat_angle(Rrel.T @ Rrel_ref)
                et = _vec_angle(trel, trel_ref)
                rot_errs.append(er); tr_errs.append(et)
                errs.append(max(er, et))
            else:
                errs.append(np.inf)
    errs = np.array(errs) if errs else np.array([np.inf])
    out = {}
    for th in thresholds:
        # area under the accuracy-vs-threshold curve, normalised to [0,1]
        ts = np.linspace(0, th, 101)[1:]
        out[f"auc@{th}"] = float(np.mean([(errs <= x).mean() for x in ts]))
    out["med_rot_err"] = float(np.median(rot_errs)) if rot_errs else np.nan
    out["med_trans_err"] = float(np.median(tr_errs)) if tr_errs else np.nan
    # ATE on camera centres
    common = [n for n in names if n in est]
    if len(common) >= 3:
        C_est = np.array([-est[n][0].T @ est[n][1] for n in common])
        C_ref = np.array([-ref[n][0].T @ ref[n][1] for n in common])
        sim = robust_sim3(C_est, C_ref)
        s, R, t = sim
        al = s * (R @ C_est.T).T + t
        all_ref = np.array([-ref[n][0].T @ ref[n][1] for n in names])
        extent = np.median(np.linalg.norm(all_ref - all_ref.mean(0), axis=1)) + 1e-9
        out["ate_rel"] = float(np.sqrt(np.mean(np.sum((al - C_ref) ** 2, 1))) / extent)
        out["ate_abs"] = float(np.sqrt(np.mean(np.sum((al - C_ref) ** 2, 1))))
    else:
        out["ate_rel"] = np.nan
        out["ate_abs"] = np.nan
    return out


def evaluate(result, ref_poses: dict | None, names: list[str]) -> dict:
    rec = result.best
    m = structure_metrics(rec, result.n_input, len(result.models))
    m.update({k: round(v, 2) for k, v in result.timings.items()})
    if ref_poses is not None:
        est = poses_from_reconstruction(rec) if rec is not None else {}
        m.update(pose_metrics(est, ref_poses, names))
    m["error"] = result.error
    return m
