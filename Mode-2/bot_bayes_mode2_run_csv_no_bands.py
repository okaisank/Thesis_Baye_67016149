# -*- coding: utf-8 -*-
"""
Band-agnostic Bayesian Item Calibration (Mode 2) — RUN (SIMULATE + CSV)
======================================================================

What this script does
---------------------
1) Mode 2 (band-agnostic) inference: estimate continuous item difficulty p via
   Beta posterior using ONLY mid-bot responses.
2) Uncertainty is reported directly on p (posterior variance + CI95 width).
3) If a discrete view is needed, it learns anonymous bins L1..L6 from p_est
   using quantiles (NOT CEFR A1..C2 and NOT any CEFR cutpoints).
4) Weak/Strong bots are used ONLY for QA gap on TEST.

Key requirement from user
-------------------------
- "ตัด band ออก": This file DOES NOT use CEFR_BANDS for inference.
- It can load an item bank from CSV and derive true_p_mid from either:
    (a) a direct probability column (e.g., true_p_mid), OR
    (b) a range (low/high) columns (e.g., band_low/band_high) using midpoint.

Typical runs
------------
python bot_bayes_mode2_run_csv_no_bands.py `
  --bank_csv question_bank_mo_with_bands_range_utf8sig.csv `
  --id_col question_id `
  --truth_col cefr_level `
  --p_from_range band_low band_high `
  --out_dir out_mode2_T200 `
  --T_max 200 `
  --step 10 `
  --seed 42

1) Run with your uploaded CSV (has band_low/band_high):

   python bot_bayes_mode2_run_csv_no_bands.py `
     --bank_csv question_bank_mo_with_bands_range_utf8sig.csv `
     --id_col question_id --truth_col cefr_level `
     --p_from_range band_low band_high `
     --out_dir out_mode2_T200 --T_max 200 --seed 42

2) Run with a CSV that already has true_p_mid:

   python bot_bayes_mode2_run_csv_no_bands.py `
     --bank_csv question_bank_mo_with_bands_range_utf8sig.csv --id_col item_id --p_col true_p_mid `
     --out_dir out_mode2_T200 --T_max 200

3) Run synthetic 180-item bank (no CSV):

   python bot_bayes_mode2_run_csv_no_bands.py --out_dir out_mode2_T200_sim --T_max 200

Outputs
-------
- out_dir/summary_by_T_mode2.csv
- out_dir/items_posterior_Txxx_mode2.csv
- out_dir/top10_high_uncertainty_Txxx_mode2.csv
- out_dir/top10_low_gap_Txxx_mode2.csv

"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpmath as mp


# -----------------------------
# Synthetic bank config (USED ONLY when --bank_csv is empty)
# Inference NEVER uses these.
# -----------------------------
LEVELS_TRUE = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_BANDS_TRUE = {
    "A1": (0.85, 0.95),
    "A2": (0.70, 0.85),
    "B1": (0.55, 0.70),
    "B2": (0.40, 0.55),
    "C1": (0.25, 0.40),
    "C2": (0.10, 0.25),
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def beta_cdf(x, a, b):
    x = float(np.clip(x, 0.0, 1.0))
    return float(mp.betainc(a, b, 0, x, regularized=True))


def beta_ppf(q, a, b, tol=1e-6, max_iter=80):
    """Inverse CDF for Beta(a,b) using bisection on [0,1]."""
    q = float(np.clip(q, 0.0, 1.0))
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        cmid = beta_cdf(mid, a, b)
        if abs(cmid - q) < tol:
            return mid
        if cmid < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_item_bank_180(seed=42):
    """Create 180 items (6 CEFR levels x 30) with hidden true_p_mid in each band."""
    rng = np.random.default_rng(seed)
    rows = []
    item_id = 1
    for lv in LEVELS_TRUE:
        lo, hi = CEFR_BANDS_TRUE[lv]
        for _ in range(30):
            true_p = float(rng.uniform(lo, hi))
            rows.append({
                "item_id": item_id,
                "cefr_true": lv,  # analysis only
                "true_p_mid": true_p,
            })
            item_id += 1
    return pd.DataFrame(rows)


def load_item_bank_csv(
    path: str,
    id_col: str,
    truth_col: str,
    p_col: str | None,
    range_low_col: str | None,
    range_high_col: str | None,
):
    """Load item bank from CSV.

    Required:
      - id_col
    Provide either:
      (A) p_col (direct probability in [0,1])
      OR
      (B) range_low_col and range_high_col (midpoint used as true_p_mid)

    Any band/range columns are ignored for inference; only used to derive true_p_mid.
    """

    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = set(df.columns)
    if id_col not in cols:
        raise ValueError(f"Missing id_col='{id_col}' in CSV columns: {list(df.columns)}")

    # derive true_p_mid
    if p_col and p_col in cols:
        p = df[p_col].astype(float)
    elif range_low_col and range_high_col and range_low_col in cols and range_high_col in cols:
        lo = df[range_low_col].astype(float)
        hi = df[range_high_col].astype(float)
        p = (lo + hi) / 2.0
    else:
        raise ValueError(
            "Could not derive true_p_mid. Provide either --p_col <col> (exists in CSV) "
            "or --p_from_range <low_col> <high_col> (both exist in CSV)."
        )

    out = pd.DataFrame({
        "item_id": df[id_col].astype(int),
        "true_p_mid": p.astype(float).clip(1e-6, 1 - 1e-6),
    })

    # optional truth label for analysis only
    if truth_col and truth_col in cols:
        out["cefr_true"] = df[truth_col].astype(str)
    else:
        out["cefr_true"] = "NA"

    return out.sort_values("item_id").reset_index(drop=True)


def simulate_responses(df_bank, T_max, seed=42, prob_model="logit_sigmoid", mod_mag=0.60):
    """Simulate three bots: weak, mid, strong.

    - mid uses true_p_mid directly
    - weak/strong are logit-shifted by +/- mod_mag (recommended)
    """
    rng = np.random.default_rng(seed)
    resp = {}  # key: (bot, item_id) -> array length T_max
    for _, r in df_bank.iterrows():
        iid = int(r["item_id"])
        p_mid = float(r["true_p_mid"])
        base_logit = logit(p_mid)

        for bot, mod in [("mid", 0.0), ("weak", -mod_mag), ("strong", +mod_mag)]:
            if prob_model == "logit_sigmoid":
                p = float(sigmoid(base_logit + mod))
            else:
                # simple clip model (kept for compatibility)
                p = float(
                    np.clip(
                        p_mid + (0.08 if bot == "strong" else (-0.08 if bot == "weak" else 0.0)),
                        1e-3,
                        1 - 1e-3,
                    )
                )
            resp[(bot, iid)] = (rng.random(T_max) < p).astype(int)

    return resp


def learn_anonymous_bins(p_est, n_bins=6):
    """Learn anonymous bins L1..Ln from p_est via quantiles."""
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(p_est, q)

    # ensure strictly increasing edges (avoid ties)
    for j in range(1, len(edges)):
        if edges[j] <= edges[j - 1]:
            edges[j] = edges[j - 1] + 1e-6
    edges[0] = max(0.0, float(edges[0]))
    edges[-1] = min(1.0, float(edges[-1]))

    cutpoints = edges[1:-1]
    idx = np.digitize(p_est, cutpoints, right=False)  # 0..n_bins-1
    labels = np.array([f"L{k+1}" for k in idx], dtype=object)

    lo = edges[idx]
    hi = edges[idx + 1]
    d_lo = np.abs(p_est - lo)
    d_hi = np.abs(hi - p_est)
    d = np.minimum(d_lo, d_hi)
    nearest = np.where(d_lo <= d_hi, lo, hi)

    return edges, labels, lo, hi, d, nearest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out_mode2")
    ap.add_argument("--T_max", type=int, default=200)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--prob_model", choices=["logit_sigmoid", "clip"], default="logit_sigmoid")
    ap.add_argument("--mod_mag", type=float, default=0.60)
    ap.add_argument("--gap_th", type=float, default=0.050)

    # CSV bank options (NO CEFR bands used for inference)
    ap.add_argument("--bank_csv", default="", help="Path to item bank CSV. If empty, uses synthetic 180-item bank.")
    ap.add_argument("--id_col", default="item_id", help="Item-id column in CSV (e.g., question_id).")
    ap.add_argument("--truth_col", default="cefr_true", help="Optional truth label column for analysis only (e.g., cefr_level).")
    ap.add_argument("--p_col", default="", help="Direct probability column in CSV (e.g., true_p_mid).")
    ap.add_argument(
        "--p_from_range",
        nargs=2,
        default=None,
        metavar=("LOW_COL", "HIGH_COL"),
        help="Derive true_p_mid as midpoint from range columns (e.g., band_low band_high).",
    )

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    fig_dir = os.path.join(args.out_dir, "figures")
    ensure_dir(fig_dir)

    # 1) Item bank
    if args.bank_csv.strip():
        low_col = high_col = None
        if args.p_from_range is not None:
            low_col, high_col = args.p_from_range[0], args.p_from_range[1]

        df_bank = load_item_bank_csv(
            args.bank_csv.strip(),
            id_col=args.id_col,
            truth_col=args.truth_col,
            p_col=(args.p_col.strip() or None),
            range_low_col=low_col,
            range_high_col=high_col,
        )
        df_bank.to_csv(os.path.join(args.out_dir, "item_bank_loaded_mode2.csv"), index=False)
    else:
        df_bank = generate_item_bank_180(seed=args.seed)
        df_bank.to_csv(os.path.join(args.out_dir, "item_bank_180.csv"), index=False)

    # 2) Simulate responses
    resp = simulate_responses(
        df_bank,
        T_max=args.T_max,
        seed=args.seed,
        prob_model=args.prob_model,
        mod_mag=args.mod_mag,
    )

    summary_rows = []

    # 3) Sweep
    for T in range(args.step, args.T_max + 1, args.step):
        rows = []
        n_train = int(round(args.train_frac * T))
        n_test = T - n_train

        for _, r in df_bank.iterrows():
            iid = int(r["item_id"])
            true_p = float(r["true_p_mid"])

            mid = resp[("mid", iid)][:T]
            mid_train = mid[:n_train]
            mid_test = mid[n_train:T]

            k = int(mid_train.sum())
            n = int(len(mid_train))

            alpha = args.alpha0 + k
            beta = args.beta0 + (n - k)

            p_est = float(alpha / (alpha + beta))
            var_p = float((alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1)))
            ci_low = beta_ppf(0.025, alpha, beta)
            ci_high = beta_ppf(0.975, alpha, beta)
            ci_width = float(ci_high - ci_low)

            # mid test metrics (optional)
            acc_test_mid = float(mid_test.mean()) if len(mid_test) else float("nan")
            brier = float(np.mean((mid_test - p_est) ** 2)) if len(mid_test) else float("nan")
            pe = float(np.clip(p_est, 1e-9, 1 - 1e-9))
            logloss = (
                float(-np.mean(mid_test * np.log(pe) + (1 - mid_test) * np.log(1 - pe)))
                if len(mid_test)
                else float("nan")
            )

            # QA gap from TEST only (weak/strong)
            weak_test = resp[("weak", iid)][n_train:T]
            strong_test = resp[("strong", iid)][n_train:T]
            acc_weak = float(weak_test.mean()) if len(weak_test) else float("nan")
            acc_strong = float(strong_test.mean()) if len(strong_test) else float("nan")
            gap = float(acc_strong - acc_weak) if (len(weak_test) and len(strong_test)) else float("nan")

            rows.append(
                {
                    "item_id": iid,
                    "cefr_true": r.get("cefr_true", "NA"),
                    "true_p_mid": true_p,
                    "alpha": alpha,
                    "beta": beta,
                    "k_train": k,
                    "n_train": n,
                    "p_est": p_est,
                    "p_error": float(p_est - true_p),
                    "acc_test_mid": acc_test_mid,
                    "brier_test_mid": brier,
                    "logloss_test_mid": logloss,
                    "acc_weak_test": acc_weak,
                    "acc_strong_test": acc_strong,
                    "gap_weak_strong_test": gap,
                    "var_p": var_p,
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "ci95_width": float(ci_width),
                }
            )

        df_items = pd.DataFrame(rows)

        # 4) Anonymous bins + boundary distance
        edges, lvl_hat, lo_hat, hi_hat, d_hat, near_hat = learn_anonymous_bins(df_items["p_est"].values, n_bins=6)
        df_items["level_hat"] = lvl_hat
        df_items["band_hat_lo"] = lo_hat
        df_items["band_hat_hi"] = hi_hat
        df_items["d_to_boundary_hat"] = d_hat
        df_items["nearest_boundary_hat"] = near_hat
        df_items["learned_edges_q"] = ",".join([f"{e:.6f}" for e in edges])

        df_items.to_csv(os.path.join(args.out_dir, f"items_posterior_T{T:03d}_mode2.csv"), index=False)

        # 5) Summary
        mae = float(np.mean(np.abs(df_items["p_error"].values)))
        rmse = float(np.sqrt(np.mean(df_items["p_error"].values ** 2)))

        x = df_items["d_to_boundary_hat"].values
        y = df_items["ci95_width"].values
        r_xy = float(np.corrcoef(x, y)[0, 1])

        n_low_gap = int(np.sum(df_items["gap_weak_strong_test"].values < args.gap_th))

        summary_rows.append(
            {
                "T": T,
                "n_items": int(len(df_items)),
                "train_frac": args.train_frac,
                "MAE_p": mae,
                "RMSE_p": rmse,
                "MeanVarP": float(df_items["var_p"].mean()),
                "MeanCI95Width": float(df_items["ci95_width"].mean()),
                "MeanBoundaryDist": float(df_items["d_to_boundary_hat"].mean()),
                "Pearson_r(dist, CI95Width)": r_xy,
                "QA_low_gap_count": n_low_gap,
                "gap_th": args.gap_th,
                "learned_edges_q": ",".join([f"{e:.6f}" for e in edges]),
            }
        )

        # 6) Artifacts at T_max only  (EXPORT CSV instead of plotting)
    if T == args.T_max:
       df = df_items.copy()

    # ---------- (A) Scatter data: true_p vs p_est ----------
    df_scatter = df[[
        "item_id", "cefr_true", "true_p_mid", "p_est",
        "p_error", "ci95_low", "ci95_high", "ci95_width",
        "var_p", "d_to_boundary_hat", "nearest_boundary_hat",
        "gap_weak_strong_test"
    ]].copy()
    df_scatter.to_csv(
        os.path.join(args.out_dir, f"T{args.T_max:03d}_truep_vs_pest_mode2.csv"),
        index=False
    )

    # ---------- (B) CI95 width raw values ----------
    df_ciw = df[["item_id", "ci95_width"]].copy()
    df_ciw.to_csv(
        os.path.join(args.out_dir, f"T{args.T_max:03d}_ci95width_values_mode2.csv"),
        index=False
    )

    # ---------- (C) CI95 width histogram bins + counts ----------
    # Match the old plot behavior (bins=20)
    vals = df["ci95_width"].values.astype(float)
    counts, edges = np.histogram(vals, bins=20)
    df_hist = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts
    })
    df_hist.to_csv(
        os.path.join(args.out_dir, f"T{args.T_max:03d}_ci95width_hist_mode2.csv"),
        index=False
    )

    # ---------- (D) Scatter data: uncertainty vs boundary distance ----------
    x = df["d_to_boundary_hat"].values.astype(float)
    y = df["ci95_width"].values.astype(float)
    r_xy = float(np.corrcoef(x, y)[0, 1])

    df_uvb = df[[
        "item_id", "d_to_boundary_hat", "ci95_width",
        "p_est", "true_p_mid", "cefr_true"
    ]].copy()
    df_uvb["pearson_r"] = r_xy  # same value repeated per row (convenient for spreadsheets)
    df_uvb.to_csv(
        os.path.join(args.out_dir, f"T{args.T_max:03d}_uncertainty_vs_boundary_mode2.csv"),
        index=False
    )

    # ---------- (E) Regression line points (optional) ----------
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(float(x.min()), float(x.max()), 200)
    yy = m * xx + b
    df_line = pd.DataFrame({"x": xx, "y_fit": yy, "slope": m, "intercept": b, "pearson_r": r_xy})
    df_line.to_csv(
        os.path.join(args.out_dir, f"T{args.T_max:03d}_uncertainty_vs_boundary_fitline_mode2.csv"),
        index=False
    )

    # ---------- (F) Top-10 exports (keep as-is) ----------
    df_lowgap = df.sort_values("gap_weak_strong_test", ascending=True).head(10)
    df_lowgap.to_csv(os.path.join(args.out_dir, f"top10_low_gap_T{T:03d}_mode2.csv"), index=False)

    df_highunc = df.sort_values("ci95_width", ascending=False).head(10)
    df_highunc.to_csv(os.path.join(args.out_dir, f"top10_high_uncertainty_T{T:03d}_mode2.csv"), index=False)

    # Optional: remove figure folder creation if you no longer need it

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(args.out_dir, "summary_by_T_mode2.csv"), index=False)

    print("Done.")
    print("Outputs in:", args.out_dir)


if __name__ == "__main__":
    main()
