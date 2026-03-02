# -*- coding: utf-8 -*-
"""
Unified Bot-driven Bayesian CEFR Inference (CSV-only, paper-aligned)
====================================================================
Supports:
(A) Synthetic simulation bank (default): 180 items (6 CEFR x 30)
(B) CSV-based bank: load items from --bank_csv

Core design (matches paper):
- Trials per item: T = step..T_max (default 10..200 step 10)
- Train/Test split within each item trials (default 70/30)
- Posterior: Beta–Binomial (mid bot only)
- Band-based CEFR: posterior mass over CEFR probability bands (Table I)
- QA gap: strong - weak on TEST only
- Outputs: CSV ONLY (no plots)

Paper-aligned outputs:
- Parameter recovery: MAE/RMSE/Corr (summary_by_T)
- Classification: exact/within1 (summary_by_T)
- Predictive: Brier/LogLoss on held-out test (summary_by_T)
- Uncertainty: entropy/confidence/margin + CI95 (items_posterior_Txxx)

Optional CSV bands:
- If CSV contains columns band_low, band_high, CEFR_BANDS are derived by grouping
  on cefr_true and taking min/max per level (for bank-specific reference bands).

Outputs (under --out_dir):
- summary_by_T.csv
- items_posterior_Txxx.csv (per T)
- review_lists/review_Txxx.csv (optional; if --export_review_lists)

Example runs:
1) Synthetic 180:
   python bot_bayes_cefr_unified_csv_only.py --out_dir out_sim --export_review_lists

2) CSV-based:
   python bot_bayes_cefr_unified_csv_only.py --out_dir out_csv --bank_csv question_bank_mo_with_bands_range_utf8sig.csv --export_review_lists
"""

import os, math, argparse
import numpy as np
import pandas as pd

LEVELS = ["A1","A2","B1","B2","C1","C2"]
ORD = {l:i for i,l in enumerate(LEVELS)}

# Default CEFR probability bands (fallback)
CEFR_BANDS_DEFAULT = {
    "A1": (0.85, 0.95),
    "A2": (0.70, 0.85),
    "B1": (0.55, 0.70),
    "B2": (0.40, 0.55),
    "C1": (0.25, 0.40),
    "C2": (0.10, 0.25),
}

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def logit(p):
    p = np.clip(p, 1e-9, 1-1e-9)
    return np.log(p/(1-p))

# --- Beta CDF / PPF helpers (SciPy preferred, mpmath fallback) ---
try:
    from scipy.stats import beta as sp_beta
    def beta_cdf(x, a, b):
        x = float(np.clip(x, 0.0, 1.0))
        return float(sp_beta.cdf(x, a, b))
    def beta_ppf(q, a, b):
        q = float(np.clip(q, 0.0, 1.0))
        return float(sp_beta.ppf(q, a, b))
except Exception:
    import mpmath as mp
    def beta_cdf(x, a, b):
        x = float(np.clip(x, 0.0, 1.0))
        return float(mp.betainc(a, b, 0, x, regularized=True))
    def beta_ppf(q, a, b, tol=1e-6, max_iter=80):
        q = float(np.clip(q, 0.0, 1.0))
        if q <= 0.0: return 0.0
        if q >= 1.0: return 1.0
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

def posterior_mass_by_bands(alpha, beta, cefr_bands):
    masses = []
    for lv in LEVELS:
        lo, hi = cefr_bands[lv]
        m = max(0.0, beta_cdf(hi, alpha, beta) - beta_cdf(lo, alpha, beta))
        masses.append(m)
    s = sum(masses)
    if s <= 0:
        return np.ones(len(LEVELS))/len(LEVELS)
    return np.array(masses)/s

def entropy(probs, eps=1e-12):
    p = np.array(probs, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p/p.sum()
    return float(-(p*np.log(p)).sum())

def distance_to_band_boundary(p, level, cefr_bands):
    lo, hi = cefr_bands[level]
    d_lo = abs(p - lo)
    d_hi = abs(hi - p)
    if d_lo <= d_hi:
        return float(d_lo), float(lo)
    return float(d_hi), float(hi)

def error_type_label(cefr_true, cefr_hat):
    diff = abs(ORD[cefr_true] - ORD[cefr_hat])
    if diff == 0: return "exact"
    if diff == 1: return "adjacent"
    return "skip"

def generate_synthetic_bank_180(seed=42, cefr_bands=None):
    rng = np.random.default_rng(seed)
    rows = []
    item_id = 1
    bands = cefr_bands if cefr_bands is not None else CEFR_BANDS_DEFAULT
    for lv in LEVELS:
        lo, hi = bands[lv]
        for _ in range(30):
            rows.append({
                "item_id": item_id,
                "cefr_true": lv,               # ground truth for evaluation only
                "true_p_mid": float(rng.uniform(lo, hi))
            })
            item_id += 1
    return pd.DataFrame(rows)

def derive_cefr_bands_from_csv(df):
    """
    If CSV includes band_low/band_high, derive CEFR_BANDS by grouping on cefr_true.
    Returns dict or None if not possible.
    """
    if ("band_low" not in df.columns) or ("band_high" not in df.columns):
        return None
    if "cefr_true" not in df.columns:
        return None
    out = {}
    for lv in LEVELS:
        sub = df[df["cefr_true"] == lv]
        if len(sub) == 0:
            return None
        lo = float(np.nanmin(sub["band_low"].astype(float).values))
        hi = float(np.nanmax(sub["band_high"].astype(float).values))
        # sanity clamp
        lo = max(0.0, min(1.0, lo))
        hi = max(0.0, min(1.0, hi))
        if hi <= lo:
            return None
        out[lv] = (lo, hi)
    return out

def load_bank_from_csv(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # CEFR label
    if "cefr_level" in df.columns:
        df["cefr_true"] = df["cefr_level"].astype(str)
    elif "cefr_true" in df.columns:
        df["cefr_true"] = df["cefr_true"].astype(str)
    else:
        raise ValueError("CSV must contain 'cefr_level' or 'cefr_true'.")

    # Item ID
    if "question_id" in df.columns:
        df["item_id"] = df["question_id"].astype(int)
    elif "item_id" in df.columns:
        df["item_id"] = df["item_id"].astype(int)
    else:
        df["item_id"] = np.arange(1, len(df)+1, dtype=int)

    # Keep only valid CEFR labels
    df = df[df["cefr_true"].isin(LEVELS)].copy()
    if len(df) == 0:
        raise ValueError("No valid CEFR rows (A1–C2) after filtering.")

    # Baseline probability for mid bot
    if "true_p_mid" not in df.columns:
        # If missing, sample by CEFR_BANDS_DEFAULT to keep pipeline runnable
        rng = np.random.default_rng(42)
        def _sample(lv):
            lo, hi = CEFR_BANDS_DEFAULT[str(lv)]
            return float(rng.uniform(lo, hi))
        df["true_p_mid"] = df["cefr_true"].apply(_sample)

    # Keep optional band columns if present
    keep = ["item_id","cefr_true","true_p_mid"]
    if "band_low" in df.columns and "band_high" in df.columns:
        keep += ["band_low","band_high"]
    return df[keep].copy()

def simulate_responses(df_bank, T_max, seed=123, mod_mag=0.15, prob_model="logit_sigmoid"):
    rng = np.random.default_rng(seed)
    bots = {"weak": -mod_mag, "mid": 0.0, "strong": +mod_mag}
    resp = {}
    for _, r in df_bank.iterrows():
        iid = int(r["item_id"])
        p_mid = float(r["true_p_mid"])
        base_logit = logit(p_mid)
        for bot, mod in bots.items():
            if prob_model == "logit_sigmoid":
                p = float(sigmoid(base_logit + mod))
            else:
                p = float(np.clip(p_mid + mod, 1e-3, 1-1e-3))
            resp[(bot, iid)] = (rng.random(T_max) < p).astype(int)
    return resp

def safe_logloss(y, p):
    p = float(np.clip(p, 1e-9, 1-1e-9))
    # y is 0/1
    return float(-(y*np.log(p) + (1-y)*np.log(1-p)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out_csv_only")
    ap.add_argument("--bank_csv", type=str, default=None,
                    help="If provided, load item bank from CSV. Otherwise generate synthetic 180.")
    ap.add_argument("--T_max", type=int, default=200)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--seed_bank", type=int, default=42)
    ap.add_argument("--seed_resp", type=int, default=123)
    ap.add_argument("--prob_model", choices=["logit_sigmoid","clip"], default="logit_sigmoid")
    ap.add_argument("--mod_mag", type=float, default=0.15)
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--gap_rho", type=float, default=0.67)
    ap.add_argument("--gap_th", type=float, default=None,
                    help="Override gap threshold. If omitted, use gap_rho * gap_max(mod_mag).")
    ap.add_argument("--entropy_th", type=float, default=0.60)
    ap.add_argument("--conf_th", type=float, default=0.60)
    ap.add_argument("--margin_th", type=float, default=0.10)
    ap.add_argument("--boundary_th", type=float, default=0.01)
    ap.add_argument("--export_review_lists", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    review_dir = os.path.join(args.out_dir, "review_lists")
    if args.export_review_lists:
        ensure_dir(review_dir)

    # ---- Load/generate bank ----
    if args.bank_csv:
        df_bank = load_bank_from_csv(args.bank_csv)
        # Optional: derive CEFR_BANDS from CSV metadata if present
        cefr_bands = derive_cefr_bands_from_csv(df_bank) or CEFR_BANDS_DEFAULT
        df_bank.to_csv(os.path.join(args.out_dir, "item_bank_from_csv.csv"),
                       index=False, encoding="utf-8-sig")
    else:
        cefr_bands = CEFR_BANDS_DEFAULT
        df_bank = generate_synthetic_bank_180(seed=args.seed_bank, cefr_bands=cefr_bands)
        df_bank.to_csv(os.path.join(args.out_dir, "item_bank_180.csv"),
                       index=False, encoding="utf-8-sig")

    N = len(df_bank)

    # gap_th from theory under logit-sigmoid (max separation near p=0.5)
    gap_max = float(sigmoid(args.mod_mag) - sigmoid(-args.mod_mag))
    if args.gap_th is None:
        args.gap_th = float(args.gap_rho * gap_max)

    # simulate once up to T_max, then slice per T
    resp = simulate_responses(df_bank, args.T_max, seed=args.seed_resp,
                              mod_mag=args.mod_mag, prob_model=args.prob_model)

    summary_rows = []
    for T in range(args.step, args.T_max + 1, args.step):
        n_train = int(math.floor(args.train_frac * T))
        n_test = max(1, T - n_train)
        rows = []

        for _, r in df_bank.iterrows():
            iid = int(r["item_id"])
            true_p = float(r["true_p_mid"])
            cefr_true = str(r["cefr_true"])

            mid_seq = resp[("mid", iid)][:T]
            mid_train = mid_seq[:n_train]
            mid_test  = mid_seq[n_train:T]

            k = int(mid_train.sum())
            n = int(len(mid_train))
            alpha = args.alpha0 + k
            beta  = args.beta0 + (n - k)
            p_est = float(alpha / (alpha + beta))
            p_err = float(p_est - true_p)

            # CI95 (paper-aligned)
            ci95_low  = float(beta_ppf(0.025, alpha, beta))
            ci95_high = float(beta_ppf(0.975, alpha, beta))
            ci95_width = float(ci95_high - ci95_low)

            # held-out predictive metrics (paper-aligned)
            if len(mid_test):
                brier = float(np.mean((mid_test - p_est) ** 2))
                logloss = float(np.mean([safe_logloss(y, p_est) for y in mid_test]))
                acc_test_mid = float(np.mean(mid_test))
            else:
                brier = float("nan")
                logloss = float("nan")
                acc_test_mid = float("nan")

            # CEFR posterior mass + outputs (band-based)
            masses = posterior_mass_by_bands(alpha, beta, cefr_bands)
            cefr_hat = LEVELS[int(np.argmax(masses))]
            conf = float(np.max(masses))
            top2 = np.sort(masses)[-2:]
            margin = float(top2[1] - top2[0])
            ent = entropy(masses)

            # boundary distance (true and predicted)
            d_true, near_true = distance_to_band_boundary(true_p, cefr_true, cefr_bands)
            d_hat,  near_hat  = distance_to_band_boundary(p_est, cefr_hat, cefr_bands)

            # QA gap from TEST only (strong - weak)
            weak_test = resp[("weak", iid)][n_train:T]
            strong_test = resp[("strong", iid)][n_train:T]
            acc_weak = float(weak_test.mean()) if len(weak_test) else float("nan")
            acc_strong = float(strong_test.mean()) if len(strong_test) else float("nan")
            gap = float(acc_strong - acc_weak)

            err_type = error_type_label(cefr_true, cefr_hat)

            # QA flags
            flag_entropy = int(ent > args.entropy_th)
            flag_conf    = int(conf < args.conf_th)
            flag_margin  = int(margin < args.margin_th)
            # In simulation, d_true is meaningful; for CSV it is a "label/reference" distance under the chosen bands.
            flag_bound   = int(d_true < args.boundary_th)
            flag_gap     = int(gap < args.gap_th)
            review = int(flag_entropy or flag_conf or flag_margin or flag_bound or flag_gap)

            rows.append({
                "item_id": iid,
                "cefr_true": cefr_true,
                "cefr_hat": cefr_hat,
                "error_type": err_type,
                "true_p_mid": true_p,
                "alpha": alpha,
                "beta": beta,
                "k_train": k,
                "n_train": n,
                "p_est": p_est,
                "p_error": p_err,
                "ci95_low": ci95_low,
                "ci95_high": ci95_high,
                "ci95_width": ci95_width,
                "acc_test_mid": acc_test_mid,
                "brier_test_mid": brier,
                "logloss_test_mid": logloss,
                "cefr_conf": conf,
                "cefr_margin": margin,
                "cefr_entropy": ent,
                "d_to_boundary_true": d_true,
                "nearest_boundary_true": near_true,
                "d_to_boundary_hat": d_hat,
                "nearest_boundary_hat": near_hat,
                "acc_weak_test": acc_weak,
                "acc_strong_test": acc_strong,
                "gap_strong_weak_test": gap,
                "flag_entropy": flag_entropy,
                "flag_conf": flag_conf,
                "flag_margin": flag_margin,
                "flag_boundary": flag_bound,
                "flag_gap": flag_gap,
                "flag_review": review,
                "T": T,
                "n_test": n_test,
            })

        df_items = pd.DataFrame(rows)
        df_items.to_csv(os.path.join(args.out_dir, f"items_posterior_T{T:03d}.csv"),
                        index=False, encoding="utf-8-sig")

        # Summary metrics (paper Table II)
        mae = float(np.mean(np.abs(df_items["p_error"].values)))
        rmse = float(np.sqrt(np.mean(df_items["p_error"].values ** 2)))
        corr = float(np.corrcoef(df_items["true_p_mid"], df_items["p_est"])[0, 1])

        exact = float(np.mean((df_items["cefr_true"] == df_items["cefr_hat"]).astype(int)))
        within1 = float(np.mean((df_items["cefr_true"].map(ORD) - df_items["cefr_hat"].map(ORD)).abs() <= 1))

        # predictive summaries (mean over items; each item contributes its held-out metric)
        brier_mean = float(np.nanmean(df_items["brier_test_mid"].values))
        logloss_mean = float(np.nanmean(df_items["logloss_test_mid"].values))

        high_ent_ct = int((df_items["cefr_entropy"] > args.entropy_th).sum())
        low_gap_ct = int((df_items["gap_strong_weak_test"] < args.gap_th).sum())
        review_ct = int(df_items["flag_review"].sum())

        summary_rows.append({
            "T": T,
            "n_items": N,
            "n_train_trials": int(n_train),
            "n_test_trials": int(n_test),
            "MAE_p": mae,
            "RMSE_p": rmse,
            "Corr_p": corr,
            "CEFR_ExactAcc": exact,
            "CEFR_Within1Acc": within1,
            "Test_Brier": brier_mean,
            "Test_LogLoss": logloss_mean,
            "MeanConf": float(df_items["cefr_conf"].mean()),
            "MeanEntropy": float(df_items["cefr_entropy"].mean()),
            "MeanCI95Width": float(df_items["ci95_width"].mean()),
            "HighEntropyCount(>H_th)": high_ent_ct,
            "LowGapCount(<gap_th)": low_gap_ct,
            "ReviewCount": review_ct,
            "H_th": float(args.entropy_th),
            "gap_th": float(args.gap_th),
            "gap_rho": float(args.gap_rho),
            "gap_max": float(gap_max),
        })

        if args.export_review_lists:
            df_items[df_items["flag_review"] == 1].to_csv(
                os.path.join(review_dir, f"review_T{T:03d}.csv"),
                index=False, encoding="utf-8-sig"
            )

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(args.out_dir, "summary_by_T.csv"),
                  index=False, encoding="utf-8-sig")

    # Also save the bands used (helps audit)
    bands_path = os.path.join(args.out_dir, "cefr_bands_used.csv")
    pd.DataFrame([
        {"cefr_level": lv, "band_low": cefr_bands[lv][0], "band_high": cefr_bands[lv][1]}
        for lv in LEVELS
    ]).to_csv(bands_path, index=False, encoding="utf-8-sig")

    print(f"[done] out_dir={args.out_dir}")
    print("       summary_by_T.csv, items_posterior_Txxx.csv, cefr_bands_used.csv (no plots)")

if __name__ == "__main__":
    main()
