# -*- coding: utf-8 -*-
"""
Unified Bot-driven Bayesian CEFR Inference (CSV-only, no plots)
==============================================================
- Supports TWO modes in one script:
  (A) Synthetic simulation bank (default): 180 items (6 CEFR x 30)
  (B) CSV-based bank: load items from --bank_csv

- Trials per item: T = step..T_max (default 10..200 step 10)
- Train/Test split within item trials (default 70/30)
- Posterior: Beta–Binomial (mid bot only)
- Band-based CEFR: posterior mass over fixed probability bands (Table I)
- QA gap: strong - weak on TEST only
- Outputs: CSV ONLY (no plots)

Outputs (under --out_dir):
- summary_by_T.csv
- items_posterior_Txxx.csv (per T)
- review_lists/review_Txxx.csv (optional; if --export_review_lists)

Example runs:
1) Synthetic 180:
   python .\bot_bayes_cefr_unified_csv_only.py --out_dir out_sim --export_review_lists

2) CSV-based:
   python bot_bayes_cefr_unified_csv_only.py --out_dir out_csv --bank_csv question_bank_mo_with_bands_range_utf8sig.csv
"""

import os, math, argparse, random
import numpy as np
import pandas as pd

LEVELS = ["A1","A2","B1","B2","C1","C2"]
ORD = {l:i for i,l in enumerate(LEVELS)}

# Fixed CEFR probability bands (mid baseline) — used for band-based labeling
CEFR_BANDS = {
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

# Beta CDF: SciPy if available; fallback to mpmath
try:
    from scipy.stats import beta as sp_beta
    def beta_cdf(x, a, b):
        x = float(np.clip(x, 0.0, 1.0))
        return float(sp_beta.cdf(x, a, b))
except Exception:
    import mpmath as mp
    def beta_cdf(x, a, b):
        x = float(np.clip(x, 0.0, 1.0))
        return float(mp.betainc(a, b, 0, x, regularized=True))

def posterior_mass_by_bands(alpha, beta):
    masses = []
    for lv in LEVELS:
        lo, hi = CEFR_BANDS[lv]
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

def distance_to_band_boundary(p, level):
    lo, hi = CEFR_BANDS[level]
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

def generate_synthetic_bank_180(seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    item_id = 1
    for lv in LEVELS:
        lo, hi = CEFR_BANDS[lv]
        for _ in range(30):
            rows.append({
                "item_id": item_id,
                "cefr_true": lv,               # ground truth for evaluation only
                "true_p_mid": float(rng.uniform(lo, hi))
            })
            item_id += 1
    return pd.DataFrame(rows)

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

    # Baseline probability for mid bot (if absent, sample from CEFR_BANDS)
    if "true_p_mid" not in df.columns:
        rng = np.random.default_rng(42)
        def _sample(lv):
            lo, hi = CEFR_BANDS[str(lv)]
            return float(rng.uniform(lo, hi))
        df["true_p_mid"] = df["cefr_true"].apply(_sample)

    # Keep only valid CEFR labels
    df = df[df["cefr_true"].isin(LEVELS)].copy()
    if len(df) == 0:
        raise ValueError("No valid CEFR rows (A1–C2) after filtering.")
    return df[["item_id","cefr_true","true_p_mid"]].copy()

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
    ap.add_argument("--export_review_lists", action="store_true",
                    help="If set, export review_Txxx.csv under review_lists/")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    review_dir = os.path.join(args.out_dir, "review_lists")
    if args.export_review_lists:
        ensure_dir(review_dir)

    # Item bank
    if args.bank_csv:
        df_bank = load_bank_from_csv(args.bank_csv)
        df_bank.to_csv(os.path.join(args.out_dir, "item_bank_from_csv.csv"), index=False, encoding="utf-8-sig")
    else:
        df_bank = generate_synthetic_bank_180(seed=args.seed_bank)
        df_bank.to_csv(os.path.join(args.out_dir, "item_bank_180.csv"), index=False, encoding="utf-8-sig")

    N = len(df_bank)

    # gap_th from theory under logit-sigmoid (max separation near p=0.5)
    gap_max = float(sigmoid(args.mod_mag) - sigmoid(-args.mod_mag))
    if args.gap_th is None:
        args.gap_th = float(args.gap_rho * gap_max)

    # Simulate once up to T_max, then slice per T
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

            # CEFR posterior mass + outputs
            masses = posterior_mass_by_bands(alpha, beta)
            cefr_hat = LEVELS[int(np.argmax(masses))]
            conf = float(np.max(masses))
            # margin = top1 - top2
            top2 = np.sort(masses)[-2:]
            margin = float(top2[1] - top2[0])
            ent = entropy(masses)

            # boundary distance (true and predicted)
            d_true, near_true = distance_to_band_boundary(true_p, cefr_true)
            d_hat,  near_hat  = distance_to_band_boundary(p_est, cefr_hat)

            # QA gap from TEST only (strong - weak)
            weak_test = resp[("weak", iid)][n_train:T]
            strong_test = resp[("strong", iid)][n_train:T]
            acc_weak = float(weak_test.mean()) if len(weak_test) else float("nan")
            acc_strong = float(strong_test.mean()) if len(strong_test) else float("nan")
            gap = float(acc_strong - acc_weak)

            err_type = error_type_label(cefr_true, cefr_hat)

            # QA flags (band-based policy)
            flag_entropy = int(ent > args.entropy_th)
            flag_conf    = int(conf < args.conf_th)
            flag_margin  = int(margin < args.margin_th)
            flag_bound   = int(d_true < args.boundary_th)  # simulation: use true cutpoint distance
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
                "p_est": p_est,
                "p_error": p_err,
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
                "n_train": n_train,
                "n_test": n_test,
            })

        df_items = pd.DataFrame(rows)
        df_items.to_csv(os.path.join(args.out_dir, f"items_posterior_T{T:03d}.csv"),
                        index=False, encoding="utf-8-sig")

        # Summary metrics
        mae = float(np.mean(np.abs(df_items["p_error"].values)))
        rmse = float(np.sqrt(np.mean(df_items["p_error"].values ** 2)))
        corr = float(np.corrcoef(df_items["true_p_mid"], df_items["p_est"])[0, 1])

        exact = float(np.mean((df_items["cefr_true"] == df_items["cefr_hat"]).astype(int)))
        within1 = float(np.mean((df_items["cefr_true"].map(ORD) - df_items["cefr_hat"].map(ORD)).abs() <= 1))

        low_gap_ct = int((df_items["gap_strong_weak_test"] < args.gap_th).sum())
        high_ent_ct = int((df_items["cefr_entropy"] > args.entropy_th).sum())
        review_ct = int(df_items["flag_review"].sum())

        summary_rows.append({
            "T": T,
            "n_items": N,
            "n_train": n_train,
            "n_test": n_test,
            "MAE_p": mae,
            "RMSE_p": rmse,
            "Corr_p": corr,
            "CEFR_ExactAcc": exact,
            "CEFR_Within1Acc": within1,
            "MeanConf": float(df_items["cefr_conf"].mean()),
            "MeanEntropy": float(df_items["cefr_entropy"].mean()),
            "HighEntropyCount(>H_th)": high_ent_ct,
            "LowGapCount(<gap_th)": low_gap_ct,
            "ReviewCount": review_ct,
            "H_th": args.entropy_th,
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
    df_sum.to_csv(os.path.join(args.out_dir, "summary_by_T.csv"), index=False, encoding="utf-8-sig")

    print(f"[done] out_dir={args.out_dir}")
    print(f"       summary_by_T.csv + items_posterior_Txxx.csv (no plots)")

if __name__ == "__main__":
    main()