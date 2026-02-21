### Built by Matthew Honorio Oliveira and Franco Cabral

import numpy as np

DEG = np.pi / 180.0

def P_VV(alpha, beta, theta, phi, visibility=1.0):
    """
    Probability both photons project onto the analyzers' transmission axes.
    State: cos(theta)|HH> + exp(i phi) sin(theta)|VV>

    alpha, beta, theta, phi in radians.
    visibility in [0,1] scales the interference term (simple decoherence/imperfection model).
    """
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta), np.cos(beta)
    st, ct = np.sin(theta), np.cos(theta)

    # Ideal expression (matches the lab manual form)
    term1 = (sa**2) * (sb**2) * (ct**2)
    term2 = (ca**2) * (cb**2) * (st**2)
    inter = 0.25 * np.sin(2*alpha) * np.sin(2*beta) * np.sin(2*theta) * np.cos(phi)

    p = term1 + term2 + visibility * inter

    # Numerical safety: keep it in [0,1]
    return np.clip(p, 0.0, 1.0)

def measure_counts(alpha, beta, theta, phi, pair_rate_hz, T_s, bg_rate_hz, rng, visibility=1.0):
    """
    One coincidence measurement at (alpha,beta). Returns a Poisson-sampled count.
    """
    mean_signal = pair_rate_hz * T_s * P_VV(alpha, beta, theta, phi, visibility=visibility)
    mean_bg = bg_rate_hz * T_s
    lam = mean_signal + mean_bg
    return rng.poisson(lam)

def measure_E(alpha, beta, theta, phi, pair_rate_hz, T_s, bg_rate_hz, rng, visibility=1.0):
    """
    Measure E(alpha,beta) via 4 coincidence settings (VV, HH, HV, VH),
    then compute the standard background-corrected estimator with -4C in denom.
    """
    # Four settings
    N_VV = measure_counts(alpha,           beta,           theta, phi, pair_rate_hz, T_s, bg_rate_hz, rng, visibility)
    N_HH = measure_counts(alpha + np.pi/2, beta + np.pi/2, theta, phi, pair_rate_hz, T_s, bg_rate_hz, rng, visibility)
    N_HV = measure_counts(alpha + np.pi/2, beta,           theta, phi, pair_rate_hz, T_s, bg_rate_hz, rng, visibility)
    N_VH = measure_counts(alpha,           beta + np.pi/2, theta, phi, pair_rate_hz, T_s, bg_rate_hz, rng, visibility)

    C = bg_rate_hz * T_s

    num = (N_VV + N_HH - N_HV - N_VH)
    den = (N_VV + N_HH + N_HV + N_VH - 4.0*C)

    # Guard against pathological cases if background is huge / time is tiny
    if den <= 0:
        return np.nan, {"VV": N_VV, "HH": N_HH, "HV": N_HV, "VH": N_VH, "C": C}

    E = num / den
    return E, {"VV": N_VV, "HH": N_HH, "HV": N_HV, "VH": N_VH, "C": C}

def measure_S(theta, phi,
              pair_rate_hz=5000.0, T_s=1.0, bg_rate_hz=20.0,
              a_deg=-45.0, ap_deg=0.0, b_deg=-22.5, bp_deg=+22.5,
              seed=None, visibility=1.0):
    """
    One CHSH run: measures 4 correlations, each requiring 4 coincidence settings => 16 Poisson samples.
    Returns (S, details_dict).
    """
    rng = np.random.default_rng(seed)

    a  = a_deg  * DEG
    ap = ap_deg * DEG
    b  = b_deg  * DEG
    bp = bp_deg * DEG

    E_ab,  d_ab  = measure_E(a,  b,  theta, phi, pair_rate_hz, T_s, bg_rate_hz, rng, visibility)
    E_abp, d_abp = measure_E(a,  bp, theta, phi, pair_rate_hz, T_s, bg_rate_hz, rng, visibility)
    E_apb, d_apb = measure_E(ap, b,  theta, phi, pair_rate_hz, T_s, bg_rate_hz, rng, visibility)
    E_apbp,d_apbp= measure_E(ap, bp, theta, phi, pair_rate_hz, T_s, bg_rate_hz, rng, visibility)

    S = E_ab - E_abp + E_apb + E_apbp

    details = {
        "E": {"ab": E_ab, "ab'": E_abp, "a'b": E_apb, "a'b'": E_apbp},
        "counts": {"ab": d_ab, "ab'": d_abp, "a'b": d_apb, "a'b'": d_apbp},
        "settings_deg": {"a": a_deg, "a'": ap_deg, "b": b_deg, "b'": bp_deg},
        "params": {"pair_rate_hz": pair_rate_hz, "T_s": T_s, "bg_rate_hz": bg_rate_hz, "visibility": visibility},
    }
    return S, details

def run_many(trials, theta, phi, **kwargs):
    Ss = []
    for k in range(trials):
        S, _ = measure_S(theta, phi, seed=None, **kwargs)
        Ss.append(S)
    Ss = np.array(Ss, dtype=float)
    return Ss

if __name__ == "__main__":
    # Example: Bell state theta=45°, phi=0
    theta = 45.0 * DEG
    phi = 0.0

    S1, details = measure_S(theta, phi, pair_rate_hz=5000, T_s=1.0, bg_rate_hz=20.0, visibility=1.0, seed=1)
    print("One-run S =", S1)
    print("E's =", details["E"])

    Ss = run_many(200, theta, phi, pair_rate_hz=5000, T_s=1.0, bg_rate_hz=20.0, visibility=1.0)
    print("Mean S over 200 runs =", np.nanmean(Ss))
    print("Std  S over 200 runs  =", np.nanstd(Ss))