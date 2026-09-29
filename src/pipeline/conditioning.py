"""Stage 2 - Condition. Position -> velocity, smoothing, QC."""
import numpy as np
import polars as pl


def velocity_from_xy(x, y, hz, win_s=0.4):
    """Schelling Formula 1: scaled mean of backward+forward displacement over +/-win_s/2."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    d = max(1, int(round((win_s / 2) * hz)))
    delta = d / hz
    back = np.hypot(x - np.roll(x, d), y - np.roll(y, d))
    fwd = np.hypot(np.roll(x, -d) - x, np.roll(y, -d) - y)
    v = (back + fwd) / 2 / delta
    v[:d] = np.nan; v[-d:] = np.nan
    return v


def qc_velocity(v, vmax=10.5):
    """Positional-jump guard: speeds above a physiological ceiling are GPS artefacts -> NaN."""
    v = np.asarray(v, float).copy()
    v[v > vmax] = np.nan
    return v


def acceleration(v, hz):
    return np.gradient(np.nan_to_num(v)) * hz


def smooth_ma(s, win):
    """Centred rolling mean (partial windows allowed at the edges)."""
    return pl.Series(s, dtype=pl.Float64).rolling_mean(win, min_samples=1, center=True).to_numpy()


def ewma(s, alpha):
    return pl.Series(s, dtype=pl.Float64).ewm_mean(alpha=alpha, adjust=False).to_numpy()


def qc_daily(df: pl.DataFrame) -> pl.Series:
    """Minimal QC hooks. Extend with the checks you already run:
    implausible session durations, effort-detection artefacts, sprint-threshold decisions.
    """
    return df["load_au"] < 0
