"""Stage 4 - Pattern recognition & forecasting."""
import numpy as np


def morlet_cwt(x, periods, w0=6):
    """Continuous wavelet transform (magnitude scalogram). Rows = periods, cols = time."""
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    n = len(x)
    conv = (w0 + np.sqrt(2 + w0 ** 2)) / (4 * np.pi)   # period = scale / conv
    out = np.zeros((len(periods), n))
    for i, p in enumerate(periods):
        s = p * conv
        half = min(int(np.ceil(4 * s)), (n - 1) // 2)
        k = np.arange(-half, half + 1)
        psi = np.exp(-(k / s) ** 2 / 2) * np.pi ** -0.25 * np.exp(1j * w0 * (k / s))
        coef = np.convolve(x, np.conj(psi)[::-1], mode="same") / np.sqrt(s)
        out[i] = np.abs(coef)
    return out


def periodogram(x, periods):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x); t = np.arange(n)
    out = []
    for p in periods:
        w = 2 * np.pi / p
        c = np.sum(x * np.cos(w * t)); s = np.sum(x * np.sin(w * t))
        out.append((c * c + s * s) / n)
    return np.array(out)


def duration_specific_mds(per_min, windows):
    """Worst-case rolling mean (m/min) for each window length in minutes."""
    out = {}
    for w in windows:
        best, at = -1.0, 0
        for i in range(len(per_min) - w + 1):
            m = float(np.mean(per_min[i:i + w]))
            if m > best:
                best, at = m, i
        out[int(w)] = {"peak": round(best, 1), "start_min": int(at)}
    return out


def changepoints(x, min_size=14, penalty=0.05, max_k=5):
    """PELT via `ruptures` if available (Schelling-aligned); else NumPy binary segmentation."""
    x = np.asarray(x, float)
    try:
        import ruptures as rpt
        beta = penalty * float(((x - x.mean()) ** 2).sum())
        bkps = rpt.Pelt(model="l2", min_size=min_size).fit(x).predict(pen=beta)
        return [int(b) for b in bkps[:-1]]
    except Exception:
        return _binseg(x, min_size, penalty, max_k)


def _cost(a):
    a = np.asarray(a, float)
    return float(((a - a.mean()) ** 2).sum())


def _binseg(x, min_size, penalty, max_k):
    cps, total = [], _cost(x)

    def rec(lo, hi):
        if hi - lo < 2 * min_size or len(cps) >= max_k:
            return
        best, bi = np.inf, -1
        for i in range(lo + min_size, hi - min_size + 1):
            c = _cost(x[lo:i]) + _cost(x[i:hi])
            if c < best:
                best, bi = c, i
        if bi > 0 and (_cost(x[lo:hi]) - best) > penalty * total:
            cps.append(bi); rec(lo, bi); rec(bi, hi)

    rec(0, len(x))
    return sorted(cps)


def holt_winters(y, m=7, alpha=0.3, beta=0.05, gamma=0.3, h=28):
    """Additive Holt-Winters (illustrative stand-in for ARIMA). Returns forecast + 95% band."""
    y = np.asarray(y, float); n = len(y)
    level = y[:m].mean(); trend = (y[m:2 * m].mean() - y[:m].mean()) / m
    seas = list(y[:m] - level); resid = []
    for t in range(n):
        s = seas[t % m]; resid.append(y[t] - (level + trend + s)); lp = level
        level = alpha * (y[t] - s) + (1 - alpha) * (level + trend)
        trend = beta * (level - lp) + (1 - beta) * trend
        seas[t % m] = gamma * (y[t] - level) + (1 - gamma) * s
    sd = float(np.sqrt(np.mean(np.square(resid))))
    fc = np.array([level + k * trend + seas[(n + k - 1) % m] for k in range(1, h + 1)])
    band = 1.96 * sd * np.sqrt(np.arange(1, h + 1))
    return {"fc": np.clip(fc, 0, None), "lo": np.clip(fc - band, 0, None), "hi": fc + band}
