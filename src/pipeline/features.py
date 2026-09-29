"""Stage 3 - Features & integration."""
import numpy as np
import polars as pl


def per_minute_distance(v, hz):
    """Distance covered in each 1-minute bin (m/min) from conditioned velocity."""
    step = np.nan_to_num(v) * (1.0 / hz)
    per_min_n = int(len(v) // (60 * hz))
    return np.array([step[i * 60 * hz:(i + 1) * 60 * hz].sum() for i in range(per_min_n)])


def acwr(daily_load, acute=7, chronic=28):
    """Rolling-average acute:chronic ratio (1.0 where the chronic mean is 0)."""
    s = pl.Series(daily_load, dtype=pl.Float64)
    a = s.rolling_mean(acute, min_samples=1)
    c = s.rolling_mean(chronic, min_samples=1)
    r = pl.select(pl.when(c > 0).then(a / c).otherwise(1.0)).to_series()
    return r.round(3).to_numpy()
