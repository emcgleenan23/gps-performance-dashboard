import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

import numpy as np
import polars as pl
from datetime import datetime, timezone

from pipeline.config import load_config
from pipeline import sources, conditioning as cond, features as feat, patterns as pat, output as out


def main():
    cfg = load_config()
    hz = cfg["sampling"]["analysis_hz"]
    real = cfg["data"]["source"] == "statsports"

    # 1 - collect (first real run scans every raw file once into cache/bins/)
    daily = sources.load_daily_load(cfg)
    try:
        track = sources.load_match_track(cfg)
    except FileNotFoundError as e:                  # drive not mounted: single-player xy trace unavailable
        print(f"[warn] match track skipped ({e})")
        track = None

    # 2 - condition
    load = daily["load_au"].to_numpy()
    daily = daily.with_columns(
        load_ma7=pl.Series(cond.smooth_ma(load, cfg["conditioning"]["daily_smooth_days"])),
        load_ewma=pl.Series(cond.ewma(load, cfg["conditioning"]["daily_ewma_alpha"])),
    )
    if track is not None:
        v = cond.velocity_from_xy(track["x"].to_numpy(), track["y"].to_numpy(), hz,
                                  cfg["conditioning"]["velocity_window_s"])
        v = cond.qc_velocity(v, cfg["conditioning"].get("vmax_ms", 10.5))
        per_min = feat.per_minute_distance(v, hz)
    else:
        per_min = np.zeros(max(cfg["mds"]["windows_min"]))

    # 3 - features
    daily = daily.with_columns(acwr=pl.Series(feat.acwr(load, cfg["features"]["acwr_acute_days"],
                                                         cfg["features"]["acwr_chronic_days"])))

    # 4 - patterns
    periods = cfg["periodisation"]["periods_days"]
    cwt = pat.morlet_cwt(load, periods, cfg["periodisation"]["wavelet_w0"])
    psd_periods = list(range(2, 61))
    psd = pat.periodogram(load, psd_periods)
    mds = pat.duration_specific_mds(per_min, cfg["mds"]["windows_min"])
    cps = pat.changepoints(daily["load_ma7"].to_numpy(), min_size=14, penalty=cfg["mds"]["changepoint_penalty"])

    cols = 26
    step = max(1, len(daily) // cols)
    grid = [[float(cwt[r, min(c * step + step // 2, len(daily) - 1)]) for c in range(cols)]
            for r in range(len(periods))]

    # 5 - output
    tables = {
        "daily_features": daily,
        "match_per_min": pl.DataFrame({"minute": np.arange(len(per_min)), "m_per_min": per_min}),
    }
    if real:
        tables["session_summary"] = sources.session_table(cfg)   # pseudonymised
    dash = {
        "meta": {"source": cfg["data"]["source"], "generated": datetime.now(timezone.utc).isoformat(),
                 **sources.SOURCE_META},
        "daily": {
            "dates": daily["date"].cast(pl.String).to_list(),
            "load": daily["load_au"].round(0).to_list(),
            "ma7": daily["load_ma7"].round(1).to_list(),
            "ewma": daily["load_ewma"].round(1).to_list(),
            "acwr": daily["acwr"].to_list(),
        },
        "match": {"per_min": [round(float(x), 1) for x in per_min]},
        "mds": mds,
        "psd": {"periods": psd_periods, "power": psd.round(2).tolist()},
        "wavelet": {"periods": periods, "grid": grid},
        "changepoints": cps,
    }
    if real:
        dash["squad"] = sources.dashboard_payload(cfg)            # per-player, pseudonymised
        key = sources._athlete_key(sources.session_summary(cfg, verbose=False)["player"].to_list())
        sb = sources.session_bins_payload(cfg, key)
        p = out.write_js(cfg, "session_bins.js", "SESSION_BINS", sb)
        print(f"[ok] {p.name}: {len(sb['sessions'])} sessions ({p.stat().st_size / 1e6:.1f} MB)")
    out_dir = out.write_outputs(cfg, tables, dash)

    print(f"[ok] source={cfg['data']['source']}  days={len(daily)}  match_min={len(per_min)}")
    print(f"[ok] ACWR latest={daily['acwr'][-1]:.2f}   "
          f"peak 1-min={mds[1]['peak']} m/min   peak 3-min={mds[3]['peak']} m/min")
    print(f"[ok] changepoints at days={cps}")
    print(f"[ok] wrote {out_dir}/  (daily_features.csv, match_per_min.csv, dashboard_data.js)")


if __name__ == "__main__":
    main()
