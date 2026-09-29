"""Stage 1 - Collect. Turn raw exports into tidy frames (Polars).

Two data levels:
  * daily load   : one row per athlete per day (load_au)  -> periodisation + ACWR
  * match track  : high-frequency x,y per athlete         -> velocity/accel/MDS

Raw STATSports files are read ONCE each with a lazy Polars scan (projection + predicate
pushdown, multi-threaded) into a 10-second bin cache (cache/bins/<file>.parquet). Every
other product - session summaries, per-minute match traces, TAL, the session-page payload -
is derived from that cache, so re-runs never touch the drive unless a file is new or changed.
"""
import json
import pathlib
import re

import numpy as np
import polars as pl


def load_daily_load(cfg):
    src = cfg["data"]["source"]
    if src == "synthetic":
        return _synthetic_daily()
    if src == "statsports":
        return _statsports_daily(cfg)
    raise ValueError(f"unknown data.source: {src}")


def load_match_track(cfg):
    src = cfg["data"]["source"]
    if src == "synthetic":
        return _synthetic_track(cfg)
    if src == "statsports":
        return _statsports_track(cfg)
    raise ValueError(f"unknown data.source: {src}")


# ----------------------------- synthetic --------------------------------------
def _synthetic_daily(seed=101, weeks=26):
    rng = np.random.default_rng(seed)
    N = weeks * 7
    wf = np.array([0.22, 0.10, 1.00, 0.85, 0.60, 0.28, 1.28])  # Sun..Sat, Sat=match
    block = np.array([0.92, 1.00, 1.12, 0.60])                 # 3 build + 1 deload
    d = np.arange(N)
    dow, wk = d % 7, d // 7
    base = np.where(dow == 6, 790.0, 640.0 * wf[dow])
    load = base * block[wk % 4] * (1 + 0.06 * wk / weeks) + rng.normal(0, 16, N)
    dates = pl.date_range(pl.date(2025, 12, 1), pl.date(2025, 12, 1) + pl.duration(days=N - 1), "1d", eager=True)
    return pl.DataFrame({"athlete": ["A01"] * N, "date": dates, "load_au": np.clip(np.round(load), 0, None)})


def _reflect(a, lo, hi):
    span = hi - lo
    m = np.mod(a - lo, 2 * span)
    return lo + np.where(m > span, 2 * span - m, m)


def _synthetic_track(cfg, seed=7):
    hz = cfg["sampling"]["analysis_hz"]
    n = 70 * 60 * hz
    dt = 1.0 / hz
    rng = np.random.default_rng(seed)
    t = np.arange(n) / hz
    speed = 2.6 + 1.1 * np.sin(t / 22) + 0.6 * np.sin(t / 6) + rng.normal(0, 0.5, n)
    speed = np.where((t > 2500) & (t < 2680), speed + 1.7, speed)   # a hard passage
    speed = np.clip(speed, 0, 9.2)
    theta = np.cumsum(rng.normal(0, 0.15, n))
    x = _reflect(np.cumsum(speed * np.cos(theta) * dt), 0, 140)
    y = _reflect(np.cumsum(speed * np.sin(theta) * dt), 0, 85)
    return pl.DataFrame({"athlete": ["A01"] * n, "t_s": t, "x": x, "y": y})


# ----------------------------- STATSports -------------------------------------
# Export format (one CSV per session, 10 Hz, long format, one row per sample):
#   Player Display Name, Date, Time, Lat, Lon, Speed (m/s), Filtered Speed,
#   Heart Rate (bpm), Drill Title, Accl X/Y/Z, Gyro X/Y/Z
# File names: YYYY-MM-DD-<Team>-<Match|Training>.csv
# IMPORTANT: every drill is exported twice - once under its own Drill Title and
# once inside "Entire Session". We keep "Entire Session" only, otherwise load is
# double-counted. Drill rows are used only for their time windows (halves, drills).
_FNAME = re.compile(r"(\d{4}-\d{2}-\d{2})-(.+?)-(Match|Training)\.csv$", re.I)
_WHOLE = "Entire Session"
_VMAX = 10.5          # m/s ceiling - anything above is a GPS artefact
_HSR, _SPRINT = 5.5, 7.0
BIN_S = 10            # seconds per bin in the cache
_SCHEMA = {"Player Display Name": pl.String, "Date": pl.String, "Time": pl.String,
           "Lat": pl.Float64, "Lon": pl.Float64, "Filtered Speed": pl.Float64, "Drill Title": pl.String}

SOURCE_META = {}      # filled by the readers, merged into dashboard meta by run.py


def _cache_dir():
    d = pathlib.Path("cache")
    d.mkdir(exist_ok=True)
    return d


def _bins_dir():
    d = _cache_dir() / "bins"
    d.mkdir(exist_ok=True)
    return d


def _session_files(cfg):
    root = pathlib.Path(cfg["data"]["root"]).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"data.root not found: {root}  (is the drive mounted?)")
    files = []
    for p in sorted(root.glob("*.csv")):
        m = _FNAME.search(p.name)
        if m:
            st = p.stat()
            files.append({"file": p.name, "path": p, "date": m.group(1), "team": m.group(2),
                          "type": m.group(3).title(), "mtime": int(st.st_mtime), "size": st.st_size})
    if not files:
        raise FileNotFoundError(f"no YYYY-MM-DD-<team>-<Match|Training>.csv files in {root}")
    return files


def _cached_files(cfg):
    """Session files from the drive if mounted, otherwise from the bin cache (so the
    dashboard can be rebuilt from cache without the drive)."""
    try:
        return _session_files(cfg)
    except FileNotFoundError:
        out = []
        for jf in sorted(_bins_dir().glob("*.json")):
            meta = json.loads(jf.read_text())
            m = _FNAME.search(jf.stem + ".csv")
            if m and (jf.with_suffix(".parquet")).exists():
                out.append({"file": jf.stem + ".csv", "path": None, "date": m.group(1), "team": m.group(2),
                            "type": m.group(3).title(), "mtime": meta["key"][0], "size": meta["key"][1]})
        if not out:
            raise
        print("  [info] data drive not mounted - using cache/bins only")
        return out


def _scan(path):
    """Lazy scan of one raw export, only the columns we use."""
    return pl.scan_csv(path, schema_overrides=_SCHEMA).select(list(_SCHEMA))


_CLOCK = (pl.col("Time").str.slice(0, 2).cast(pl.Int32) * 3600
          + pl.col("Time").str.slice(3, 2).cast(pl.Int32) * 60
          + pl.col("Time").str.slice(6).cast(pl.Float64))


# ----------------------------- the one pass over each raw file -----------------
# Per player, per 10 s of clock time: samples, distance, HSR, sprint distance, TAL, max speed.
# TAL follows Schelling et al. (2025, Appl. Sci. 15:10575): acceleration = difference between
# consecutive velocity samples; TAL = sum of |dv| (m/s). Computed at 10 Hz from Filtered Speed,
# only between consecutive samples <= 0.25 s apart (no bridging across dropouts).
def _bin_file(path, hz):
    dt = 1.0 / hz
    lf = _scan(path).with_columns(sec=_CLOCK)
    drills_q = lf.group_by("Drill Title").agg(pl.col("sec").min().alias("a"), pl.col("sec").max().alias("b"))
    bins_q = (lf.filter(pl.col("Drill Title") == _WHOLE)
              .select(pl.col("Player Display Name").alias("player"), "sec",
                      pl.col("Filtered Speed").clip(0, _VMAX).alias("v"))
              .sort(["player", "sec"])
              .with_columns(
                  dv=pl.when((pl.col("sec") - pl.col("sec").shift(1).over("player")) <= 0.25)
                       .then((pl.col("v") - pl.col("v").shift(1).over("player")).abs())
                       .otherwise(0.0),
                  bin=(pl.col("sec") // BIN_S).cast(pl.Int64))
              .group_by(["player", "bin"])
              .agg(n=pl.len(),
                   td=pl.col("v").sum() * dt,
                   hsr=pl.col("v").filter(pl.col("v") >= _HSR).sum() * dt,
                   spr=pl.col("v").filter(pl.col("v") >= _SPRINT).sum() * dt,
                   tal=pl.col("dv").sum(),
                   vmax=pl.col("v").max())
              .sort(["player", "bin"]))
    bins, drills = pl.collect_all([bins_q, drills_q])     # one scan, shared by both queries
    drills = {r["Drill Title"]: [float(r["a"]), float(r["b"])] for r in drills.iter_rows(named=True)}
    return bins, drills


def session_bins(cfg, verbose=True):
    """Build / reuse cache/bins/<file>.parquet (+ .json: file key + drill/half windows)."""
    hz = cfg["sampling"]["input_hz"]
    d = _bins_dir()
    out = {}
    for f in _cached_files(cfg):
        stem = f["file"][:-4]
        pf, jf = d / f"{stem}.parquet", d / f"{stem}.json"
        fresh = pf.exists() and jf.exists() and json.loads(jf.read_text()).get("key") == [f["mtime"], f["size"]]
        if not fresh:
            if f["path"] is None:
                continue
            if verbose:
                print(f"  scanning {f['file']} ({f['size'] / 1e6:.0f} MB)", flush=True)
            bins, drills = _bin_file(f["path"], hz)
            bins.write_parquet(pf)
            jf.write_text(json.dumps({"key": [f["mtime"], f["size"]], "drills": drills}))
        out[f["file"]] = f
    return out


def _read_bins(fname):
    return pl.read_parquet(_bins_dir() / f"{fname[:-4]}.parquet")


def _read_drills(fname):
    return json.loads((_bins_dir() / f"{fname[:-4]}.json").read_text())["drills"]


# ----------------------------- products derived from the bin cache ------------
def session_summary(cfg, verbose=True):
    """One row per player per session (Entire Session), derived from the bin cache."""
    dt = 1.0 / cfg["sampling"]["input_hz"]
    files = session_bins(cfg, verbose=verbose)
    frames = []
    for fname, f in files.items():
        frames.append(_read_bins(fname).group_by("player").agg(
            pl.col("n").sum(), pl.col("td").sum(), pl.col("hsr").sum(), pl.col("spr").sum(),
            pl.col("tal").sum(), pl.col("vmax").max())
            .with_columns(file=pl.lit(fname), date=pl.lit(f["date"]).str.to_date(), type=pl.lit(f["type"])))
    s = pl.concat(frames)
    return (s.select("player", "file", "date", "type",
                     (pl.col("n") * dt / 60).round(2).alias("duration_min"),
                     pl.col("td").round(1).alias("distance_m"), pl.col("hsr").round(1).alias("hsr_m"),
                     pl.col("spr").round(1).alias("sprint_m"), pl.col("tal").round(1).alias("tal_ms"),
                     pl.col("vmax").round(2).alias("vmax_ms"))
             .sort(["date", "player"]))


def _athlete_key(names):
    """Stable pseudonyms P01, P02... (alphabetical). The key stays in cache/ (git-ignored)."""
    names = sorted(set(names))
    key = pl.DataFrame({"player": names, "athlete": [f"P{i + 1:02d}" for i in range(len(names))]})
    key.write_csv(_cache_dir() / "athlete_key.csv")
    return dict(zip(key["player"], key["athlete"]))


def _resolve_athlete(cfg, key):
    """data.athlete: 'squad' | pseudonym ('P03') | display name. Returns (name or None, label)."""
    want = str(cfg["data"].get("athlete", "squad"))
    if want.lower() == "squad":
        return None, "Squad average"
    inv = {v: k for k, v in key.items()}
    if want in inv:
        return inv[want], want
    if want in key:
        return want, key[want]
    raise ValueError(f"data.athlete '{want}' not found (use 'squad', a P-code, or a display name)")


def session_table(cfg):
    """Pseudonymised per-player session table for outputs/."""
    s = session_summary(cfg, verbose=False)
    key = _athlete_key(s["player"].to_list())
    return (s.with_columns(athlete=pl.col("player").replace_strict(key))
             .select("athlete", "date", "type", "duration_min", "distance_m", "hsr_m", "sprint_m", "tal_ms", "vmax_ms"))


def _statsports_daily(cfg):
    """Tidy daily load: columns = ["athlete", "date", "load_au"].
    load_au = total distance (m) from STATSports Filtered Speed, 'Entire Session' only.
    Multiple sessions in a day are summed; non-session days are 0.
    data.athlete = 'squad' -> mean of players who recorded that day."""
    s = session_summary(cfg)
    key = _athlete_key(s["player"].to_list())
    name, label = _resolve_athlete(cfg, key)
    per = s.group_by(["player", "date"]).agg(pl.col("distance_m").sum())
    if name is None:
        day = per.group_by("date").agg(pl.col("distance_m").mean())
    else:
        day = per.filter(pl.col("player") == name).select("date", "distance_m")
        if day.is_empty():
            raise ValueError(f"no sessions for athlete {label}")
    cal = pl.DataFrame({"date": pl.date_range(s["date"].min(), s["date"].max(), "1d", eager=True)})
    daily = (cal.join(day, on="date", how="left")
                .select(pl.lit(label).alias("athlete"), "date",
                        pl.col("distance_m").fill_null(0.0).round(0).alias("load_au")))
    matches = s.filter(pl.col("type") == "Match")
    SOURCE_META.update({
        "athlete": label, "n_players": s["player"].n_unique(), "n_sessions": s["file"].n_unique(),
        "n_matches": matches["file"].n_unique(),
        "load_metric": "Total distance", "load_unit": "m",
        "range": [str(cal["date"][0]), str(cal["date"][-1])],
        "match_dates": sorted({str(d) for d in matches["date"]}),
    })
    return daily


def _statsports_track(cfg):
    """One player's match trace: columns = ["athlete", "t_s", "x", "y"] on a regular
    analysis_hz grid (gaps left as NaN). data.match = 'latest' or 'YYYY-MM-DD'.
    In squad mode the player with the longest recording in that match is used.
    Needs the raw file (drive mounted)."""
    hz = cfg["sampling"]["analysis_hz"]
    s = session_summary(cfg, verbose=False)
    key = _athlete_key(s["player"].to_list())
    name, label = _resolve_athlete(cfg, key)
    m = s.filter(pl.col("type") == "Match")
    want = str(cfg["data"].get("match", "latest"))
    mdate = m["date"].max() if want == "latest" else pl.Series([want]).str.to_date()[0]
    m = m.filter(pl.col("date") == mdate)
    if m.is_empty():
        raise ValueError(f"no match export for data.match = {want}")
    if name is None:
        name = m.sort("duration_min")["player"][-1]
    elif name not in set(m["player"]):
        raise ValueError(f"{label} has no data in the {mdate} match")
    fname = m.filter(pl.col("player") == name)["file"][0]
    path = pathlib.Path(cfg["data"]["root"]).expanduser() / fname
    tr = (_scan(path)
          .filter((pl.col("Drill Title") == _WHOLE) & (pl.col("Player Display Name") == name))
          .with_columns(sec=_CLOCK)
          .filter((pl.col("Lat").abs() > 1e-6) & (pl.col("Lon").abs() > 1e-6))    # drop 0,0 dropouts
          .unique("sec", keep="first").sort("sec")
          .collect())
    lat0, lon0 = tr["Lat"].median(), tr["Lon"].median()
    x = ((tr["Lon"] - lon0) * np.cos(np.radians(lat0)) * 111_320.0).to_numpy()   # local equirectangular, m
    y = ((tr["Lat"] - lat0) * 110_540.0).to_numpy()
    sec = tr["sec"].to_numpy()
    ok = np.hypot(x, y) < 500                                                    # > 500 m from pitch = artefact
    x, y, sec = x[ok], y[ok], sec[ok]
    k = np.round((sec - sec[0]) * hz).astype(int)
    k, first = np.unique(k, return_index=True)
    gx, gy = np.full(k.max() + 1, np.nan), np.full(k.max() + 1, np.nan)
    gx[k], gy[k] = x[first], y[first]
    SOURCE_META.update({"match_date": str(mdate), "match_athlete": key[name],
                        "match_minutes": round(len(gx) / hz / 60, 1)})
    return pl.DataFrame({"athlete": [key[name]] * len(gx), "t_s": np.arange(len(gx)) / hz, "x": gx, "y": gy})


# ----------------------------- dashboard payloads ------------------------------
def match_minutes(cfg):
    """Per-player distance per clock minute for every MATCH (from the bin cache)."""
    frames = []
    for fname, f in session_bins(cfg, verbose=False).items():
        if f["type"] != "Match":
            continue
        frames.append(_read_bins(fname)
                      .group_by("player", (pl.col("bin") * BIN_S // 60).alias("clock_min"))
                      .agg(pl.col("td").sum().round(1).alias("m"))
                      .with_columns(date=pl.lit(f["date"])))
    return pl.concat(frames) if frames else pl.DataFrame()


def dashboard_payload(cfg):
    """Pseudonymised, compact data so the dashboard can filter by date / player / match in the browser."""
    s = session_summary(cfg, verbose=False)
    key = _athlete_key(s["player"].to_list())
    ids = sorted(key.values())
    idx = {a: i for i, a in enumerate(ids)}
    s = s.with_columns(a=pl.col("player").replace_strict(key).replace_strict(idx), d=pl.col("date").cast(pl.String))
    sessions = {
        "a": s["a"].to_list(), "d": s["d"].to_list(), "t": s["type"].str.slice(0, 1).to_list(),
        "dist": s["distance_m"].round(0).cast(pl.Int64).to_list(), "hsr": s["hsr_m"].round(0).cast(pl.Int64).to_list(),
        "spr": s["sprint_m"].round(0).cast(pl.Int64).to_list(), "tal": s["tal_ms"].round(0).cast(pl.Int64).to_list(),
        "vmax": s["vmax_ms"].round(2).to_list(), "dur": s["duration_min"].round(1).to_list(),
    }
    matches = {}
    mm = match_minutes(cfg)
    for (date,), g in mm.group_by("date", maintain_order=True):
        start, end = int(g["clock_min"].min()), int(g["clock_min"].max())
        players = {}
        for (pl_name,), gp in g.group_by("player"):
            arr = [None] * (end - start + 1)
            for cm, m in zip(gp["clock_min"], gp["m"]):
                arr[int(cm) - start] = round(float(m))
            players[key[pl_name]] = arr
        matches[str(date)] = {"start_clock": f"{start // 60:02d}:{start % 60:02d}", "players": players}
    ref = reference_payload(cfg, key)
    days = set(sessions["d"])
    return {"athletes": ids, "sessions": sessions, "matches": dict(sorted(matches.items())),
            "fixtures": {d: f for d, f in ref["fixtures"].items() if d in days},
            "positions": ref["positions"], "unmatched_positions": ref["unmatched"]}


def session_bins_payload(cfg, key):
    """Compact per-session, per-player 10 s series for the session page (P-codes only)."""
    sessions = []
    files = session_bins(cfg, verbose=False)
    for fname, f in sorted(files.items(), key=lambda kv: (kv[1]["date"], kv[1]["type"])):
        b = _read_bins(fname)
        if b.is_empty():
            continue
        b0, b1 = int(b["bin"].min()), int(b["bin"].max())
        n = b1 - b0 + 1
        players = {}
        for (pl_name,), gp in b.group_by("player"):
            if pl_name not in key:
                continue
            idx = (gp["bin"] - b0).to_numpy()
            def arr(col, scale):
                a = np.full(n, -1, dtype=np.int64)
                a[idx] = np.round(gp[col].to_numpy() * scale).astype(np.int64)
                return a.tolist()
            players[key[pl_name]] = {"td": arr("td", 1), "hsr": arr("hsr", 1), "spr": arr("spr", 1),
                                     "tal": arr("tal", 10), "vmax": arr("vmax", 100)}
        # drill / half windows as bin offsets (Entire Session is implicit)
        dr = {k: [int(a // BIN_S) - b0, int(z // BIN_S) - b0] for k, (a, z) in _read_drills(fname).items() if k != _WHOLE}
        sessions.append({"d": f["date"], "type": f["type"], "start": b0 * BIN_S, "n": n,
                         "drills": dr, "players": dict(sorted(players.items()))})
    return {"bin_s": BIN_S, "scale": {"td": 1, "hsr": 1, "spr": 1, "tal": 10, "vmax": 100}, "sessions": sessions}


# ----------------------------- reference data (fixtures, positions) -----------
def _read_ref(cfg, key):
    p = (cfg.get("reference") or {}).get(key)
    if p and pathlib.Path(p).exists():
        return pl.read_csv(p, infer_schema_length=0)     # all strings; blanks -> null
    return None


def reference_payload(cfg, key):
    """Fixture context per match date and position per P-code (names never leave this function)."""
    out = {"fixtures": {}, "positions": {}, "unmatched": 0}
    fx = _read_ref(cfg, "season_index")
    if fx is not None:
        for r in fx.iter_rows(named=True):
            out["fixtures"][r["Date"]] = {"grade": r["Grade"] or "", "comp": r["Competition"] or "", "opp": r["Opposition"] or ""}
    info = {}
    pos = _read_ref(cfg, "positions")
    if pos is not None:
        for name, line in pos.select(pos.columns[:2]).iter_rows():
            if line:
                info[name] = {"line": line}
    pc = _read_ref(cfg, "player_categories")
    if pc is not None:
        for r in pc.iter_rows(named=True):
            d = info.setdefault(r["player"], {})
            d.update({k: v for k, v in {"line": r["position_line"], "cat": r["category"],
                                         "group": r["new_rule_category"], "role": r["specific_position"]}.items() if v})
    for name, pid in key.items():
        if name in info:
            out["positions"][pid] = info[name]
        else:
            out["unmatched"] += 1
    return out
