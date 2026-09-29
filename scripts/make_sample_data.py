"""Generate a synthetic squad of STATSports-style 10 Hz exports, so the whole pipeline
and dashboard can be run end to end without any real athlete data.

    python scripts/make_sample_data.py            # 8 weeks, 14 players  (~0.8 GB of CSV)
    python scripts/make_sample_data.py --weeks 6 --players 12

Writes (all git-ignored):
  sample_data/raw/YYYY-MM-DD-Sample-<Match|Training>.csv   one file per session, long format
  sample_data/reference/season_index.csv                   fixture context per match date
  sample_data/reference/player_categories.csv              position line / playing group per player

Everything here is invented: player names, fixtures, opposition, pitch location and the
movement itself. The generator is only meant to produce data with the right *shape* and
plausible magnitudes for Gaelic football (roughly 100-130 m/min in matches, repeated
high-speed efforts, a weekly training rhythm with a deload week), not to be a model of
real players.

Format notes (matches what src/pipeline/sources.py expects):
  * columns: Player Display Name, Date, Time, Lat, Lon, Speed (m/s), Filtered Speed,
    Heart Rate (bpm), Drill Title
  * every sample appears under Drill Title == "Entire Session"; drill / half windows are
    also exported, as in the real files. To keep the sample small, drill rows are written
    at 1 Hz (the pipeline only uses them for their start and end times).
"""
import argparse
import datetime as dt
import pathlib

import numpy as np
import polars as pl
from scipy.signal import lfilter

HZ = 10
ROOT = pathlib.Path(__file__).resolve().parents[1]
LAT0, LON0 = 53.40, -7.20              # an arbitrary field; only relative movement matters
PITCH_X, PITCH_Y = 140.0, 85.0

LINES = ["Full-back line", "Half-back line", "Midfield", "Half-forward line", "Full-forward line"]
GROUP = {"Full-back line": "Defence", "Half-back line": "Defence", "Midfield": "Middle third",
         "Half-forward line": "Attack", "Full-forward line": "Attack"}
# relative running intensity by line (midfield / half lines cover the most ground)
LINE_INT = {"Full-back line": 0.90, "Half-back line": 1.03, "Midfield": 1.10,
            "Half-forward line": 1.04, "Full-forward line": 0.92}
OPPOSITION = ["Northfield", "Riverside", "St Brendan's", "Glenmore", "Kilbride", "Ashgrove",
              "Carrowbeg", "Lisnagry", "Ballyvale", "Tullow Road"]


def make_squad(n, rng):
    first = ["Aidan", "Brian", "Cathal", "Declan", "Eoin", "Fergal", "Gerard", "Hugh", "Ciaran",
             "Jarlath", "Kevin", "Liam", "Mark", "Niall", "Oisin", "Padraig", "Ronan", "Sean"]
    squad = []
    for i in range(n):
        line = LINES[i % len(LINES)]
        squad.append({
            "name": f"{first[i % len(first)]} Sample{i + 1:02d}",
            "line": line,
            "group": GROUP[line],
            "intensity": LINE_INT[line] * rng.normal(1.0, 0.05),
            "vmax": rng.uniform(8.3, 9.4),
            "senior": i < int(n * 0.7),           # rest play reserve
        })
    return squad


def ou(n, mean, sd, tau_s, rng):
    """Ornstein-Uhlenbeck process sampled at HZ: a smooth, mean-reverting jog/run baseline."""
    a = np.exp(-1.0 / (tau_s * HZ))
    noise = rng.normal(0, sd * np.sqrt(1 - a * a), n)
    return mean + lfilter([1.0], [1.0, -a], noise)          # AR(1) recursion, vectorised


def add_efforts(v, rate_per_min, vmax, rng):
    """Superimpose short high-speed efforts (accelerate, hold, decelerate)."""
    n = len(v)
    k = rng.poisson(rate_per_min * n / HZ / 60)
    for start in rng.integers(0, max(1, n - 12 * HZ), k):
        peak = rng.uniform(0.55, 0.97) * vmax
        acc, hold, dec = (int(rng.uniform(a, b) * HZ) for a, b in [(1.2, 2.5), (0.5, 3.5), (1.0, 2.2)])
        shape = np.concatenate([np.linspace(0, 1, acc) ** 0.7, np.ones(hold), np.linspace(1, 0, dec) ** 1.3])
        seg = slice(start, min(n, start + len(shape)))
        v[seg] = np.maximum(v[seg], peak * shape[: seg.stop - seg.start])
    return v


def smooth(v, w=5):
    return np.convolve(v, np.ones(w) / w, mode="same")


def positions(speed, rng):
    """Integrate speed along a wandering heading, reflected inside the pitch -> lat/lon."""
    theta = np.cumsum(rng.normal(0, 0.06, len(speed)))
    x = np.cumsum(speed * np.cos(theta) / HZ) + PITCH_X / 2
    y = np.cumsum(speed * np.sin(theta) / HZ) + PITCH_Y / 2
    x = PITCH_X - np.abs(np.mod(x, 2 * PITCH_X) - PITCH_X)
    y = PITCH_Y - np.abs(np.mod(y, 2 * PITCH_Y) - PITCH_Y)
    lat = LAT0 + (y - PITCH_Y / 2) / 110_540.0
    lon = LON0 + (x - PITCH_X / 2) / (111_320.0 * np.cos(np.radians(LAT0)))
    return lat, lon


def clock(sec):
    h = (sec // 3600).astype(int)
    m = ((sec % 3600) // 60).astype(int)
    s = sec % 60
    return [f"{a:02d}:{b:02d}:{c:04.1f}" for a, b, c in zip(h, m, s)]


def session_plan(kind, rng):
    """List of (drill title, minutes, intensity multiplier, efforts per minute); None = rest."""
    if kind == "Match":
        return [("1st Half", 35, 1.0, 0.85), (None, 12, 0, 0), ("2nd Half", 35, 0.96, 0.8)]
    plan = [("Warm Up", 15, 0.55, 0.15), (None, 3, 0, 0),
            ("Possession Game", int(rng.integers(14, 22)), 0.95, 1.1), (None, 4, 0, 0),
            ("Tactical", int(rng.integers(12, 20)), 0.65, 0.4), (None, 3, 0, 0)]
    if rng.random() < 0.7:
        plan += [("Conditioning", int(rng.integers(8, 14)), 1.15, 1.8)]
    return plan


def player_session(p, kind, date, t0, load_scale, rng, minutes_played=None):
    plan = session_plan(kind, rng)
    parts_v, drill_rows, t = [], [], 0.0
    for title, mins, inten, eff in plan:
        n = int(mins * 60 * HZ)
        if title is None:                                   # rest: slow walking
            v = np.clip(ou(n, 0.6, 0.25, 8, rng), 0, 2)
        else:
            mean = 1.85 * inten * p["intensity"] * load_scale
            v = np.clip(ou(n, mean, 0.75, 6, rng), 0, None)
            v = add_efforts(v, eff * p["intensity"] * load_scale, p["vmax"], rng)
            drill_rows.append((title, t, t + n / HZ))
        parts_v.append(v)
        t += n / HZ
    v = np.clip(np.concatenate(parts_v), 0, p["vmax"])
    if minutes_played is not None:                          # substitutions: trim the match
        v = v[: int(minutes_played * 60 * HZ)]
        drill_rows = [(d, a, min(b, len(v) / HZ)) for d, a, b in drill_rows if a < len(v) / HZ]
    fv = smooth(v)
    raw = np.clip(v + rng.normal(0, 0.12, len(v)), 0, None)
    lat, lon = positions(fv, rng)
    sec = t0 + np.arange(len(v)) / HZ
    hr = np.clip(95 + 22 * smooth(fv, 300) + rng.normal(0, 2, len(v)), 60, 200).round()
    whole = pl.DataFrame({
        "Player Display Name": p["name"], "Date": date, "Time": clock(sec),
        "Lat": lat.round(7), "Lon": lon.round(7), "Speed (m/s)": raw.round(2),
        "Filtered Speed": fv.round(2), "Heart Rate (bpm)": hr, "Drill Title": "Entire Session"})
    drills = [whole.slice(int(a * HZ), int((b - a) * HZ)).gather_every(HZ).with_columns(pl.lit(d).alias("Drill Title"))
              for d, a, b in drill_rows]
    return pl.concat([whole, *drills])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=8)
    ap.add_argument("--players", type=int, default=14)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--start", default="2026-03-02", help="a Monday")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    raw = ROOT / "sample_data" / "raw"
    ref = ROOT / "sample_data" / "reference"
    raw.mkdir(parents=True, exist_ok=True)
    ref.mkdir(parents=True, exist_ok=True)
    for old in raw.glob("*.csv"):
        old.unlink()

    squad = make_squad(args.players, rng)
    start = dt.date.fromisoformat(args.start)
    block = [0.94, 1.00, 1.08, 0.82]                       # three build weeks, one deload
    fixtures = []
    for w in range(args.weeks):
        scale = block[w % 4] * (1 + 0.04 * w / args.weeks)
        week = start + dt.timedelta(weeks=w)
        sessions = [(week + dt.timedelta(days=1), "Training", 19.5 * 3600),   # Tue 19:30
                    (week + dt.timedelta(days=3), "Training", 19.5 * 3600)]   # Thu 19:30
        grade = "Senior" if w % 3 != 2 else "Reserve"
        sessions.append((week + dt.timedelta(days=6 if grade == "Senior" else 5), "Match", 14.0 * 3600))
        for day, kind, t0 in sessions:
            date = day.isoformat()
            frames = []
            if kind == "Match":
                pool = [p for p in squad if p["senior"] == (grade == "Senior")]
                pool += [p for p in squad if p["senior"] != (grade == "Senior")][:3]   # a few dual-grade players
                opp = OPPOSITION[w % len(OPPOSITION)]
                fixtures.append({"Date": date, "Competition": "League" if w < args.weeks - 2 else "Championship",
                                 "Opposition": opp, "Grade": grade})
                for i, p in enumerate(pool):
                    mins = None if i < len(pool) - 3 else float(rng.uniform(30, 70))  # last three = subs used
                    frames.append(player_session(p, kind, date, t0, scale, rng, mins))
            else:
                for p in squad:
                    if rng.random() < 0.9:                  # occasional missed session
                        frames.append(player_session(p, kind, date, t0, scale, rng))
            out = raw / f"{date}-Sample-{kind}.csv"
            pl.concat(frames).write_csv(out)
            print(f"  wrote {out.name}  ({out.stat().st_size / 1e6:.0f} MB)", flush=True)

    pl.DataFrame(fixtures).write_csv(ref / "season_index.csv")
    pl.DataFrame({"player": [p["name"] for p in squad],
                  "position_line": [p["line"] for p in squad],
                  "category": ["Senior" if p["senior"] else "Reserve" for p in squad],
                  "new_rule_category": [p["group"] for p in squad],
                  "specific_position": [p["line"].replace(" line", "") for p in squad]}
                 ).write_csv(ref / "player_categories.csv")
    print(f"[ok] {args.weeks} weeks, {args.players} players -> {raw.relative_to(ROOT)}/ and {ref.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
