# GPS performance pipeline & dashboard

An end-to-end training-load pipeline and interactive dashboard for team-sport GPS data, built
around Gaelic football and STATSports 10 Hz exports. Raw per-sample files go in; a
pseudonymised, browser-only dashboard comes out.

**[Live demo →](https://emcgleenan23.github.io/gps-performance-dashboard/)** (synthetic squad, no real athlete data)

| Session analysis | Load monitoring |
|---|---|
| One match or training session at 10-second resolution: per-minute running, most demanding scenarios (1–10 min), player comparison, benchmarks by position and playing group. | Daily load, acute:chronic workload, changepoint detection and a time-series toolkit (smoothing, power spectrum, Morlet wavelet, Holt-Winters forecast). |

## Why it's built this way

The pipeline follows the five-stage decision-support framework of Schelling & Robertson (2020)
and Schelling et al. (2025):

1. **Collect** – read raw STATSports exports once each into a 10 s bin cache (`src/pipeline/sources.py`)
2. **Condition** – position → velocity, smoothing, artefact removal (`conditioning.py`)
3. **Features** – session totals, per-minute distance, ACWR (`features.py`)
4. **Patterns** – periodisation (wavelet + spectrum), duration-specific MDS, changepoints (`patterns.py`)
5. **Output** – tidy tables plus the JS payloads the dashboard reads (`output.py`)

Method choices worth calling out:

- **Most demanding scenarios are per-minute with 1–10 min windows**, the scale that fits Gaelic
  football, rather than borrowing basketball's 12-second possession window.
- **Total acceleration load (TAL)** = Σ|Δv| between consecutive 10 Hz filtered-speed samples,
  never bridging dropouts longer than 0.25 s (Schelling et al., 2025).
- **Changepoint detection** uses PELT via `ruptures`, with a NumPy binary-segmentation fallback.
- **Each raw file is scanned once** with a lazy [Polars](https://pola.rs) query (projection and
  predicate pushdown); everything downstream is derived from the parquet cache, so reruns are
  seconds rather than minutes.
- **Drill rows are de-duplicated.** STATSports exports every drill twice (under its own title and
  inside "Entire Session"); only "Entire Session" counts towards load, drill rows supply time windows.

## Run it

```bash
pip install -r requirements.txt
python scripts/make_sample_data.py     # synthetic 14-player squad, 8 weeks (~1 GB, git-ignored)
python run.py                          # pipeline -> outputs/
open dashboard/index.html              # works straight from disk, no server
```

`python publish.py --public` inlines each page (CSS, JS, data) into a single file in `docs/` for
GitHub Pages. It refuses to run unless the pipeline is pointed at the synthetic sample.

## Using real data

Point `data.root` in `config.yaml` (or `GPS_DATA_ROOT`) at a folder of
`YYYY-MM-DD-<Team>-<Match|Training>.csv` exports. Optional squad context (fixtures, grades,
positions) goes in the CSVs listed under `reference:`.

Data governance is built in:

- Players are pseudonymised (P01, P02…) before anything reaches `outputs/` or the browser; the
  name key stays in `cache/`.
- Raw data, caches, reference files, outputs and all CSVs are git-ignored.
- For sharing real data, `python publish.py` (without `--public`) encrypts each page with
  AES-256-GCM, key derived from a password with PBKDF2-SHA256 (600k iterations). The Pages site
  then holds only ciphertext and decrypts in the viewer's browser.

## About the sample data

`scripts/make_sample_data.py` invents everything: names, fixtures, opposition, pitch location and
the movement itself. Speeds follow a mean-reverting baseline with superimposed high-speed efforts,
scaled by position line, across a weekly rhythm (two trainings and a match) with a three-build,
one-deload block. It targets plausible magnitudes (about 100–110 m/min in matches), not a model of
real players.

## References

- Schelling, X., & Robertson, S. (2020). A development framework for decision support systems in
  high-performance sport. *International Journal of Computer Science in Sport*, 19(1), 1–23.
- Schelling, X., et al. (2025). *Applied Sciences*, 15, 10575.

---

Eamon McGleenan · PhD researcher, Predictive Sports Analytics, Queen's University Belfast
