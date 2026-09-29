"""Stage 5 - Output. Tidy tables + JS payloads the dashboard reads."""
import json
import pathlib


def write_outputs(cfg, tables, dash):
    out = pathlib.Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitkeep").touch()
    for name, df in tables.items():          # polars DataFrames
        df.write_csv(out / f"{name}.csv")
    # loaded via <script src>, so it works from file:// with no CORS/fetch
    (out / "dashboard_data.js").write_text(
        "window.DASH_DATA = " + json.dumps(dash, default=float) + ";\n"
    )
    return out


def write_js(cfg, name, var, payload):
    out = pathlib.Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(f"window.{var} = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    return out / name
