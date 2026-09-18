"""Small judge-facing service that never fabricates geospatial provenance."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Ready Prototype — Wildfire Monitoring",
    version="0.1.0",
    description="Competition-first AF and Burn Severity prototype",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ready-prototype", "version": "0.1.0"}


@app.get("/api/spec")
def spec() -> dict:
    return {
        "modules": {
            "AF": {
                "output": "binary pixel mask",
                "primary_metric": "pixel F1",
                "core_inputs": ["VIIRS I1", "I2", "I3", "I4", "I5"],
            },
            "BS": {
                "output": "0/1/2/3 severity mask",
                "primary_metrics": ["burn IoU", "severity mIoU"],
                "core_inputs": [
                    "Sentinel-2 pre/post",
                    "Sentinel-1 pre/post",
                    "SCL",
                    "land cover",
                    "terrain",
                ],
            },
        },
        "score_formula": "0.35*F1_AF + 0.35*IoU_burn + 0.30*mIoU_severity",
        "inference_command": "python inference.py --data-dir /path/to/test --output submission.csv",
        "principles": [
            "no test geolocation/date reconstruction",
            "no FIRMS or ready-made fire products for private-test answers",
            "no fabricated metrics or scene provenance",
        ],
    }


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ready Prototype</title>
<style>
body{font-family:Inter,Arial,sans-serif;margin:0;background:#0d1117;color:#e6edf3}
main{max-width:980px;margin:auto;padding:40px 20px}
h1{font-size:38px;margin-bottom:8px}.muted{color:#8b949e}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin:28px 0}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px}
.code{font-family:ui-monospace,SFMono-Regular,monospace;background:#010409;padding:14px;border-radius:8px;overflow:auto}
strong{color:#58a6ff}.ok{color:#3fb950}
</style>
</head>
<body><main>
<div class="muted">КосмоХакатон 2026 · competition-first</div>
<h1>Ready Prototype</h1>
<p>Два независимых pixel-wise модуля и единый воспроизводимый inference.</p>
<div class="grid">
<div class="card"><h2>AF · Active Fire</h2><p><strong>VIIRS I1–I5</strong> → binary mask.</p><p>Baseline: MIR/TIR contrast + local anomaly + contextual land-cover prior.</p></div>
<div class="card"><h2>BS · Burn Severity</h2><p><strong>S2 pre/post + S1 + SCL + land cover</strong> → 0/1/2/3.</p><p>Baseline: B8A/B12 dNBR + cloud mask + land-cover-aware thresholds.</p></div>
<div class="card"><h2>Scoring</h2><p class="ok">0.35 F1(AF) + 0.35 IoU(burn) + 0.30 mIoU(severity)</p><p>No synthetic metric is shown as real validation.</p></div>
</div>
<div class="card"><h2>Reproducible entry point</h2>
<div class="code">python inference.py --data-dir /path/to/test --output submission.csv</div>
</div>
<p class="muted">API: /health · /api/spec</p>
</main></body></html>"""
