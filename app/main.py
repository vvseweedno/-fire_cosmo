"""Judge-facing wildfire service with explicit provenance boundaries."""

from __future__ import annotations  # noqa: I001

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from wildfire.aoi import load_monitoring_aoi, monitoring_aoi_summary
from wildfire.operational import operational_capabilities

app = FastAPI(
    title="Wildfire Monitoring — КосмоХакатон 2026",
    version="0.3.0",
    description=(
        "Two-stage near-real-time remote-sensing service: active-fire detection "
        "followed by burned-area/severity assessment."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "wildfire-monitoring", "version": app.version}


@app.get("/api/spec")
def spec() -> dict:
    return {
        "public_task": {
            "stage_1": (
                "detect active burning from thermal Earth-observation data "
                "with false-positive suppression"
            ),
            "stage_2": (
                "map burned area from Sentinel-2 and estimate damage/severity"
            ),
            "product": "web service/API with fire map and burned area in hectares",
            "latency_semantics": (
                "near-real-time after a new satellite observation becomes available"
            ),
        },
        "modules": {
            "AF": {
                "output": "binary active-fire mask / detections",
                "working_metric": "pixel F1",
                "public_sensor_family": ["MODIS", "VIIRS", "Landsat"],
                "implemented_core": ["VIIRS I1", "I2", "I3", "I4", "I5"],
            },
            "BS": {
                "output": "burned-area + 0/1/2/3 internal severity mask",
                "working_metrics": ["burn IoU", "severity mIoU"],
                "public_sensor_family": ["Sentinel-2"],
                "optional_research_context": [
                    "Sentinel-1 pre/post",
                    "SCL",
                    "land cover",
                    "terrain",
                ],
            },
        },
        "working_score_formula": (
            "0.35*F1_AF + 0.35*IoU_burn + 0.30*mIoU_severity"
        ),
        "working_score_status": (
            "configured competition objective; bind organizer confirmation "
            "to release evidence before calling it publicly verified"
        ),
        "aoi_endpoint": "/api/aoi",
        "readiness_endpoint": "/api/readiness",
        "labelled_inference_command": (
            "python inference.py --data-dir /path/to/test --output submission.csv"
        ),
        "principles": [
            "no private-test geolocation/date reconstruction",
            "no hidden-boundary reconstruction",
            "no answer-producing external fire products for private-test labels",
            "no fabricated metrics, coordinates, timestamps, hectares or provenance",
        ],
    }


@app.get("/api/models")
def models() -> dict[str, object]:
    """Expose only real adapter capabilities; no fabricated accuracy metrics."""

    return {
        "service_version": app.version,
        "capabilities": operational_capabilities(),
        "accuracy": {
            "status": "UNVERIFIED_ON_ORGANIZER_LABELS",
            "metrics": None,
        },
    }


@app.get("/api/readiness")
def readiness() -> dict[str, object]:
    """Expose machine-readable demo/release readiness without inventing evidence."""

    aoi_path = os.getenv("WILDFIRE_AOI_GEOJSON")
    aoi_ready = False
    aoi_error: str | None = None
    if aoi_path:
        try:
            load_monitoring_aoi(aoi_path)
            aoi_ready = True
        except (OSError, ValueError) as exc:
            aoi_error = str(exc)

    checks = {
        "service": {"ready": True, "detail": "API process is serving requests"},
        "aoi": {
            "ready": aoi_ready,
            "detail": (
                "configured organizer/public AOI parsed successfully"
                if aoi_ready
                else "set WILDFIRE_AOI_GEOJSON to a valid organizer/public GeoJSON"
            ),
        },
        "organizer_labelled_metrics": {
            "ready": False,
            "detail": "not verified: no organizer-labelled metrics are bundled or claimed",
        },
    }
    if aoi_error:
        checks["aoi"]["error"] = aoi_error

    blockers = [name for name, check in checks.items() if not check["ready"]]
    return {
        "status": "READY_FOR_DEMO" if not blockers else "PARTIAL",
        "service_version": app.version,
        "checks": checks,
        "blockers": blockers,
    }


@app.get("/api/aoi")
def aoi() -> dict[str, object]:
    """Expose only explicitly configured organizer/public AOI metadata."""

    path = os.getenv("WILDFIRE_AOI_GEOJSON")
    if not path:
        raise HTTPException(
            status_code=503,
            detail=(
                "AOI is not configured. Set WILDFIRE_AOI_GEOJSON to an explicit "
                "organizer/public GeoJSON file."
            ),
        )
    try:
        parsed = load_monitoring_aoi(path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"invalid AOI configuration: {exc}") from exc
    return monitoring_aoi_summary(parsed)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wildfire Monitoring</title>
<style>
body{font-family:Inter,Arial,sans-serif;margin:0;background:#0d1117;color:#e6edf3}
main{max-width:1040px;margin:auto;padding:40px 20px}
h1{font-size:40px;margin-bottom:8px}.muted{color:#8b949e}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin:28px 0}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px}
.code{font-family:ui-monospace,SFMono-Regular,monospace;background:#010409;padding:14px;border-radius:8px;overflow:auto}
strong{color:#58a6ff}.ok{color:#3fb950}.warn{color:#d29922}
</style>
</head>
<body><main>
<div class="muted">КосмоХакатон 2026 · двухэтапный мониторинг природных пожаров</div>
<h1>От термоаномалии до карты гари</h1>
<p>Near-real-time после появления нового спутникового наблюдения — без выдуманной
«непрерывности» между пролётами спутников.</p>
<div class="grid">
<div class="card"><h2>1 · Active Fire</h2>
<p><strong>Тепловые ДЗЗ-наблюдения</strong> → обнаружение активного горения.</p>
<p>VIIRS physics/context core + отдельные кандидаты подавления ложных срабатываний.</p>
</div>
<div class="card"><h2>2 · Burned Area / Severity</h2>
<p><strong>Sentinel-2 pre/post</strong> → гарь и степень поражения.</p>
<p>Физические индексы + cloud-aware контекст; дополнительные модели допускаются
только после измеренного выигрыша.</p>
</div>
<div class="card"><h2>Evidence Gate</h2>
<p class="ok">Никакой кандидат не заменяет baseline без leakage-safe проверки.</p>
<p>Dataset fingerprint · error analysis · bootstrap · reproducibility · CI.</p>
</div>
</div>
<div class="card"><h2>Операционный результат</h2>
<p>Карта очагов и гари, источник наблюдения, версия модели и время обработки.
Площадь в гектарах показывается только когда геопривязка позволяет корректно
вычислить площадь пикселя.</p>
<p class="warn">Private-test координаты, даты и скрытые границы не реконструируются.</p>
</div>
<p class="muted">API: /health · /api/spec · /api/models · /api/aoi · /api/readiness</p>
</main></body></html>"""
