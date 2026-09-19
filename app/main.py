"""Judge-facing wildfire service with explicit provenance boundaries."""

from __future__ import annotations  # noqa: I001

import json
import os
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from wildfire.aoi import load_monitoring_aoi, monitoring_aoi_summary
from wildfire.operational import operational_capabilities
from wildfire.service import (
    feature_collection,
    load_results,
    normalize_polygon,
    parse_bbox,
    query_results,
)


class SpatialTemporalRequest(BaseModel):
    """Machine-readable spatial-temporal query for the offline result catalog."""

    bbox: tuple[float, float, float, float] | None = None
    polygon: list[list[float]] | None = None
    start_date: date | None = None
    end_date: date | None = None

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
        "result_endpoints": {
            "query": "/api/query",
            "summary": "/api/summary",
            "geojson": "/api/export/geojson",
            "map": "/map",
        },
        "demo_catalog": "synthetic offline catalog; replace with explicit WILDFIRE_RESULTS_GEOJSON",
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

    checks: dict[str, dict[str, object]] = {
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


def _query_results(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    polygon: list[list[float]] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, object]:
    try:
        catalog = load_results()
        normalized_polygon = normalize_polygon(polygon)
        return query_results(
            catalog,
            bbox=bbox,
            polygon=normalized_polygon,
            start_date=start_date,
            end_date=end_date,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid result query/catalog: {exc}") from exc


@app.get("/api/query")
def query(
    bbox: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, object]:
    try:
        parsed_bbox = parse_bbox(bbox)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _query_results(
        bbox=parsed_bbox,
        start_date=start_date,
        end_date=end_date,
    )


@app.post("/api/query")
def query_post(request: SpatialTemporalRequest) -> dict[str, object]:
    return _query_results(
        bbox=request.bbox,
        polygon=request.polygon,
        start_date=request.start_date,
        end_date=request.end_date,
    )


@app.get("/api/observations")
def observations(
    bbox: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, object]:
    """Alias kept short for demo clients that call the result catalog directly."""

    return query(bbox=bbox, start_date=start_date, end_date=end_date)


@app.get("/api/summary")
def summary(
    bbox: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, object]:
    return query(bbox=bbox, start_date=start_date, end_date=end_date)["summary"]


@app.get("/api/export/geojson")
def export_geojson(
    bbox: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> Response:
    payload = query(bbox=bbox, start_date=start_date, end_date=end_date)
    active = payload["active_fire_points"]["features"]
    burned = payload["burned_area_contours"]["features"]
    export = feature_collection([*active, *burned])
    return Response(
        content=json.dumps(export, ensure_ascii=False),
        media_type="application/geo+json",
        headers={"Content-Disposition": 'attachment; filename="wildfire_results.geojson"'},
    )


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
<p class="muted">API: /health · /api/spec · /api/readiness · /api/query · /api/summary · /api/export/geojson</p>
</main></body></html>"""


@app.get("/map", response_class=HTMLResponse)
def map_view() -> str:
    """Render a dependency-free demo map so the service works offline."""

    payload = _query_results()
    serialized = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wildfire results map</title>
<style>body{{font-family:system-ui,sans-serif;margin:0;padding:20px;color:#1f2937;background:#f8fafc}}#map{{max-width:900px;border:1px solid #cbd5e1;background:#e2e8f0}}svg{{display:block;width:100%;height:auto}}pre{{white-space:pre-wrap}}</style></head>
<body><h1>Wildfire results</h1><div id="map" aria-label="Offline wildfire map"></div><pre id="summary"></pre>
<script>
const payload = {serialized};
const features = [...payload.active_fire_points.features, ...payload.burned_area_contours.features];
const points = features.flatMap(f => f.geometry.type === 'Point' ? [f.geometry.coordinates] : f.geometry.coordinates[0]);
const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
const minX = Math.min(...xs) - 0.02, maxX = Math.max(...xs) + 0.02, minY = Math.min(...ys) - 0.02, maxY = Math.max(...ys) + 0.02;
const project = (p) => [((p[0]-minX)/(maxX-minX))*760+20, 360-((p[1]-minY)/(maxY-minY))*320];
let svg = '<svg viewBox="0 0 800 400" role="img" aria-label="Offline fire and burned area map"><rect width="800" height="400" fill="#dbeafe"/><path d="M20 360H780 M20 280H780 M20 200H780 M20 120H780 M20 40H780" stroke="#bfdbfe"/>';
for (const feature of payload.burned_area_contours.features) {{ const pts = feature.geometry.coordinates[0].map(project).map(p => p.join(',')).join(' '); const sev = feature.properties.severity_class; svg += `<polygon points="${{pts}}" fill="${{sev === 3 ? '#dc2626' : sev === 2 ? '#f97316' : '#facc15'}}" fill-opacity=".55" stroke="#7f1d1d"/><text x="${{project(feature.geometry.coordinates[0][0])[0]}}" y="${{project(feature.geometry.coordinates[0][0])[1]}}" font-size="12">S${{sev}}</text>`; }}
for (const feature of payload.active_fire_points.features) {{ const p = project(feature.geometry.coordinates); svg += `<circle cx="${{p[0]}}" cy="${{p[1]}}" r="6" fill="#111827" stroke="#fff" stroke-width="2"/>`; }}
svg += '</svg>'; document.getElementById('map').innerHTML = svg; document.getElementById('summary').textContent = JSON.stringify(payload.summary, null, 2);
</script></body></html>"""
