# Operational Architecture

## Scope

Competition scoring and operational ingestion are separate concerns.

The operational path answers a different question:

> Given a new satellite observation with explicit provenance, can the current
> repository process it without guessing sensor, time, geometry or channel
> identity?

The service is **near-real-time after a new satellite observation is
available**. It does not claim continuous observation between satellite
overpasses.

## Pipeline

```text
organizer/public AOI
        |
        v
discover/acquire new observation
        |
        v
validate explicit provenance
(sensor, timestamp, CRS, bbox, channels)
        |
        +--------------------+
        |                    |
        v                    v
active-fire path         burn-assessment path
VIIRS I4/I5 today        Sentinel-2 pre/post today
        |                    |
        v                    v
AF score/mask            physics/candidate BS
        |                    |
        v                    v
event/update layer       burned area + severity
        \                    /
         \                  /
          v                v
             API / map
```

External acquisition adapters (STAC, provider APIs, local ingestion) belong
above the ML core. They must not silently rename or reorder bands.

## Observation descriptor

`wildfire.operational.ObservationDescriptor` requires:

- explicit observation id;
- explicit sensor family;
- timezone-aware acquisition timestamp;
- explicit, parseable CRS;
- explicit bounding box;
- explicit channel names;
- explicit non-empty source/provenance string.

Missing or malformed metadata fails instead of being guessed. CRS identifiers
are parsed through Rasterio/PROJ rather than accepted merely because they are
non-empty strings.

## Current capability truth table

### Active fire

Public case sensor families:

- MODIS
- VIIRS
- Landsat

Currently implemented inference adapter:

- **VIIRS**, minimum I4 + I5

MODIS/Landsat observations may be structurally valid but are reported
`inference_adapter_implemented=false` until real adapters exist and pass tests.

### Burn assessment

Public case sensor family:

- Sentinel-2

Currently implemented operational pairing contract:

- explicit Sentinel-2 PRE observation;
- explicit Sentinel-2 POST observation;
- PRE timestamp strictly earlier than POST;
- equivalent parsed CRS and same bounding box;
- B8A + B12 present in both observations.

The strict pair represents co-registration metadata. Raster-level shape,
transform and channel identity must still pass the existing I/O checks before
inference.

## Area in hectares

`wildfire.area.burned_area_hectares` calculates area only for a projected CRS
with metre linear units.

For an affine transform the pixel area is the absolute determinant:

```text
pixel_area_m2 = abs(a * e - b * d)
area_ha = burned_pixels * pixel_area_m2 / 10000
```

EPSG:4326 degree grids are rejected instead of treating degrees as metres.
A geographic raster requires a proper geodesic-area implementation before the
UI may display hectares.

## AOI

`wildfire.aoi` reads the explicit organizer/public GeoJSON. It never
reconstructs private-test block geometry.

The API exposes AOI metadata only when `WILDFIRE_AOI_GEOJSON` points to an
explicit local GeoJSON file. Otherwise `GET /api/aoi` returns 503 rather than
inventing a default location.

## Temporal hard-negative prior

Industrial flares and other persistent thermal sources may recur at fixed
locations. `wildfire.temporal.persistent_heat_prior` can derive a recurrence
map from genuinely co-registered historical thermal detections.

This is an **optional soft feature**, never an unconditional fire rejection:

- no aligned history -> no penalty;
- too few valid historical observations -> no penalty;
- explicit recurrence map -> separate AF candidate;
- candidate may receive zero ensemble weight;
- promotion still requires leakage-safe measured gain.

## Safety boundary

Operational public-data integrations must be kept separate from private-test
answer generation. Hidden block boundaries, dates or labels must not be
reconstructed from external fire products when competition rules prohibit it.
