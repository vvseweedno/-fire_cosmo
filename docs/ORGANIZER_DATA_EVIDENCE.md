# Organizer Data Evidence — 2026-09-18

This document records only material actually observed in the connected
organizer dataset package. It is intentionally separate from model assumptions.

## Source package

Observed logical path:

`Мониторинг DATA/fire-aoi`

The accessible package contained the following items:

| File | Size (bytes) |
|---|---:|
| fire_monitoring_aoi.geojson | 2907 |
| fire_monitoring_aoi.gpkg | 36864 |
| fire_monitoring_aoi_shapefile.zip | 1546 |
| fire_monitoring_aoi.shp | 700 |
| fire_monitoring_aoi.shx | 124 |
| fire_monitoring_aoi.dbf | 2362 |
| fire_monitoring_aoi.prj | 256 |
| fire_monitoring_aoi.cpg | 5 |
| README.md | 2774 |

The GeoJSON SHA256 observed during inspection:

`b5eeac1942c5bf23a92d3155db95fb6df23f1ad8d1b8114087312186279c87ba`

No private Drive URL or credential is stored in the repository.

## GeoJSON facts

The GeoJSON root is a FeatureCollection with three features:

1. `aoi`
2. `utm_32637`
3. `utm_32638`

The explicit `aoi` properties are:

- name: `Нижнее Поволжье и Подонье`
- role: monitoring boundary for natural fires
- CRS: `EPSG:4326`
- seasons: `2019–2025`
- months: `04–10`
- recommended UTM zones: `EPSG:32637`, `EPSG:32638`
- declared area: `435273 km²`

Geometry checks:

- type: Polygon
- exterior positions including closure: 17
- bounding box: `[38.3, 44.6, 48.0, 52.7]`
- first and final positions are identical

## README facts

The organizer README states that:

- the coverage is Нижнее Поволжье and Подонье;
- the AOI spans Rostov, Volgograd and Astrakhan regions, western/central
  Saratov region and the Republic of Kalmykia;
- UTM 37N and 38N are supplied to simplify satellite-scene processing;
- the package is intended to support independent collection/processing of
  imagery;
- private-test block geometries are deliberately not included.

That final point is a competition-safety boundary. This repository does not
attempt to derive or reconstruct those hidden geometries.

## What was not observed in this package

At inspection time the accessible package did **not** expose:

- labelled AF train chips;
- labelled burn-severity train chips;
- `meta.csv`;
- `sample_submission.csv`;
- public/private test block geometry;
- a machine-readable scoring formula;
- a ready-made train/test raster archive.

This does not prove that such material cannot be supplied later. It only records
the current evidence.

## Engineering consequence

The repository keeps its labelled metric-validation harness, but the operational
data path must also support:

`AOI -> satellite observation acquisition -> validation -> AF detection -> event
update -> later Sentinel-2 observation -> burned-area/severity update -> map/API`

No accuracy, hidden-test score or official metric claim can be inferred from
this AOI package alone.
