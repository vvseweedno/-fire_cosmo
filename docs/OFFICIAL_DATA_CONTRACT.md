# Official / Organizer Data Contract

## Evidence status

Do **not** assume that the hackathon provides a ready-made train/test chip
dataset.

The organizer material currently accessible on 2026-09-18 contains an AOI
package named `fire-aoi`. Its README and GeoJSON explicitly describe a
monitoring territory for self-collection/processing of satellite observations.
No `meta.csv`, `sample_submission.csv`, labelled AF/BS chips, or public/private
test geometries were present in the accessible package at inspection time.

Therefore:

- the repository's chip-based evaluation harness remains useful for labelled
  experiments when/if such data are supplied;
- its schema must **not** be described as the organizer's observed file format;
- official/organizer accuracy is **BLOCKED** until labelled evaluation material
  and its scoring rules are actually supplied or documented;
- operational AOI ingestion is a first-class competition path, not a P3
  afterthought.

## Observed organizer AOI package

The accessible GeoJSON is a FeatureCollection with three explicit features:

- `aoi` — monitoring boundary;
- `utm_32637` — recommended UTM 37N longitude strip;
- `utm_32638` — recommended UTM 38N longitude strip.

The `aoi` feature explicitly reports:

- name: Нижнее Поволжье и Подонье;
- CRS: EPSG:4326;
- seasons: 2019–2025;
- months: 04–10;
- recommended projected zones: EPSG:32637 and EPSG:32638;
- declared area: 435273 km²;
- one closed Polygon exterior ring.

The organizer README explicitly says that private-test block geometries are not
included. The repository must never attempt to reconstruct those hidden
boundaries for answer generation.

Inspect a local copy with:

```bash
python scripts/inspect_aoi.py \
  --geojson fire_monitoring_aoi.geojson \
  --output artifacts/aoi_audit.json
```

The AOI parser validates explicit geometry/properties only; it does not infer
private regions or labels.

## Public task contract

The public case description requires a two-stage remote-sensing service:

1. detect active burning from thermal observations (MODIS, VIIRS, Landsat)
   while suppressing false alarms;
2. map burned areas from Sentinel-2 and estimate forest damage/severity.

The expected product surface is a web service or API with a fire map and burned
area in hectares.

This public product contract is independent of any internal experimental
metric. A metric formula must not be called "official" unless organizer
documentation explicitly confirms it.

## Repository labelled-evaluation contract (conditional)

The repository also supports a strict chip-based labelled evaluation harness.
This section describes the **repository input contract**, not a claim about the
currently observed organizer package.

A labelled train/test root accepted by the current harness contains
`meta.csv` with:

- `chip_id`
- `kind` (`af` or `bs`)
- `width`
- `height`
- `gsd`

For strict leakage-safe validation, organizer-provided grouping should exist
under one of these explicit columns:

- `fire_event_id`
- `event_id`
- `incident_id`
- `group_id`

If grouping is absent, chip-level fallback is detectable but **must not** be
described as strict event-safe validation.

A submission-style test root, when used, additionally contains
`sample_submission.csv`.

### Active Fire (AF)

Minimum supported physical channels:

- `I4`
- `I5`

Optional supported context includes:

- `I1`, `I2`, `I3`
- `LANDCOVER`
- `VALID_MASK`
- sun/sensor geometry
- atmospheric context when explicitly supplied

Labelled AF training chips contain `TARGET`.

### Burn Severity (BS)

Minimum supported channels:

- `B8A_PRE`
- `B12_PRE`
- `B8A_POST`
- `B12_POST`

Optional supported inputs include additional Sentinel-2 pre/post bands, SCL,
Sentinel-1 VV/VH pre/post, land cover, terrain, and valid masks.

The current internal severity harness supports classes:

- 0 = unburned
- 1 = low severity
- 2 = moderate severity
- 3 = high severity

These semantics are an internal evaluation contract until organizer
documentation explicitly confirms the target definition.

## Multiband raster safety

Band order is never guessed.

A stacked GeoTIFF is accepted only when channel identity is available from at
least one explicit source:

1. band descriptions;
2. recognised per-band tags;
3. an explicit sidecar band/channel mapping.

An ambiguous multiband raster must fail with an actionable error.

NPZ stacks are accepted only when their array keys identify supported channels.

## Shape contract for labelled chips

All channels belonging to one chip must have identical raster shape. That shape
must match the `width` and `height` recorded in `meta.csv`.

## Audit procedure when labelled data arrive

```bash
python scripts/preflight_dataset.py --data-dir /path/train --mode train --deep \
  --output artifacts/train_preflight.json

python scripts/preflight_dataset.py --data-dir /path/test --mode test --deep \
  --output artifacts/test_preflight.json

python scripts/fingerprint_dataset.py --data-dir /path/train \
  --output artifacts/train_data_audit.json

python scripts/fingerprint_dataset.py --data-dir /path/test \
  --output artifacts/test_data_audit.json
```

For the full labelled pipeline, `scripts/finalize_competition.py` writes the
combined train/test fingerprint to `artifacts/data_audit.json`.

## What must still be learned from real evaluation material

If additional organizer data arrive, record rather than guess:

- actual directory layout;
- whether a labelled train/test split exists;
- AF/BS sample counts;
- raster formats and dimensions;
- dtypes and nodata;
- CRS and transforms;
- band descriptions/tags/sidecars;
- available sensor channels;
- target definitions/classes;
- organizer grouping/event metadata;
- scoring rules;
- submission format, if any;
- stable dataset fingerprint.

Never fill these values from expectation, memory, or private-test
reconstruction.
