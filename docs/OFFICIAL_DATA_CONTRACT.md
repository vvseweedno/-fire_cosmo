# Official Data Contract

## Status

The repository does **not** assume the physical layout or band order of the
official archive. Until an official dataset is mounted and
`scripts/preflight_dataset.py --deep` plus
`scripts/fingerprint_dataset.py` complete successfully, official-data
experiments are **BLOCKED** and no competition accuracy claim is valid.

## Required metadata

Each train/test root must contain `meta.csv` with:

- `chip_id`
- `kind` (`af` or `bs`)
- `width`
- `height`
- `gsd`

For strict leakage-safe competition validation, organiser-provided grouping must
also exist under one of these explicit columns:

- `fire_event_id`
- `event_id`
- `incident_id`
- `group_id`

If grouping is absent, chip-level fallback is detectable but **must not** be
described as strict event-safe validation.

The test root must additionally contain `sample_submission.csv`.

## Task contracts

### Active Fire (AF)

Minimum required physical channels:

- `I4`
- `I5`

Optional supported context includes:

- `I1`, `I2`, `I3`
- `LANDCOVER`
- `VALID_MASK`
- sun/sensor geometry
- ERA5-Land-style atmospheric context when supplied by the organiser

Training AF chips must contain `TARGET`.

### Burn Severity (BS)

Minimum required channels:

- `B8A_PRE`
- `B12_PRE`
- `B8A_POST`
- `B12_POST`

Optional supported inputs include additional Sentinel-2 pre/post bands, SCL,
Sentinel-1 VV/VH pre/post, land cover, terrain, and valid masks.

Training BS chips must contain `TARGET`, whose supported competition semantics
are classes:

- 0 = unburned
- 1 = low severity
- 2 = moderate severity
- 3 = high severity

These class meanings must still be checked against organiser documentation
before an official release is marked PROVEN.

## Multiband raster safety

Band order is never guessed.

A stacked GeoTIFF is accepted only when channel identity is available from at
least one explicit source:

1. band descriptions;
2. recognised per-band tags;
3. an explicit sidecar band/channel mapping.

An ambiguous multiband raster must fail with an actionable error.

NPZ stacks are accepted only when their array keys identify supported channels.

## Shape contract

All channels belonging to one chip must have identical raster shape. That shape
must match the `width` and `height` recorded in `meta.csv`.

## Official audit procedure

Run:

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

For the final pipeline, `scripts/finalize_competition.py` writes the combined
train/test fingerprint to `artifacts/data_audit.json`.

## What must be recorded from the real archive

The first official-data audit must populate evidence for:

- actual directory layout;
- train/test split;
- AF/BS chip counts;
- TIFF/NPY/NPZ counts;
- raster dimensions;
- dtypes;
- nodata;
- CRS and transform where present;
- band descriptions/tags/sidecars;
- available sensor channels;
- target classes;
- organiser grouping coverage;
- sample submission structure;
- stable dataset manifest SHA256.

Do not fill any of these values from expectation or memory.
