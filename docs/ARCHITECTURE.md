# System Architecture

```text
official role-based rasters
        |
        v
strict preflight + dataset fingerprint
        |
        v
grouped cross-fit / canonical micro scorer
        |
        v
frozen deployment config
        |
        v
loud-fail inference -> template-aligned RLE submission

result catalog -> REST query -> offline map / GeoJSON / area summary
```

The accuracy path and service path share physical data contracts but have
different release responsibilities. The competition path is judged by exact
submission correctness and cross-fitted evidence. The service path is a
near-real-time demonstration surface with explicit provenance and no private
test reconstruction.

## Accuracy path

`wildfire.io` resolves split-band, metadata-labelled stacks, sidecars, and the
official role-based GeoTIFF names. `wildfire.baselines` produces deterministic
AF/BS scores. `wildfire.crossfit`, `wildfire.evaluation`, and
`wildfire.release_gate` own selection, scoring, and proof status.

## Release path

`scripts/preflight_dataset.py` validates inputs. `inference.py` handles one
production process and fails with chip/source diagnostics. `scripts/validate_submission.py`
checks exact template order and canonical RLE. `scripts/benchmark_inference.py`,
`scripts/readiness_scorecard.py`, and `scripts/reproduce_final.py` create
measured runtime, readiness, and manifest artifacts.

## Service path

`app.main` exposes health/spec/model capabilities, spatial-temporal query,
analytical summary, GeoJSON export, and a dependency-free offline map. The
default catalog is synthetic and is replaced only by an explicit
`WILDFIRE_RESULTS_GEOJSON` file.
