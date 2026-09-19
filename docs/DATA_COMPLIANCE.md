# Data Compliance Register

Status: **PENDING final organizer rule confirmation and access audit**.

| Resource | Purpose | Allowed scope | License/provenance | Test lookup check |
|---|---|---|---|---|
| Organizer-provided labelled chips | train/evaluate the case | organizer train/test only | organizer terms | pending package audit |
| Synthetic offline demo catalog | API demonstration | no model training or scoring | repository-authored | contains no private-test data |
| Public Python dependencies | execution | code runtime only | package licenses must be captured in release | not answer-producing data |

The release process forbids using FIRMS, VNP14, MOD14/MYD14, Sentinel-3 World
Fire Atlas, MCD64A1, FireCCI51, GABAM, EFFIS, or similar answer-producing fire
products to derive private-test answers. Historical open data may be added only
for permitted training scopes, with source, license, access date, and a written
statement that private test territories/dates were not queried.

Before a final release, every external source must be added to this table and
the release manifest. Missing provenance keeps the release `NOT READY`.
