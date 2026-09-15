# CONUS WUI mapping with structure-location data

This repository contains the main scripts and final tabular outputs used for the manuscript **“Mapping the Wildland–Urban Interface in the Conterminous United States Using Structure Location Data.”** It compares three WUI representations:

- **WUI-P:** retained OpenAddresses records after exact-record deduplication;
- **WUI-S:** Microsoft Building Footprint centroids;
- **WUI-Z:** a fixed 2020 Census-block-based product.

The national comparison covers the 48 conterminous states and the District of Columbia. WUI-P and WUI-S use a common 500 m neighborhood setting for the national analysis. California, Colorado, Florida, Pennsylvania, and Texas are also evaluated from 100 to 1000 m at 100 m intervals. WUI-Z is fixed and does not use a circular neighborhood radius.

## Final method used in the manuscript

- Development-density threshold: `D > 6.17 units/km²`.
- Intermix WUI: local wildland vegetation `> 50%`.
- Interface WUI: local wildland vegetation `<= 50%` and distance to a qualifying patch `<= 2.4 km`.
- Qualifying patch: wildland vegetation `> 75%` of valid land and contiguous area `>= 5 km²`.
- WUI-P input policy: address-only P2, removing only exact duplicate records.
- Common grid: 30 m, CONUS Albers / EPSG:5070.

The Step51 WUI-Z rebuild explicitly applies the `> 50% / <= 50%` boundary. In the WUI-P and WUI-S circular binary kernels, an exact 50% local value does not occur because the kernels contain an odd number of cells, so the alternative equality placement gives the same classification for those products.

## Final national results

| Method | WUI area (km²) | WUI population | Population share |
|---|---:|---:|---:|
| WUI-P | 1,028,551.2660 | 89,663,483 | 27.2318% |
| WUI-S | 1,112,334.4431 | 94,793,529 | 28.7898% |
| WUI-Z | 761,129.7120 | 97,266,925 | 29.5410% |

National micro-averaged Jaccard similarity is 0.619997 for WUI-P/WUI-S, 0.416217 for WUI-S/WUI-Z, and 0.396658 for WUI-P/WUI-Z.

## Repository layout

```text
scripts/                  Frozen workflow scripts and one labeled legacy reference
results/appendix_tables/  Final Appendix A1–A6 CSV files
results/national_summaries/ National area, population, county, and Jaccard results
results/moran/            Final Global and Local Moran tabular results
results/qc/               Step51 quality-control records
docs/                     Method, data, script, and release notes
environment/              Recorded software environments
config/                   Shared local path configuration template
```

## Configure paths

The repository copy does not contain the original server or ResearchDrive paths. Copy the example configuration and edit the roots for the computer being used:

```bash
cp config/paths.example.json config/paths.json
```

All shared locations are read through `scripts/repo_config.py`. A different configuration file can be selected with the `WUI_CONFIG_FILE` environment variable. The scientific logic and fixed Step directories remain in the numbered scripts so the analysis history is still visible.

The configured locations can be checked without running an analysis:

```bash
python scripts/check_configuration.py
```

This folder is still a **repository candidate**, not yet a tested one-command public release. A clean small-state canary should be run after the external inputs have been arranged under the configured roots. See [docs/REPOSITORY_READINESS.md](docs/REPOSITORY_READINESS.md).

Raw OpenAddresses records, building footprints, Census geometries, NLCD rasters, WUI rasters, and permutation simulation arrays are intentionally excluded. They are too large for a normal GitHub repository and some require separate license or redistribution checks.

## Software

Two recorded environments were used because the geospatial production workflow and the frozen Moran protocol had different package stacks. The files in `environment/` document those environments; they are reference records and should not be installed directly on the production server without approval.

## Citation and license

The manuscript has not yet received a final journal citation. Citation metadata for the current author list are provided in [`CITATION.cff`](CITATION.cff) and should be updated when the manuscript receives a DOI or other permanent identifier.

The original software in this repository is distributed under the **GNU General Public License v3.0 only (`GPL-3.0-only`)**. See [`LICENSE`](LICENSE) for the complete terms. Modified or redistributed versions of the covered software must follow the same license requirements.

The GPL license applies to the project software, not automatically to third-party input data. OpenAddresses records, Microsoft Building Footprints, NLCD products, Census data, and other external inputs remain subject to their own source licenses and terms. These large source datasets are not redistributed in this repository.
