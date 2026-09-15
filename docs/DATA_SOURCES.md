# Data sources and repository policy

The workflow depends on data that should be downloaded from their original providers or stored in a separate data archive. They should not be committed to the GitHub code repository.

| Use | Source | Main fields or role | GitHub policy |
|---|---|---|---|
| WUI-P development input | OpenAddresses | Address-point records | Do not upload raw records; provide download date and processing instructions |
| WUI-S development input | Microsoft Building Footprints | Building polygons converted to centroids | Do not upload raw national files; link to provider and check redistribution terms |
| Vegetation | NLCD 2022 land cover | Binary wildland-vegetation mask and qualifying patches | Do not upload national rasters; cite and link to source |
| WUI-Z development input | 2020 Decennial Census blocks | `HOUSING20` and `ALAND20` | Do not upload national geometry; document download source |
| Population | 2020 Decennial Census P.L. 94-171 blocks | `POP20` | Do not upload national geometry; aggregated output tables may be shared |
| Spatial analysis | Census county and county-equivalent boundaries | County IDs and contiguity weights | Do not upload source geometry unless redistribution is confirmed |
| Reporting units | Census state boundaries | 48 states plus DC | Do not upload source geometry unless redistribution is confirmed |

The CSV files under `results/` contain aggregated state, county, or method-level statistics. They do not contain street addresses, house numbers, units, postcodes, or point coordinates. Server paths in copied result tables were replaced with `<PROJECT_ROOT>`, `<DATA_ROOT>`, or `<LEGACY_ROOT>`; numerical results were not changed.

Before public release, add the exact download date, provider URL, version, and license or terms-of-use reference for every source above.

