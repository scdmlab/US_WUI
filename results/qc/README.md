# Step51 WUI-Z Ketchpaw >50% Boundary Rebuild

This run aligns the WUI-Z local vegetation boundary with the manuscript and the
Ketchpaw-style classification used in this study:

- Intermix: development density > 6.17 units/km2 and local wildland vegetation > 50%.
- Interface: development density > 6.17 units/km2, local wildland vegetation <= 50%,
  and distance <= 2.4 km from a qualifying patch.
- Qualifying patch: area >= 5 km2 and wildland vegetation > 75% of total valid land.

Only WUI-Z was rebuilt. WUI-P, WUI-S, frozen county weights, and the frozen
Python/esda/libpysal environment were not modified.

## Final status

`WUIZ_KETCHPAW_GT50_NATIONAL49_COMPLETE`

All 49 reporting units passed transition and geometry/raster QC. The only
allowed changes were class 1 to class 2 (Intermix to Interface) and class 1 to
class 0 (Intermix to Non-WUI).

## Boundary audit and raster impact

- Census blocks audited: 8,089,668
- Blocks with exactly 50% local vegetation: 17,305
- Dense exactly-50% blocks: 6,759
- Dense exactly-50% blocks inside the qualifying-patch buffer: 4,791
- Dense exactly-50% blocks outside the buffer: 1,968
- Changed raster pixels: 688,910
- Intermix to Interface pixels: 456,638
- Intermix to Non-WUI pixels: 232,272
- Net WUI-Z area change: -209.0448 km2
- Unexpected transitions: 0

## Updated national results

- WUI-P: 1,028,551.2660 km2; 89,663,483.35 residents (27.2318%)
- WUI-S: 1,112,334.4431 km2; 94,793,528.78 residents (28.7898%)
- WUI-Z: 761,129.7120 km2; 97,266,925 residents (29.5410%)

National micro-averaged Jaccard values:

- WUI-P / WUI-S: 0.619997125555
- WUI-P / WUI-Z: 0.396657550737
- WUI-S / WUI-Z: 0.416216665473

## Moran and appendix

- Candidate combinations: 245
- Formal eligible combinations: 174
- Significant positive Global Moran combinations: 164
- Non-significant eligible Global Moran combinations: 10
- Excluded or undefined combinations: 71
- Local Moran FDR-significant observations: 1,555
- FDR categories: HH=656, LL=872, LH=20, HL=7
- Global and Local two-run reproducibility: exact
- Appendix full-design rows: A1=252, A2=198, A3=252, A4=450,
  A5=297, A6=297 (total 1,746)

Compared with the preceding strict-patch run, the Global Moran significance
decisions did not change. One WUI-Z county result changed after FDR correction:
Garfield County, Washington (FIPS 53023), from NOT_SIGNIFICANT to LL.

## Key directories

- `audit/`: exact-50% block audit
- `rasters/` and `vectors/`: rebuilt WUI-Z products
- `downstream_metrics/`: area, population, and county metrics
- `pairwise/`: national and five-state intersection/Jaccard results
- `moran_appendix/`: Moran results, figures, and final Appendix A1-A6 tables

No manuscript or Overleaf file was modified in this run.
