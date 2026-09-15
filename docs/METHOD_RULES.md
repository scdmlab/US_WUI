# Final method rules

This note records the rules behind the Step51 manuscript results.

## Study design

- National scope: 48 conterminous states plus the District of Columbia.
- National setting: WUI-P and WUI-S at 500 m; WUI-Z fixed.
- Five-state sensitivity analysis: California, Colorado, Florida, Pennsylvania, and Texas at 100–1000 m in 100 m steps for WUI-P and WUI-S.
- Grid: 30 m in EPSG:5070.

## Development inputs

- WUI-P uses OpenAddresses records only.
- The P2 policy removes exact duplicate records but does not remove records merely because they share coordinates.
- WUI-S converts each mapped building footprint to one centroid. Multiple centroids in the same 30 m cell are counted separately.
- WUI-Z uses 2020 Census housing-unit density at the block level.

These are three different development-density proxies and should not be read as identical physical units.

## Classification

All three methods use the common numerical density threshold `D > 6.17 units/km²`.

- Intermix: `D > 6.17` and local wildland vegetation `> 50%`.
- Interface: `D > 6.17`, local wildland vegetation `<= 50%`, and Euclidean distance to a qualifying patch `<= 2.4 km`.
- Non-WUI: all other valid locations.
- Qualifying patch: wildland vegetation `> 75%` of total valid land and contiguous area `>= 5 km²`.

Class 0 is valid non-WUI. True outside-domain or unavailable cells are kept separate as NoData (255 in the final classification rasters).

## Pairwise agreement

Intermix and interface are merged into binary WUI before pairwise comparison. Each pair is evaluated only on its common valid domain; valid class 0 cells remain in the analysis. Jaccard similarity is intersection divided by union. The national micro value is calculated after summing intersection and union counts across all 49 reporting units.

## Population and county metrics

Population comes from the 2020 Decennial Census. County analyses include counties and county-equivalent units. Population closure is checked within each method and setting.

## Moran protocol

- Separate county-level network for each state-level unit.
- First-order Queen contiguity and row standardization.
- Islands retained without artificial neighbors and reported as undefined.
- Formal inference only for combinations with at least 30 county or county-equivalent units.
- 999 permutations and fixed seed 20260728.
- Formal Local Moran significance uses a two-sided pseudo p-value derived from stored conditional simulations.
- Benjamini–Hochberg FDR is applied within each eligible state–method–variable combination.

Of 245 candidate combinations, 174 met the formal eligibility rules. Among them, 164 had significant positive Global Moran's I and 10 were not significant. Sixty-six small-network combinations were excluded from formal inference, and five District of Columbia combinations were undefined.

