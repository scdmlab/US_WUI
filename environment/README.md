# Recorded environments

The production workflow used two Python 3.10.19 environments. They are listed separately because forcing all packages into one environment was not part of the validated analysis.

- `geospatial-production.txt` records the main raster/vector processing stack used in Step43 and related geospatial work.
- `moran-frozen.txt` records the Step45C/Step51 Moran environment and its compatibility protocol.

These files are provenance records, not instructions to modify the current server. Recreate and test them only in a separate environment after approval.

