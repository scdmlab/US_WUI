# Repository readiness audit

Prepared on 2026-09-15 from the frozen Step51 workflow.

## What is ready

- 39 workflow scripts, one clearly labeled legacy reference script, and two shared configuration helpers are included.
- User-specific server and ResearchDrive paths have been removed from the repository copy.
- All shared roots are read from `config/paths.json` through `scripts/repo_config.py`.
- Every analysis script has the same short repository header explaining the configuration and external-data policy.
- Final Appendix A1–A6 tables and national summary results are included.
- Final Global and Local Moran tables are included; large permutation arrays are excluded.
- No raw address records, address examples, point coordinates, raster products, vector products, or archives are included.
- A basic secret scan found no passwords, API keys, email addresses, or private keys.
- Result-file server paths were normalized to placeholders.
- GPL-3.0-only licensing and preliminary citation metadata are included.

## What still needs work before a public release

1. **Fresh-environment test:** copy `paths.example.json` to `paths.json`, enter the local roots, and run a small-state canary. Do not change the current production server environments to do this.
2. **One entry point:** the workflow is recorded as numbered research steps. A small runner or Makefile/Snakemake workflow can be added after the inputs and restart behavior are agreed on.
3. **Timestamped dependencies:** several scripts intentionally name frozen Step directories. Confirm whether these should stay fixed for provenance or be moved into a separate run manifest.
4. **Data instructions:** add official download links, versions, dates, and license notes.
5. **Citation:** add the manuscript DOI and final journal details to `CITATION.cff` after acceptance or preprint release.
6. **Authorship approval:** confirm that all authors agree with the public code and data release.

## Important exclusions

`process_oa_v2.py` is not included because it represents the superseded mixed address/building legacy policy and could be mistaken for the final WUI-P production method. The copied `legacy_reference/09_analyze_wui_p_raster_fast.py` is retained only because its hash was frozen in the Step43 provenance record; it is not the final address-policy definition.

The repository is suitable for an initial **private backup** and its paths are now configurable. The code license is settled as GPL-3.0-only, but the repository should not yet be described as a fully tested public reproduction package until the clean small-state canary and data-license review are complete.
