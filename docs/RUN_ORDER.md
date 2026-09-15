# Workflow order

The scripts are kept with their original step numbers so the analysis history remains easy to follow. This is the shortest reading order for the final manuscript workflow.

## 1. Confirm the WUI-P input policy

1. `41_wuip_source_dedup_nodata_impact_audit.py`
2. `42a_manuscript_source_policy_inventory.py`
3. `42b_five_state_address_identity_policy_audit.py`
4. `42c_finalize_address_policy_audit.py`

These steps establish the address-only P2 policy and separate exact duplicate records from legitimate records that share coordinates.

## 2. Build the address-only WUI-P base products

1. `43a_prepare_p2_formal_rebuild.py`
2. `43a_repair_step42_canary_input_freeze.py`
3. `43b_run_p2_formal_rebuild.py`
4. `43c_finalize_p2_formal_rebuild.py`

Step43 is the formal P2 classification rebuild. Step44 then computes the initial area, population, county, and structure-count outputs.

## 3. Apply the strict qualifying-patch rule

1. `47_patch75_silvis_colorado_canary.py`
2. `47b_patch75_colorado_remaining_radii.py`
3. `47c_patch75_colorado_wuiz.py`
4. `47d_patch75_four_state_psz.py`
5. `47d_finalize_four_state_psz.py`
6. `49_single_state_patch75_500m.py`
7. `49_patch75_remaining44_500m.py`
8. `49_merge_national49_patch75_500m.py`

The Colorado canary is checked first, followed by the five-state sensitivity products and the remaining national 500 m products.

## 4. Compute downstream statistics

1. `48a_patch75_five_state_downstream_metrics.py`
2. `48b_patch75_moran_figures_tables.py`
3. `50a_patch75_national49_metrics.py`
4. `50b_patch75_national49_moran_appendix.py`

## 5. Apply the final WUI-Z 50% boundary and replace affected results

1. `51_wuiz_ketchpaw_gt50_rebuild.py`
2. `51a_wuiz_gt50_downstream_metrics.py`
3. `51b_wuiz_gt50_pairwise.py`
4. `51c_wuiz_gt50_moran_appendix.py`

Step51 is the authoritative final version used by the manuscript. Earlier Step50 WUI-Z values should not be reported as final.

## 6. Rebuild manuscript tables and figures

- `53_generate_step51_appendix_latex.py` formats Appendix A1–A6.
- Scripts `52` through `56` rebuild the method and result figures.

## Current limitation

This order describes the verified server workflow. Shared roots are now configurable, but the scripts still retain frozen Step directory names. Configure the external data layout and complete a clean small-state canary before advertising a public reproduction command.
