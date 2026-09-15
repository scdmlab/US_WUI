# WUI second-round workflow authority

Confirmed by the project owner on 2026-07-27.

## Method and buffer scope

- WUI-P and WUI-S use buffer distance.
- The five sensitivity states are California, Colorado, Florida, Pennsylvania,
  and Texas, with buffers from 100 m through 1000 m at 100 m intervals.
- For the other reporting states, the formal national WUI-P/WUI-S result uses
  the 500 m buffer.
- WUI-Z is a fixed Census-zone classification for each state. It has no
  100–1000 m buffer dimension and must not be relabeled as a 500 m result.

## Workflow precedence

- `<PROJECT_ROOT>` is the authoritative second-round
  update and validation project.
- Scripts, versioned step outputs, manifests, and QC products in this project
  define the main workflow for revised results.
- `<DATA_ROOT>` is the authoritative location for large formal
  inputs and preserved upstream products used by the project workflow.
- Historical manuscript values and legacy scripts are comparison evidence, not
  the revised source of truth, unless a versioned project QC product explicitly
  adopts them.

## Manuscript implication

The existing manuscript is treated as the old-version baseline until its text,
tables, and figures are explicitly replaced from validated outputs in this
project.
