#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Resumable dispatcher for strict-patch 500 m P/S/Z in the remaining 44 units."""
from __future__ import annotations

from repo_config import portable_path
import argparse,csv,hashlib,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
from pathlib import Path
import shutil
import pandas as pd

ROOT=Path(portable_path("project"))
STEP43=ROOT/"step43_wuip_p2_formal_rebuild_20260727T180822Z"
STEP46=ROOT/"step46_wuiz_pairwise_completion_20260728T203108Z"
DRIVE=Path(portable_path("data"))
SINGLE=ROOT/"scripts/49_single_state_patch75_500m.py"
FIVE={"CA","CO","FL","PA","TX"}
NAMES={
"AL":"Alabama","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado",
"CT":"Connecticut","DE":"Delaware","DC":"DistrictofColumbia","FL":"Florida",
"GA":"Georgia","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
"KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
"MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
"MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"NewHampshire",
"NJ":"NewJersey","NM":"NewMexico","NY":"NewYork","NC":"NorthCarolina",
"ND":"NorthDakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania",
"RI":"RhodeIsland","SC":"SouthCarolina","SD":"SouthDakota","TN":"Tennessee",
"TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
"WV":"WestVirginia","WI":"Wisconsin","WY":"Wyoming"}
TARGETS=[s for s in NAMES if s not in FIVE]
def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(16*1024*1024),b""): h.update(b)
 return h.hexdigest()
def required(s):
 n=NAMES[s]
 return [
 STEP43/f"intermediate/{s}_p2_sparse_cells.npz",
 DRIVE/f"WUI_S_Paper/{n}/{n}_wildland_bin.tif",
 DRIVE/f"WUI_S_Paper/{n}/{n}_dist_to_largepatch.tif",
 DRIVE/f"mbf_work/centroids_5070/MBF_{n}_centroids_5070.gpkg",
 DRIVE/f"WUI_Z_Results/WUI_Z_Paper_{n}.gpkg",
 STEP46/f"cache/wuiz_masks/{s}/WUI_Z_{s}_class.tif",
 STEP46/f"cache/wuiz_masks/{s}/WUI_Z_{s}_valid_domain.tif",
 STEP43/f"rasters_500m/{s}/WUI_P_P2_{s}_r0500m.tif",
 DRIVE/f"WUI_S_Paper/{n}/WUI_S_{n}_r0500m.tif"]
def run_state(s,out):
 sd=out/"states"/s; status=sd/"step47d_state_status.json"
 if status.exists() and json.loads(status.read_text()).get("status")=="STATE_PSZ_GT75_COMPLETE":
  return {"state":s,"status":"RESUMED_PASS","output":str(sd)}
 if sd.exists():
  failed=out/"failed_attempts"
  failed.mkdir(exist_ok=True)
  stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  shutil.move(str(sd),str(failed/f"{s}_{stamp}"))
 log=out/"logs"/f"{s}.log"
 with log.open("w") as stream:
  cp=subprocess.run([sys.executable,str(SINGLE),"--state",s,"--output",str(sd)],
    stdout=stream,stderr=subprocess.STDOUT,text=True)
 if cp.returncode:
  raise RuntimeError(f"{s} failed; see {log}")
 return {"state":s,"status":"PASS","output":str(sd)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",required=True,type=Path)
 ap.add_argument("--state",choices=TARGETS,default="")
 ap.add_argument("--workers",type=int,default=1);a=ap.parse_args()
 if a.workers < 1 or a.workers > 2:
  raise ValueError("--workers must be 1 or 2")
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 (out/"states").mkdir(exist_ok=True);(out/"logs").mkdir(exist_ok=True)
 chosen=[a.state] if a.state else TARGETS
 inv=[]
 for s in chosen:
  for p in required(s): inv.append({"state":s,"input":str(p),"exists":p.exists(),"size":p.stat().st_size if p.exists() else 0})
 iv=pd.DataFrame(inv);iv.to_csv(out/"step49_input_preflight.csv",index=False)
 if not iv.exists.all(): raise RuntimeError(iv[~iv.exists].to_dict("records"))
 statuses=[];started=time.monotonic()
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  future_state={pool.submit(run_state,s,out):s for s in chosen}
  for i,future in enumerate(as_completed(future_state),1):
   s=future_state[future]
   statuses.append(future.result())
   elapsed=time.monotonic()-started; rate=i/elapsed
   eta=(len(chosen)-i)/rate
   print(
    f"[STEP49] {s} {i}/{len(chosen)} {100*i/len(chosen):.1f}% "
    f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m workers={a.workers}",
    flush=True,
   )
 statuses.sort(key=lambda row: chosen.index(row["state"]))
 pd.DataFrame(statuses).to_csv(out/"step49_state_status.csv",index=False)
 state_rows=[]; impacts=[]; pairs=[]
 for s in chosen:
  sd=out/"states"/s
  state_rows.append(json.loads((sd/"step47d_state_status.json").read_text()))
  impacts.append(pd.read_csv(sd/f"{s}_psz_gt75_impact.csv"))
  pairs.append(pd.read_csv(sd/f"{s}_psz_pairwise.csv"))
 pd.DataFrame(state_rows).to_json(out/"step49_state_status_detail.json",orient="records",indent=2)
 pd.concat(impacts,ignore_index=True).to_csv(out/"step49_patch75_500m_impact.csv",index=False)
 pd.concat(pairs,ignore_index=True).to_csv(out/"step49_patch75_500m_pairwise.csv",index=False)
 final={"step":"STEP49_PATCH75_REMAINING44_500M","status":"CANARY_COMPLETE" if a.state else "REMAINING44_PSZ_500M_COMPLETE",
  "completed_utc":now(),"units":len(chosen),"expected_units":1 if a.state else 44,
  "methods":["WUI-P","WUI-S","WUI-Z"],"buffer_m":500,"formal_overwrite":False}
 (out/"step49_status.json").write_text(json.dumps(final,indent=2)+"\n")
 files=sorted(p for p in out.rglob("*") if p.is_file() and p.name!="sha256_manifest.txt")
 (out/"sha256_manifest.txt").write_text("\n".join(f"{sha(p)}  {p.relative_to(out)}" for p in files)+"\n")
 print(json.dumps(final),flush=True)
if __name__=="__main__":main()
