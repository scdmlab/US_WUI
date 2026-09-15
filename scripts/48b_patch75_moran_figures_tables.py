#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""STEP48B: Moran, candidate figures, and appendix tables for Step48A."""
from __future__ import annotations

from repo_config import portable_path
import argparse, hashlib, importlib.util, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from esda import Moran
from libpysal.weights import W

ROOT=Path(portable_path("project"))
WEIGHTS=ROOT/"step45_p2_formal_spatial_analysis_20260728T025216Z"
MOD_PATH=ROOT/"scripts/45c_p2_formal_moran_py310_compatibility.py"
STATES=["CA","CO","FL","PA","TX"]

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s)
    sys.modules[name]=m; s.loader.exec_module(m); return m
def atomic_csv(p,d):
    p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+".partial")
    d.to_csv(q,index=False,float_format="%.12f"); os.replace(q,p)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(16*1024*1024),b""): h.update(b)
    return h.hexdigest()
def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def combos(metrics, mod):
    nei=pd.read_csv(WEIGHTS/"p2_spatial_weights_neighbors.csv")
    man=pd.read_csv(WEIGHTS/"p2_spatial_weights_manifest.csv")
    result=[]
    specs=[]
    for state in STATES:
        for method in ["WUI-P","WUI-S"]:
            for radius in range(100,1001,100):
                for var,frozen in [("p_a","area_proportion"),("p_s","structure_proportion")]:
                    specs.append((state,method,radius,var,frozen,method))
        specs.append((state,"WUI-Z",0,"p_a","area_proportion","WUI-P"))
    for state,method,radius,var,frozen,weight_method in specs:
        rec=man[(man.state.eq(state))&(man.method.eq(weight_method))&
                (man.variable.eq(frozen))].iloc[0]
        ids=[int(x) for x in str(rec.id_order).split("|")]
        nf=nei[(nei.state.eq(state))&(nei.method.eq(weight_method))&
               (nei.variable.eq(frozen))]
        neighbors={i:[] for i in ids}
        for r in nf.itertuples(index=False): neighbors[int(r.fips)].append(int(r.neighbor_fips))
        for i in neighbors: neighbors[i]=sorted(neighbors[i])
        part=metrics[(metrics.state.eq(state))&(metrics.method.eq(method))&
                     (metrics.buffer_m.eq(radius))][["GEOID_INT",var]].dropna().sort_values("GEOID_INT")
        if part.GEOID_INT.astype(int).tolist()!=ids:
            raise RuntimeError(f"FIPS order mismatch {state}/{method}/{radius}/{var}")
        w=W(neighbors,id_order=ids,silence_warnings=True); w.transform="r"
        y=part[var].to_numpy(float)
        result.append({"state":state,"method":method,"buffer_m":radius,
          "variable":var,"frozen_variable":frozen,"ids":ids,"county":[""]*len(ids),
          "neighbors":neighbors,"w":w,"y":y,
          "islands":np.array([not neighbors[i] for i in ids],bool),
          "weights_sha256":str(rec.weights_sha256),"manifest":rec})
    return result

def run_moran(out, cs, mod):
    grows=[]; lrows=[]; fam=[]; repro=[]; total=len(cs)
    for k,c in enumerate(cs,1):
        mi=Moran(c["y"],c["w"],transformation="r",permutations=0,two_tailed=True)
        grows.append({"state":c["state"],"method":c["method"],"buffer_m":c["buffer_m"],
          "variable":c["variable"],"n_units":len(c["ids"]),"n_islands":int(c["islands"].sum()),
          "moran_i":mi.I,"expected_i":mi.EI,"variance_norm":mi.VI_norm,
          "z_norm":mi.z_norm,"p_norm":mi.p_norm,
          "significant_p_norm":bool(mi.p_norm<0.05),"status":"FORMAL_COMPLETE",
          "weights_sha256":c["weights_sha256"]})
        a1=mod.run_local_library(c); rows,fr=mod.local_rows_for_combo(c,a1)
        family_id=(
            f"{c['state']}__{c['method'].replace('-','')}__"
            f"{c['variable']}__r{c['buffer_m']:04d}m"
        )
        for r in rows:
            r["buffer_m"]=c["buffer_m"]; r["analysis_scope"]="county_within_state_patch75"
            r["fdr_family_id"]=family_id
        fr["buffer_m"]=c["buffer_m"]; fr["fdr_family_id"]=family_id
        lrows.extend(rows); fam.append(fr)
        a2=mod.run_local_library(c)
        ok=all(mod.array_sha256(a1[x])==mod.array_sha256(a2[x])
               for x in ["Is","sim","p_sim_directed_legacy","p_sim_two_sided","z_sim"])
        repro.append({"state":c["state"],"method":c["method"],"buffer_m":c["buffer_m"],
          "variable":c["variable"],"exact_reproduction":ok})
        print(f"[MORAN] {k}/{total} {c['state']} {c['method']} {c['buffer_m']} {c['variable']}",flush=True)
    g=pd.DataFrame(grows); l=pd.DataFrame(lrows); f=pd.DataFrame(fam); r=pd.DataFrame(repro)
    atomic_csv(out/"moran/patch75_global_moran_205.csv",g)
    atomic_csv(out/"moran/patch75_local_moran_results.csv",l)
    atomic_csv(out/"moran/patch75_local_fdr_family_audit.csv",f)
    atomic_csv(out/"moran/patch75_moran_reproducibility.csv",r)
    return g,l,f,r

def tables_figures(out, base, metrics, g, l):
    area=pd.read_csv(base/"patch75_five_state_area.csv")
    pop=pd.read_csv(base/"patch75_five_state_population.csv")
    a2=(metrics[metrics.method.isin(["WUI-P","WUI-S"])]
        .groupby(["state","method","buffer_m"],as_index=False)
        [["Total_struct","Intermix_struct","Interface_struct","WUI_struct"]].sum())
    atomic_csv(out/"tables/Appendix_A1_patch75_area_candidate.csv",area)
    atomic_csv(out/"tables/Appendix_A2_patch75_structure_counts_candidate.csv",a2)
    atomic_csv(out/"tables/Appendix_A3_patch75_population_candidate.csv",pop)
    atomic_csv(out/"tables/Appendix_A4_patch75_global_moran_candidate.csv",g)
    counts=(l.groupby(["state","method","buffer_m","variable","formal_fdr_cluster_type"])
            .size().unstack(fill_value=0).reset_index())
    atomic_csv(out/"tables/patch75_local_moran_cluster_counts_candidate.csv",counts)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for metric,frame,y,label in [
            ("area",area,"wui_area_km2","WUI area (km²)"),
            ("population",pop,"wui_population","WUI population")]:
            fig,axs=plt.subplots(1,5,figsize=(17,3.4),sharey=False)
            for ax,state in zip(axs,STATES):
                d=frame[(frame.state.eq(state))&frame.method.isin(["WUI-P","WUI-S"])]
                for m in ["WUI-P","WUI-S"]:
                    q=d[d.method.eq(m)].sort_values("buffer_m")
                    ax.plot(q.buffer_m,q[y],marker="o",ms=3,label=m)
                ax.set_title(state); ax.set_xlabel("Radius (m)"); ax.grid(alpha=.25)
            axs[0].set_ylabel(label); axs[-1].legend()
            fig.tight_layout(); fig.savefig(out/f"figures/patch75_{metric}_sensitivity.png",dpi=300); plt.close(fig)
        fig,axs=plt.subplots(1,5,figsize=(17,3.4))
        for ax,state in zip(axs,STATES):
            d=g[(g.state.eq(state))&g.method.isin(["WUI-P","WUI-S"])&g.variable.eq("p_a")]
            for m in ["WUI-P","WUI-S"]:
                q=d[d.method.eq(m)].sort_values("buffer_m"); ax.plot(q.buffer_m,q.moran_i,marker="o",ms=3,label=m)
            ax.set_title(state); ax.grid(alpha=.25); ax.set_xlabel("Radius (m)")
        axs[0].set_ylabel("Global Moran's I"); axs[-1].legend(); fig.tight_layout()
        fig.savefig(out/"figures/patch75_global_moran_sensitivity.png",dpi=300); plt.close(fig)
    except Exception as e:
        (out/"figures/FIGURE_ERROR.txt").write_text(repr(e)+"\n")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--base",type=Path,required=True); args=ap.parse_args()
    base=args.base.resolve(); out=base/"step48b_moran_figures_tables"; out.mkdir(exist_ok=True)
    for s in ["moran","tables","figures"]: (out/s).mkdir(exist_ok=True)
    mod=load(MOD_PATH,"step48b_mod")
    metrics=pd.read_csv(base/"county_metrics/patch75_five_state_county_metrics.csv")
    cs=combos(metrics,mod)
    if len(cs)!=205: raise RuntimeError(f"Expected 205 combos, got {len(cs)}")
    g,l,f,r=run_moran(out,cs,mod); tables_figures(out,base,metrics,g,l)
    qc={"combinations":len(g),"local_rows":len(l),"families":len(f),
        "reproduction_all":bool(r.exact_reproduction.all()),
        "wui_z_ps_excluded":True,"completed_utc":now(),
        "status":"PATCH75_FIVE_STATE_MORAN_FIGURES_TABLES_COMPLETE"}
    (out/"step48b_status.json").write_text(json.dumps(qc,indent=2)+"\n")
    files=sorted(p for p in out.rglob("*") if p.is_file() and p.name!="sha256_manifest.txt")
    (out/"sha256_manifest.txt").write_text("\n".join(f"{sha(p)}  {p.relative_to(out)}" for p in files)+"\n")
    print(json.dumps(qc),flush=True)
if __name__=="__main__": main()
