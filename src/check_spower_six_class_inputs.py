#!/usr/bin/env python3
"""Audit Spower six-class input tables before training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {label}. Tried: {[str(p) for p in paths]}")


def parse_args():
    p=argparse.ArgumentParser(description="Audit Spower six-class inputs")
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def norm_key(frame: pd.DataFrame) -> pd.Series:
    return frame["OP"].astype(str).str.strip()+"||"+frame["Contingency"].astype(str).str.strip()


def main():
    args=parse_args(); root=args.data_path.expanduser().resolve()
    paths={
      "dataset_spower": first_existing([root/"Dataset"/"Dataset_Spower.csv",root/"Dataset_Spower.csv"],"Dataset_Spower.csv"),
      "dataset_voltage": first_existing([root/"Dataset"/"Dataset_Voltage.csv",root/"Dataset_Voltage.csv"],"Dataset_Voltage.csv"),
      "kpi_spower": first_existing([root/"KPI"/"KPI_spower.csv",root/"KPI_spower.csv"],"KPI_spower.csv"),
      "disc_spower": first_existing([root/"Disconnections"/"DISC_spower.csv",root/"DISC"/"DISC_spower.csv",root/"Dataset"/"DISC_spower.csv",root/"DISC_spower.csv"],"DISC_spower.csv"),
      "graphs": first_existing([root/"op_graphs"],"op_graphs"),
      "electric": first_existing([root/"op_electric_distance"],"op_electric_distance"),
    }
    ds=pd.read_csv(paths["dataset_spower"],low_memory=False)
    dv=pd.read_csv(paths["dataset_voltage"],low_memory=False)
    kp=pd.read_csv(paths["kpi_spower"],low_memory=False)
    dc=pd.read_csv(paths["disc_spower"],low_memory=False)
    for name,frame in [("Dataset_Spower",ds),("Dataset_Voltage",dv),("KPI_spower",kp),("DISC_spower",dc)]:
        missing={"OP","Contingency"}.difference(frame.columns)
        if missing: raise KeyError(f"{name} missing columns: {sorted(missing)}")
    target_cols=[c for c in ds.columns if c not in {"OP","Contingency"}]
    if not target_cols: raise RuntimeError("No Spower target columns found")
    for name,frame in [("KPI_spower",kp),("DISC_spower",dc)]:
        missing=set(target_cols).difference(frame.columns)
        if missing: raise KeyError(f"{name} missing targets: {sorted(missing)}")
    keys=norm_key(ds)
    if keys.duplicated().any(): raise RuntimeError("Duplicate OP-contingency rows in Dataset_Spower.csv")
    keyset=set(keys)
    checks={
      "Dataset_Voltage": set(norm_key(dv)),
      "KPI_spower": set(norm_key(kp)),
      "DISC_spower": set(norm_key(dc)),
    }
    key_mismatches={name:{"missing":len(keyset-other),"extra":len(other-keyset)} for name,other in checks.items()}
    if any(v["missing"] or v["extra"] for v in key_mismatches.values()):
        raise RuntimeError(f"OP-contingency key mismatch: {key_mismatches}")
    labels=ds[target_cols].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float)
    if np.isnan(labels).any(): raise RuntimeError(f"NaN class labels: {int(np.isnan(labels).sum())}")
    if not np.all(np.equal(labels,np.round(labels))): raise RuntimeError("Non-integer class labels found")
    if labels.min()<0 or labels.max()>5: raise RuntimeError(f"Class range is [{labels.min()}, {labels.max()}], expected 0..5")
    kpi=kp[target_cols].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float)
    disc=dc[target_cols].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(dtype=float)>0.5
    class5=labels==5
    mismatch=int(np.sum(disc!=class5))
    finite_activity=int(np.isfinite(kpi[~class5]).sum())
    activity_total=int((~class5).sum())
    finite_class5=int(np.isfinite(kpi[class5]).sum())
    ops=sorted(ds["OP"].astype(str).unique().tolist())
    missing_graph_ops=[op for op in ops if not (paths["graphs"] / f"{op}.pt").exists()]
    missing_electric_ops=[op for op in ops if not (paths["electric"] / f"{op}.csv").exists()]
    graph_files=sorted(paths["graphs"].glob("operating_point_*.pt"))
    electric_files=sorted(paths["electric"].glob("operating_point_*.csv"))
    report={
      "status":"ok" if mismatch==0 and finite_activity==activity_total and finite_class5==0 and not missing_graph_ops and not missing_electric_ops else "warning",
      "paths":{k:str(v) for k,v in paths.items()},
      "rows":int(len(ds)),
      "operating_points":ops,
      "targets":target_cols,
      "class_counts":{str(i):int((labels==i).sum()) for i in range(6)},
      "class5_disc_mismatches":mismatch,
      "activity_kpi_finite":finite_activity,
      "activity_targets":activity_total,
      "class5_kpi_finite_should_be_zero":finite_class5,
      "graph_file_count":len(graph_files),
      "electric_distance_file_count":len(electric_files),
      "missing_graph_ops":missing_graph_ops,
      "missing_electric_distance_ops":missing_electric_ops,
      "key_mismatches":key_mismatches,
    }
    print(json.dumps(report,indent=2))
    if args.output:
        args.output.expanduser().resolve().write_text(json.dumps(report,indent=2),encoding="utf-8")
    if mismatch or finite_activity!=activity_total or finite_class5!=0 or missing_graph_ops or missing_electric_ops:
        raise SystemExit(1)

if __name__=="__main__": main()
