#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


PAT_ACC = re.compile(r"Final Test Acc:\s*([0-9.]+)")
PAT_MIF = re.compile(r"Final miF:\s*([0-9.]+)")
PAT_MAF = re.compile(r"Final maF:\s*([0-9.]+)")
PAT_HEADER = re.compile(r"Namespace\((.+)\)")


def parse_one(path: Path) -> dict[str, str]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, str] = {"file": path.name, "framework": "", "backbone": "", "acc": "", "mif": "", "maf": ""}

    m = PAT_HEADER.search(txt)
    if m:
        header = m.group(1)
        for key in ("framework", "backbone", "simclr_contrastive", "pretrain_epochs", "lincls_epochs", "lr", "lr_cls", "batch_size"):
            mk = re.search(rf"{key}=('?[^,)]+'?|[^,)]*)", header)
            if mk:
                out[key] = mk.group(1).strip("'")

    for key, pat in (("acc", PAT_ACC), ("mif", PAT_MIF), ("maf", PAT_MAF)):
        m2 = pat.search(txt)
        if m2:
            out[key] = m2.group(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize SSL/SL run logs by final metrics.")
    ap.add_argument("--run_logs", type=str, default="run_logs")
    ap.add_argument("--pattern", type=str, default="*.out", help="glob under run_logs, e.g. 'slurm_*_simclr*.out'")
    args = ap.parse_args()

    run_dir = Path(args.run_logs)
    rows = [parse_one(p) for p in sorted(run_dir.glob(args.pattern))]
    rows = [r for r in rows if r.get("acc")]
    rows.sort(key=lambda r: float(r["acc"]), reverse=True)

    if not rows:
        print("No completed logs with 'Final Test Acc' matched.")
        return

    cols = ["file", "framework", "backbone", "simclr_contrastive", "batch_size", "lr", "lr_cls", "pretrain_epochs", "lincls_epochs", "acc", "mif", "maf"]
    widths = {c: max(len(c), max(len(r.get(c, "")) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(r.get(c, "").ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
