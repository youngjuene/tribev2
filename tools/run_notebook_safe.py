#!/usr/bin/env python3
"""Execute the Mac notebook in its safe/probe modes with bounded timeouts."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=Path("tribe_demo_mac_m2.ipynb"))
    parser.add_argument(
        "--mode",
        default="safe",
        choices=["safe", "mlx_probe", "mlx_hybrid", "mps_experimental", "mlx_vjepa_probe", "mlx_vjepa_predict"],
    )
    parser.add_argument("--kernel-name", default="tribev2-mac")
    parser.add_argument("--timeout-per-cell", type=int, default=60)
    parser.add_argument("--max-total-seconds", type=int, default=180)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    os.environ["TRIBEV2_NOTEBOOK_MODE"] = args.mode
    os.environ.setdefault("TRIBEV2_SKIP_LONG_VJEPA", "1")

    nb = nbformat.read(args.notebook, as_version=4)
    client = NotebookClient(
        nb,
        kernel_name=args.kernel_name,
        timeout=args.timeout_per_cell,
        allow_errors=False,
        force_raise_errors=True,
        resources={"metadata": {"path": str(Path.cwd())}},
    )
    start = time.perf_counter()
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if time.perf_counter() - start > args.max_total_seconds:
                raise TimeoutError(f"Notebook exceeded {args.max_total_seconds}s")
            if cell.cell_type != "code":
                continue
            client.execute_cell(cell, i)
    if args.output:
        nbformat.write(nb, args.output)
    print(f"Executed {args.notebook} in {time.perf_counter() - start:.2f}s (mode={args.mode})")


if __name__ == "__main__":
    main()
