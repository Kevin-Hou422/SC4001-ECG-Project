"""Compile all notebook cells and execute portable data cells on real data.

This does not represent a Google-hosted Colab run. Cloning/dependency install,
interactive Google authentication and Drive mounting are outside this check.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/notebook_validation.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    import matplotlib
    matplotlib.use("Agg")
    notebook = json.loads((root / "notebooks/01_M1_Data_Setup.ipynb").read_text(encoding="utf-8"))
    namespace = {"ROOT": root, "Path": Path, "sys": sys, "subprocess": subprocess,
                 "display": lambda table: print(table.to_string(index=False))}
    executed, compiled = [], []
    # Select by content instead of fragile cell numbers. These are repo-owned
    # Python cells, not arbitrary code obtained from the browser.
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compiled.append(index)
        bytecode = compile(source, f"notebook_cell_{index}", "exec")
        if any(token in source for token in (
                "SEED = 42", "hashes = verify_raw", "manifest_path =",
                "display(pd.read_csv", "train_dataset =", "BACKUP_TO_DRIVE =")):
            exec(bytecode, namespace)
            executed.append(index)
    import matplotlib.pyplot as plt
    plt.gcf().savefig(root / "docs/data_preview.png", dpi=130)
    plt.close("all")
    report = {"status": "passed", "python": platform.python_version(),
              "protocol": namespace["PROTOCOL"],
              "platform": platform.system(), "compiled_code_cells": compiled,
              "executed_portable_cells": executed,
              "raw_files_verified": len(namespace["hashes"]),
              "raw_bytes": namespace["raw_bytes"],
              "pytorch_batch_shapes": {key: list(value.shape)
                                       for key, value in namespace["batch"].items()},
              "hosted_google_colab_run": False,
              "not_executed": ["Google login/runtime connection", "Drive mount/backup",
                               "clone/install cells (environment provisioned separately)"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Portable notebook validation passed:", args.output)


if __name__ == "__main__":
    main()
