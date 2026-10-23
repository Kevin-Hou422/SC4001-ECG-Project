"""Execute portable E6 notebook cells using the published encoders; not hosted Colab."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    import matplotlib
    matplotlib.use("Agg")
    notebook = json.loads((root / "notebooks/02_M1_Autoencoder_Pretraining.ipynb").read_text(encoding="utf-8"))
    compiled, executed = [], []
    namespace = {"ROOT": root, "PROTOCOL": "patient", "SEEDS": [42, 43, 44],
                 "EPOCHS": 10, "PROCESSED": root / "data/processed",
                 "Path": Path, "sys": sys, "json": json, "subprocess": subprocess}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        bytecode = compile(source, f"pretraining_notebook_cell_{index}", "exec")
        compiled.append(index)
        if any(token in source for token in ("audits = []", "RUN_PRETRAINING = False",
                                             "selected_seed =", "run_record =")):
            exec(bytecode, namespace)
            executed.append(index)
    report = {"status": "passed", "compiled_code_cells": compiled,
              "executed_portable_cells": executed, "hosted_google_colab_run": False,
              "published_exports_audited": 3, "pretraining_rerun_in_this_check": False,
              "feature_map_shape": list(namespace["feature_maps"].shape),
              "embedding_shape": list(namespace["embeddings"].shape)}
    (root / "docs/pretrain_notebook_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    import matplotlib.pyplot as plt
    plt.close("all")
    print("E6 portable notebook validation passed.")


if __name__ == "__main__":
    main()
