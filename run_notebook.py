import copy
import json
import os
import sys
from pathlib import Path

import nbformat
import plotly.io as pio
from jupyter_client import KernelManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
    os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".jupyter_runtime")
    path = ROOT / "Telco_Customer_Churn.ipynb"
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=1200,
        resources={"metadata": {"path": str(ROOT)}},
    )
    print(
        "Executing notebook; comparing classifiers on fixed customer splits.",
        flush=True,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)

    export_html(notebook)
    print(f"Executed notebook: {path}", flush=True)


def export_html(notebook):
    report = copy.deepcopy(notebook)
    include_library = True
    for cell in report.cells:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            plot = data.get("application/vnd.plotly.v1+json")
            if plot is None:
                continue
            figure = pio.from_json(json.dumps(plot))
            data["text/html"] = figure.to_html(
                full_html=False, include_plotlyjs=include_library
            )
            include_library = False
    html, _ = HTMLExporter().from_notebook_node(report)
    output = ROOT / "reports" / "notebook.html"
    output.write_text(html, encoding="utf-8")
    print(f"Browser-readable report: {output}", flush=True)


if __name__ == "__main__":
    main()
