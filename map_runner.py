"""Helper functions for the app: build the config, run the map script,
make label tables. No Streamlit code here, so it can be tested alone."""
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from io import BytesIO

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_SCRIPT = os.path.join(HERE, "interaction_map.py")
LABEL_SCRIPT = os.path.join(HERE, "make_labels.py")

MODES = {
    "pp": "Protein vs protein (two chains)",
    "intra": "Inside one protein (intra-protein)",
    "ligand": "Protein vs ligand (small molecule)",
}

DEFAULT_THRESHOLDS = {
    "contact": 4.0,
    "hbond": 3.5,
    "salt_bridge": 4.0,
    "pi_cation": 6.0,
    "hydrophobic": 4.0,
    "pi_stacking": 5.0,
}


def make_label_table(domain_bytes, pdb_bytes, chain, out_path):
    """Run make_labels.py. Returns (True, message) or (False, message)."""
    with tempfile.TemporaryDirectory() as tmp:
        dom = os.path.join(tmp, "domains.csv")
        pdb = os.path.join(tmp, "structure.pdb")
        with open(dom, "wb") as f:
            f.write(domain_bytes)
        cmd = [sys.executable, LABEL_SCRIPT, "--domains", dom,
               "--out", out_path]
        if pdb_bytes:
            with open(pdb, "wb") as f:
                f.write(pdb_bytes)
            cmd += ["--pdb", pdb, "--chain", chain]
        res = subprocess.run(cmd, capture_output=True, text=True)
    text = (res.stdout or "") + (res.stderr or "")
    return res.returncode == 0, text.strip()


def run_map(structures, mode, side1, side2, thresholds, min_complexes,
            min_sep=3, require_contact=False, label_files=None):
    """structures: list of (file name, bytes).
    side1: {"chain", "name", "use_labels"}
    side2: {"chain", "name", "use_labels"} for pp, or {"resname", "chain",
           "name"} for ligand, or None for intra.
    label_files: {"side1": (name, bytes) or None, "side2": ...}
    Returns a dict with ok, error, log, png, svg, tables, zip."""
    label_files = label_files or {}
    out = {"ok": False, "error": "", "log": "", "png": None, "svg": None,
           "tables": {}, "zip": None}
    with tempfile.TemporaryDirectory() as temp:
        sdir = os.path.join(temp, "structures")
        os.makedirs(sdir)
        for name, data in structures:
            with open(os.path.join(sdir, os.path.basename(name)), "wb") as f:
                f.write(data)

        def label_path(key):
            item = label_files.get(key)
            if not item:
                return None
            path = os.path.join(temp, key + "_labels.csv")
            with open(path, "wb") as f:
                f.write(item[1])
            return path

        s1 = {"chain": side1["chain"], "name": side1["name"],
              "use_labels": side1.get("use_labels", "residue_label")}
        p1 = label_path("side1")
        if p1:
            s1["residue_labels"] = p1

        cfg = {
            "structures_dir": "structures",
            "file_pattern": "*.pdb",
            "ranking": None,
            "side1": s1,
            "thresholds": dict(thresholds),
            "min_complexes": int(min_complexes),
            "require_contact": bool(require_contact),
            "output_dir": "results",
        }
        if mode == "pp":
            s2 = {"chain": side2["chain"], "name": side2["name"],
                  "use_labels": side2.get("use_labels", "residue_label")}
            p2 = label_path("side2")
            if p2:
                s2["residue_labels"] = p2
            cfg["side2"] = s2
        elif mode == "ligand":
            lig = {"resname": side2["resname"]}
            if side2.get("chain"):
                lig["chain"] = side2["chain"]
            cfg["side2"] = {"ligand": lig, "name": side2["name"]}
        else:
            cfg["min_sequence_separation"] = int(min_sep)

        config_path = os.path.join(temp, "config.json")
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        output = os.path.join(temp, "results")
        res = subprocess.run(
            [sys.executable, MAP_SCRIPT, "--config", config_path,
             "--out", output],
            capture_output=True, text=True)
        out["log"] = (res.stdout or "") + (res.stderr or "")
        if res.returncode != 0:
            out["error"] = (res.stderr or res.stdout or "Unknown error").strip()
            return out
        if not os.path.isdir(output):
            out["error"] = "No results folder was made."
            return out
        for fname in os.listdir(output):
            path = os.path.join(output, fname)
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as f:
                data = f.read()
            if fname == "interaction_map.png":
                out["png"] = data
            elif fname == "interaction_map.svg":
                out["svg"] = data
            elif fname.endswith(".csv"):
                out["tables"][fname] = data
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for fname in os.listdir(output):
                path = os.path.join(output, fname)
                if os.path.isfile(path):
                    z.write(path, fname)
        out["zip"] = buf.getvalue()
        out["ok"] = True
    return out
