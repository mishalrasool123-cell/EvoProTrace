#!/usr/bin/env python3
"""interaction_map.py

Residue-pair interaction map for a set of protein complexes.
No AI is used at run time. The same input always gives the same output.

Usage:
    python interaction_map.py --config config.json --out results
"""
import argparse
import glob
import json
import os
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
APOLAR = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP",
          "PRO", "TYR", "CYS"}
RING = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
}
SALT_CATION = {("LYS", "NZ"), ("ARG", "NE"),
               ("ARG", "NH1"), ("ARG", "NH2")}
SALT_ANION = {("ASP", "OD1"), ("ASP", "OD2"),
              ("GLU", "OE1"), ("GLU", "OE2")}
PICAT_CATION = {("LYS", "NZ"), ("ARG", "CZ")}

TYPES = ["hbond", "salt_bridge", "pi_cation", "hydrophobic"]
DEFAULTS = {
    "contact": 4.0, "hbond": 3.5, "salt_bridge": 4.0,
    "pi_cation": 6.0, "hydrophobic": 4.0, "pi_stacking": 5.5,
}
# Offsets of the four small boxes inside one cell (y axis is inverted,
# so (0, 0) is the top-left box).
OFFSETS = {"hbond": (0.0, 0.0), "salt_bridge": (0.5, 0.0),
           "pi_cation": (0.0, 0.5), "hydrophobic": (0.5, 0.5)}
CMAPS = {"hbond": "Blues", "salt_bridge": "Reds",
         "pi_cation": "Greens", "hydrophobic": "Oranges"}
FIXED_DOMAIN_COLORS = {"CDR1": "#F4B183", "CDR2": "#9DC3E6",
                       "CDR3": "#A9D18E"}
EMPTY = "#E8E8E8"


# ----------------------------------------------------------------- input
def read_pdb(path, chains):
    """Read heavy atoms of the wanted chains from the ATOM records."""
    out = {c: [] for c in chains}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[16] not in (" ", "A", "1"):
                continue
            chain = line[21]
            if chain not in out:
                continue
            name = line[12:16].strip()
            elem = line[76:78].strip().upper()
            if not elem:
                elem = name.lstrip("0123456789")[:1].upper()
            if elem in ("H", "D"):
                continue
            out[chain].append((
                int(line[22:26]), line[17:20].strip(), name, elem,
                float(line[30:38]), float(line[38:46]),
                float(line[46:54])))
    return out


class Side:
    """All atoms of one chain, as numpy arrays."""

    def __init__(self, atoms):
        self.num = np.array([a[0] for a in atoms], dtype=int)
        self.res = np.array([a[1] for a in atoms], dtype=object)
        self.name = np.array([a[2] for a in atoms], dtype=object)
        self.elem = np.array([a[3] for a in atoms], dtype=object)
        self.xyz = np.array([a[4:7] for a in atoms], dtype=float)
        self.resname = {}
        for n, r in zip(self.num, self.res):
            self.resname.setdefault(int(n), str(r))

    def mask(self, fn):
        return np.array([fn(r, n, e) for r, n, e in
                         zip(self.res, self.name, self.elem)], dtype=bool)

    def rings(self):
        """Ring centres of Phe, Tyr and Trp residues."""
        out = []
        for n in sorted(set(self.num.tolist())):
            sel = self.num == n
            r = str(self.res[sel][0])
            if r not in RING:
                continue
            keep = np.isin(self.name[sel], RING[r])
            if keep.sum() >= 5:
                out.append((n, self.xyz[sel][keep].mean(axis=0)))
        return out


def near(a_xyz, a_num, b_xyz, b_num, cutoff):
    """Residue-number pairs with any atom pair within cutoff."""
    if len(a_xyz) == 0 or len(b_xyz) == 0:
        return set()
    i, j = np.where(cdist(a_xyz, b_xyz) <= cutoff)
    return set(zip(a_num[i].tolist(), b_num[j].tolist()))


def sub(side, mask):
    return side.xyz[mask], side.num[mask]


def ring_pairs(ra, rb, cutoff):
    if not ra or not rb:
        return set()
    xa = np.array([c for _, c in ra])
    xb = np.array([c for _, c in rb])
    i, j = np.where(cdist(xa, xb) <= cutoff)
    return {(ra[a][0], rb[b][0]) for a, b in zip(i, j)}


def ring_vs_atoms(rings, side, mask, cutoff):
    """Ring centre of one side versus selected atoms of the other."""
    if not rings or not mask.any():
        return set()
    rc = np.array([c for _, c in rings])
    i, j = np.where(cdist(rc, side.xyz[mask]) <= cutoff)
    nums = side.num[mask]
    return {(rings[a][0], int(nums[b])) for a, b in zip(i, j)}


def measure(s1, s2, th, require_contact):
    """Return {(num1, num2): set of interaction types}."""
    found = {}

    def add(pairs, kind):
        for p in pairs:
            found.setdefault(p, set()).add(kind)

    # hydrogen bonds: N or O atoms
    m1 = np.isin(s1.elem, ["N", "O"])
    m2 = np.isin(s2.elem, ["N", "O"])
    add(near(*sub(s1, m1), *sub(s2, m2), th["hbond"]), "hbond")

    # salt bridges: charged atoms of opposite sign
    def pick(side, table):
        return side.mask(lambda r, n, e: (r, n) in table)
    c1, a1 = pick(s1, SALT_CATION), pick(s1, SALT_ANION)
    c2, a2 = pick(s2, SALT_CATION), pick(s2, SALT_ANION)
    add(near(*sub(s1, c1), *sub(s2, a2), th["salt_bridge"]),
        "salt_bridge")
    add(near(*sub(s1, a1), *sub(s2, c2), th["salt_bridge"]),
        "salt_bridge")

    # pi-cation: ring centre near Lys NZ or Arg CZ of the other side
    r1, r2 = s1.rings(), s2.rings()
    p1, p2 = pick(s1, PICAT_CATION), pick(s2, PICAT_CATION)
    add(ring_vs_atoms(r1, s2, p2, th["pi_cation"]), "pi_cation")
    add({(b, a) for a, b in ring_vs_atoms(r2, s1, p1, th["pi_cation"])},
        "pi_cation")

    # hydrophobic: side-chain carbons of apolar residues,
    # plus ring centre to ring centre (pi-stacking)
    def hyd(r, n, e):
        return r in APOLAR and e == "C" and n not in ("C", "CA")
    h1, h2 = s1.mask(hyd), s2.mask(hyd)
    add(near(*sub(s1, h1), *sub(s2, h2), th["hydrophobic"]),
        "hydrophobic")
    add(ring_pairs(r1, r2, th["pi_stacking"]), "hydrophobic")

    if require_contact:
        ok = near(s1.xyz, s1.num, s2.xyz, s2.num, th["contact"])
        found = {p: t for p, t in found.items() if p in ok}
    return found


# --------------------------------------------------------------- tables
def one_letter(res):
    return AA3.get(res, res[:1])


def load_labels(path):
    if not path:
        return {}, {}
    df = pd.read_csv(path)
    lab, dom = {}, {}
    for row in df.itertuples(index=False):
        n = int(row.residue_number)
        label = getattr(row, "label", "")
        domain = getattr(row, "domain", "")
        lab[n] = "" if pd.isna(label) else str(label)
        dom[n] = "" if pd.isna(domain) else str(domain)
    return lab, dom


def choose_files(cfg, base):
    sdir = os.path.join(base, cfg["structures_dir"])
    pattern = cfg.get("file_pattern", "*.pdb")
    files = sorted(glob.glob(os.path.join(sdir, pattern)))
    rk = cfg.get("ranking")
    if not rk:
        return files
    path = os.path.join(base, rk["file"])
    if path.lower().endswith((".xlsx", ".xls")):
        table = pd.read_excel(path)
    else:
        table = pd.read_csv(path)
    table = table.sort_values(rk["score_column"],
                              ascending=rk.get("ascending", True))
    ids = table[rk["id_column"]].astype(str).tolist()
    top_n = rk.get("top_n")
    chosen = []
    for cid in ids:
        hit = [f for f in files if os.path.basename(f).startswith(cid)]
        if hit:
            chosen.append(hit[0])
        else:
            print("  note: no structure file for", cid)
        if top_n and len(chosen) >= top_n:
            break
    return chosen


def run_measurements(files, cfg, th):
    c1, c2 = cfg["side1"]["chain"], cfg["side2"]["chain"]
    rows, names1, names2 = [], {}, {}
    used = []
    for path in files:
        sid = os.path.basename(path).rsplit(".", 1)[0]
        atoms = read_pdb(path, [c1, c2])
        if not atoms[c1] or not atoms[c2]:
            print("  skipped (chain missing):", sid)
            continue
        s1, s2 = Side(atoms[c1]), Side(atoms[c2])
        for n, r in s1.resname.items():
            names1.setdefault(n, []).append(r)
        for n, r in s2.resname.items():
            names2.setdefault(n, []).append(r)
        found = measure(s1, s2, th, cfg.get("require_contact", False))
        for (n1, n2), kinds in sorted(found.items()):
            row = {"structure": sid,
                   "side1_number": n1, "side1_residue": s1.resname[n1],
                   "side2_number": n2, "side2_residue": s2.resname[n2]}
            for k in TYPES:
                row[k] = k in kinds
            rows.append(row)
        used.append(sid)
    cols = ["structure", "side1_number", "side1_residue",
            "side2_number", "side2_residue"] + TYPES
    return pd.DataFrame(rows, columns=cols), names1, names2, used


def summarize(pairs):
    pairs = pairs.copy()
    pairs["any"] = pairs[TYPES].any(axis=1)
    g = pairs.groupby(["side1_number", "side2_number"])
    summ = g.agg(n_any=("any", "sum"), hbond=("hbond", "sum"),
                 salt_bridge=("salt_bridge", "sum"),
                 pi_cation=("pi_cation", "sum"),
                 hydrophobic=("hydrophobic", "sum")).reset_index()
    return summ


# --------------------------------------------------------------- drawing
def common(names, n):
    return Counter(names[n]).most_common(1)[0][0]


def domain_color(name, table):
    if name in FIXED_DOMAIN_COLORS:
        return FIXED_DOMAIN_COLORS[name]
    if name.upper().startswith("FR") or name == "":
        return "#DADADA"
    if name not in table:
        pal = plt.get_cmap("tab20").colors
        table[name] = pal[len(table) % len(pal)]
    return table[name]


def runs(domains):
    """Consecutive blocks of the same domain: (name, start, length)."""
    out, start = [], 0
    for i in range(1, len(domains) + 1):
        if i == len(domains) or domains[i] != domains[start]:
            out.append((domains[start], start, i - start))
            start = i
    return out


def draw(summ, names1, names2, lab1, dom1, lab2, dom2, cfg,
         n_struct, out_png, out_svg):
    min_c = cfg.get("min_complexes", 1)
    keep = summ[summ["n_any"] >= min_c]
    if keep.empty:
        print("No residue pair reaches min_complexes; nothing to draw.")
        return False
    rows = sorted(keep["side1_number"].unique())
    cols = sorted(keep["side2_number"].unique())
    ri = {n: i for i, n in enumerate(rows)}
    ci = {n: i for i, n in enumerate(cols)}

    def tick(n, names, lab):
        text = one_letter(common(names, n)) + str(n)
        return text + (" | " + lab[n] if lab.get(n) else "")

    ytick = [tick(n, names1, lab1) for n in rows]
    xtick = [tick(n, names2, lab2) for n in cols]
    d1 = [dom1.get(n, "") for n in rows]
    d2 = [dom2.get(n, "") for n in cols]
    has_left = any(d1)
    has_top = any(d2)

    sw = 1.5 if has_left else 0.0
    sh = 1.2 if has_top else 0.0
    fig_w = max(6.0, 0.55 * len(cols) + 4.5)
    fig_h = max(4.0, 0.5 * len(rows) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmaps = {k: plt.get_cmap(v) for k, v in CMAPS.items()}

    for r in range(len(rows)):
        for c in range(len(cols)):
            for k in TYPES:
                dx, dy = OFFSETS[k]
                ax.add_patch(Rectangle((c + dx, r + dy), 0.5, 0.5,
                                       fc=EMPTY, ec="white", lw=0.8))
    for row in keep.itertuples(index=False):
        r, c = ri[row.side1_number], ci[row.side2_number]
        for k in TYPES:
            count = int(getattr(row, k))
            if count == 0:
                continue
            frac = count / n_struct
            dx, dy = OFFSETS[k]
            ax.add_patch(Rectangle(
                (c + dx, r + dy), 0.5, 0.5,
                fc=cmaps[k](0.30 + 0.45 * frac), ec="white", lw=0.8))
            ax.text(c + dx + 0.25, r + dy + 0.25, str(count),
                    ha="center", va="center", fontsize=6,
                    color="white" if frac >= 0.4 else "black")

    table = {}
    if has_left:
        for name, start, length in runs(d1):
            ax.add_patch(Rectangle((-sw, start), sw - 0.1, length,
                                   fc=domain_color(name, table),
                                   ec="white", lw=1.0))
            ax.text(-sw / 2 - 0.05, start + length / 2, name,
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", rotation=90 if length < 3 else 0)
    if has_top:
        for name, start, length in runs(d2):
            ax.add_patch(Rectangle((start, -sh), length, sh - 0.1,
                                   fc=domain_color(name, table),
                                   ec="white", lw=1.0))
            ax.text(start + length / 2, -sh / 2 - 0.05, name,
                    ha="center", va="center", fontsize=8,
                    fontweight="bold")

    ax.set_xlim(-sw, len(cols))
    ax.set_ylim(len(rows), -sh)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(xtick, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(ytick, fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(cfg["side2"].get("name", "Side 2") + " residue",
                  fontsize=10)
    ylabel = cfg["side1"].get("name", "Side 1") + " residue"
    if any(lab1.values()):
        ylabel += " | numbering label"
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title("Interaction types per residue pair, %d complexes\n"
                 "(number in a box = complexes with that interaction;"
                 " darker = more complexes)" % n_struct, fontsize=10)

    handles = [
        Patch(fc=cmaps["hbond"](0.7), label="top-left: hydrogen bond"),
        Patch(fc=cmaps["salt_bridge"](0.7),
              label="top-right: salt bridge"),
        Patch(fc=cmaps["pi_cation"](0.7), label="bottom-left: pi-cation"),
        Patch(fc=cmaps["hydrophobic"](0.7),
              label="bottom-right: hydrophobic + pi-stacking"),
        Patch(fc=EMPTY, label="grey: not found"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_svg)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    base = os.path.dirname(cfg_path)
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    out = args.out or cfg.get("output_dir", "results")
    out = out if os.path.isabs(out) else os.path.join(base, out)
    os.makedirs(out, exist_ok=True)

    th = dict(DEFAULTS)
    th.update(cfg.get("thresholds", {}))

    files = choose_files(cfg, base)
    print("Structures selected:", len(files))
    if not files:
        sys.exit("No structure files found. Check structures_dir.")

    pairs, names1, names2, used = run_measurements(files, cfg, th)
    if pairs.empty:
        sys.exit("No interactions found in any structure.")
    summ = summarize(pairs)

    def lab_path(side):
        p = cfg[side].get("labels")
        return os.path.join(base, p) if p else None

    lab1, dom1 = load_labels(lab_path("side1"))
    lab2, dom2 = load_labels(lab_path("side2"))

    pairs.to_csv(os.path.join(out, "interaction_pairs.csv"), index=False)
    summ.to_csv(os.path.join(out, "interaction_summary.csv"), index=False)
    ok = draw(summ, names1, names2, lab1, dom1, lab2, dom2, cfg,
              len(used), os.path.join(out, "interaction_map.png"),
              os.path.join(out, "interaction_map.svg"))

    # keep everything needed to repeat the run
    shutil.copy(cfg_path, os.path.join(out, "config_used.json"))
    for side in ("side1", "side2"):
        p = lab_path(side)
        if p and os.path.exists(p):
            shutil.copy(p, os.path.join(out, "labels_" + side + ".csv"))
    with open(os.path.join(out, "run_info.txt"), "w") as fh:
        fh.write("date: %s\n" % datetime.now().isoformat(timespec="seconds"))
        fh.write("python: %s\n" % platform.python_version())
        fh.write("numpy: %s\n" % np.__version__)
        fh.write("pandas: %s\n" % pd.__version__)
        fh.write("structures used (%d):\n" % len(used))
        for s in used:
            fh.write("  " + s + "\n")
    print("Done. Results in:", out)
    if ok:
        print("Map: interaction_map.png and interaction_map.svg")


if __name__ == "__main__":
    main()
