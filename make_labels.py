"""Make a residue label table (labels.csv) from a domain table.

Domain table (CSV) columns: Sub_Domain, Domain, Start_Resid, End_Resid
Output columns: residue_number, residue_name, residue_label,
                sub_domain, domain

residue_label = sub-domain name + position inside it, e.g. H1.3
Residues outside every listed range get an empty label and domain.

Usage:
  python make_labels.py --domains domains.csv --pdb protein.pdb \
         --chain A --out labels.csv
  (without --pdb it covers the residue numbers of the domain table)
"""
import argparse
import csv
import sys

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
       "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
       "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
       "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def read_domains(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        need = {"Sub_Domain", "Domain", "Start_Resid", "End_Resid"}
        if not reader.fieldnames or not need.issubset(
                {h.strip() for h in reader.fieldnames}):
            sys.exit("Domain table needs the columns: " + ", ".join(
                sorted(need)))
        for line, row in enumerate(reader, start=2):
            row = {k.strip(): (v or "").strip() for k, v in row.items()}
            if not row["Sub_Domain"]:
                continue
            try:
                start = int(row["Start_Resid"])
                end = int(row["End_Resid"])
            except ValueError:
                sys.exit("Line %d: Start_Resid and End_Resid must be "
                         "whole numbers." % line)
            if start > end:
                sys.exit("Line %d (%s): start %d is after end %d." % (
                    line, row["Sub_Domain"], start, end))
            rows.append((row["Sub_Domain"], row["Domain"], start, end))
    seen = set()
    for sub, dom, start, end in rows:
        if sub in seen:
            sys.exit("Sub_Domain '%s' appears twice." % sub)
        seen.add(sub)
    ordered = sorted(rows, key=lambda r: r[2])
    for a, b in zip(ordered, ordered[1:]):
        if b[2] <= a[3]:
            sys.exit("Ranges overlap: %s (%d-%d) and %s (%d-%d)." % (
                a[0], a[2], a[3], b[0], b[2], b[3]))
    return rows


def read_residues(pdb, chain):
    found = {}
    with open(pdb) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            if line[16] not in (" ", "A", "1"):
                continue
            number = int(line[22:26])
            if number not in found:
                found[number] = AA3.get(line[17:20].strip(), "X")
    if not found:
        sys.exit("No ATOM records for chain '%s' in %s." % (chain, pdb))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--pdb")
    ap.add_argument("--chain")
    ap.add_argument("--out", default="labels.csv")
    args = ap.parse_args()

    domains = read_domains(args.domains)
    if args.pdb:
        if not args.chain:
            sys.exit("--chain is required with --pdb.")
        residues = read_residues(args.pdb, args.chain)
    else:
        residues = {n: "X" for _, _, s, e in domains
                    for n in range(s, e + 1)}

    where = {}
    for sub, dom, start, end in domains:
        for n in range(start, end + 1):
            where[n] = (sub, dom, n - start + 1)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["residue_number", "residue_name", "residue_label",
                    "sub_domain", "domain"])
        for n in sorted(residues):
            if n in where:
                sub, dom, k = where[n]
                w.writerow([n, residues[n], "%s.%d" % (sub, k), sub, dom])
            else:
                w.writerow([n, residues[n], "", "", ""])

    missing = [n for n in where if n not in residues]
    print("Wrote %d residues to %s" % (len(residues), args.out))
    if missing:
        print("Note: %d residue numbers in the domain table are not in "
              "the structure (for example %d)." % (len(missing),
                                                   min(missing)))


if __name__ == "__main__":
    main()
