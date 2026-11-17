"""Multiple sequence alignment for the Interaction Map Tool.

Uses MAFFT if it is installed. If not, uses a built-in center-star
alignment (plain Python, no extra packages, no AI).
"""
import os
import re
import shutil
import subprocess
import tempfile

MATCH = 3
MISMATCH = -1
GAP = -2


def clean_sequence(seq):
    """Keep letters only (removes spaces, gaps, * and digits)."""
    return re.sub(r"[^A-Za-z]", "", seq).upper()


def pairwise_align(a, b):
    """Global alignment (Needleman-Wunsch). Returns two gapped strings."""
    n, m = len(a), len(b)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]  # 1 diag, 2 up, 3 left
    for i in range(1, n + 1):
        score[i][0] = i * GAP
        back[i][0] = 2
    for j in range(1, m + 1):
        score[0][j] = j * GAP
        back[0][j] = 3
    for i in range(1, n + 1):
        ai = a[i - 1]
        row = score[i]
        prev = score[i - 1]
        for j in range(1, m + 1):
            s = MATCH if ai == b[j - 1] else MISMATCH
            d = prev[j - 1] + s
            u = prev[j] + GAP
            l = row[j - 1] + GAP
            if d >= u and d >= l:
                row[j] = d
                back[i][j] = 1
            elif u >= l:
                row[j] = u
                back[i][j] = 2
            else:
                row[j] = l
                back[i][j] = 3
    ga, gb = [], []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == 1:
            ga.append(a[i - 1])
            gb.append(b[j - 1])
            i -= 1
            j -= 1
        elif move == 2:
            ga.append(a[i - 1])
            gb.append("-")
            i -= 1
        else:
            ga.append("-")
            gb.append(b[j - 1])
            j -= 1
    return "".join(reversed(ga)), "".join(reversed(gb))


def center_star(seqs):
    """Align a list of sequences. Returns gapped sequences, same order."""
    k = len(seqs)
    if k == 1:
        return list(seqs)
    # choose the center: the sequence most similar to all the others
    best_total, center = None, 0
    for i in range(k):
        total = 0
        for j in range(k):
            if i != j:
                ga, gb = pairwise_align(seqs[i], seqs[j])
                total += sum(1 for x, y in zip(ga, gb) if x == y and x != "-")
        if best_total is None or total > best_total:
            best_total, center = total, i
    c = seqs[center]
    length = len(c)
    inserts = {}  # sequence index -> list (length+1) of insertion strings
    columns = {}  # sequence index -> list (length) of aligned characters
    for idx in range(k):
        if idx == center:
            continue
        gc, go = pairwise_align(c, seqs[idx])
        ins = [""] * (length + 1)
        col = ["-"] * length
        pos = 0
        for x, y in zip(gc, go):
            if x == "-":
                ins[pos] += y
            else:
                col[pos] = y
                pos += 1
        inserts[idx] = ins
        columns[idx] = col
    width = [0] * (length + 1)
    for ins in inserts.values():
        for p in range(length + 1):
            width[p] = max(width[p], len(ins[p]))
    out = []
    for idx in range(k):
        parts = []
        for p in range(length + 1):
            if idx == center:
                parts.append("-" * width[p])
            else:
                parts.append(inserts[idx][p].ljust(width[p], "-"))
            if p < length:
                parts.append(c[p] if idx == center else columns[idx][p])
        out.append("".join(parts))
    return out


def run_mafft(seqs):
    """Run MAFFT if available. Returns gapped sequences, or None."""
    exe = shutil.which("mafft")
    if exe is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "in.fasta")
        with open(path, "w") as f:
            for i, s in enumerate(seqs):
                f.write(">s%d\n%s\n" % (i, s))
        res = subprocess.run([exe, "--auto", "--quiet", path],
                             capture_output=True, text=True)
    if res.returncode != 0:
        return None
    found = {}
    name = None
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith(">"):
            name = line[1:]
            found[name] = ""
        elif name is not None:
            found[name] += line.upper()
    try:
        return [found["s%d" % i] for i in range(len(seqs))]
    except KeyError:
        return None


def align_sequences(named):
    """named: dict {name: sequence}. Returns (aligned dict, method)."""
    names = list(named.keys())
    seqs = [clean_sequence(named[n]) for n in names]
    if len(seqs) < 2 or any(len(s) == 0 for s in seqs):
        raise ValueError("Need at least two non-empty sequences.")
    aligned = run_mafft(seqs)
    method = "MAFFT"
    if aligned is None:
        aligned = center_star(seqs)
        method = "built-in center-star aligner"
    return dict(zip(names, aligned)), method


def to_fasta(aligned):
    return "".join(">%s\n%s\n" % (n, s) for n, s in aligned.items())
def conservation_scores(aligned):
    """Conservation of every alignment column.

    score = (1 - entropy / log2(20)) * (1 - gap fraction)
    1.0 = every sequence has the same residue, 0.0 = no conservation.
    Returns a list with one dictionary per column (column starts at 1).
    """
    import math
    seqs = list(aligned.values())
    count = len(seqs)
    max_entropy = math.log2(20)
    rows = []
    for col in range(len(seqs[0])):
        letters = [s[col] for s in seqs if s[col] != "-"]
        gap_fraction = 1 - len(letters) / count
        if letters:
            freq = {}
            for x in letters:
                freq[x] = freq.get(x, 0) + 1
            entropy = -sum((n / len(letters)) * math.log2(n / len(letters))
                           for n in freq.values())
            top = max(freq, key=freq.get)
            score = (1 - entropy / max_entropy) * (1 - gap_fraction)
        else:
            top, score = "-", 0.0
        rows.append({
            "column": col + 1,
            "most_common": top,
            "conservation": round(score, 3),
            "gap_fraction": round(gap_fraction, 3),
        })
    return rows
def residue_conservation(aligned, scores, sequence_name):
    """Map alignment conservation scores to individual residues."""
    sequence = aligned[sequence_name]
    result = {}
    residue_number = 0

    for column, amino_acid in enumerate(sequence):
        if amino_acid != "-":
            residue_number += 1
            result[residue_number] = scores[column]["conservation"]

    return result
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}


def map_conservation_to_pdb(sequence, scores, pdb_residues):
    """Map sequence conservation scores to actual PDB residue numbers."""
    fasta_seq = clean_sequence(sequence)

    pdb_seq = "".join(
        AA3_TO_1.get(resname, "X")
        for resnum, resname in pdb_residues
    )

    aligned_fasta, aligned_pdb = pairwise_align(fasta_seq, pdb_seq)

    seq_pos = 0
    pdb_pos = 0
    result = {}

    for aa, pdb_aa in zip(aligned_fasta, aligned_pdb):
        if aa != "-":
            seq_pos += 1
        if pdb_aa != "-":
            pdb_pos += 1

        if aa != "-" and pdb_aa != "-":
            result[pdb_residues[pdb_pos - 1][0]] = scores.get(seq_pos, 0)

    return result