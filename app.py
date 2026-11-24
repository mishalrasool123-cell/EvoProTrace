import streamlit as st
import tempfile
import subprocess
import json
import os
import zipfile
from io import BytesIO

import pandas as pd
import numpy as np
import py3Dmol
import streamlit.components.v1 as components
from scipy.spatial.distance import cdist

from map_runner import MODES, make_label_table, run_map

from msa_tools import (
    align_sequences,
    to_fasta,
    conservation_scores,
    residue_conservation,
    map_conservation_to_pdb
)


st.set_page_config(
    page_title="EvoProTrace",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<div style="background:linear-gradient(135deg,#F3EEFA 0%,#E6F4F5 100%);border:1px solid #D9E2EC;border-radius:16px;padding:22px 28px;display:flex;align-items:center;justify-content:center;gap:26px;flex-wrap:wrap;margin-bottom:8px;">
<svg width="110" height="96" viewBox="0 0 110 96" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="g1" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#008C95"/><stop offset="1" stop-color="#5B8DEF"/></linearGradient>
<linearGradient id="g2" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#8E6BBF"/><stop offset="1" stop-color="#C9B6E8"/></linearGradient>
</defs>
<path d="M10 62 C 22 22, 36 88, 52 52 S 80 20, 100 40" fill="none" stroke="url(#g2)" stroke-width="11" stroke-linecap="round" opacity="0.55"/>
<path d="M10 54 C 22 14, 36 80, 52 44 S 80 12, 100 32" fill="none" stroke="url(#g1)" stroke-width="11" stroke-linecap="round"/>
<circle cx="24" cy="84" r="5" fill="#8E6BBF"/><circle cx="56" cy="74" r="5" fill="#008C95"/>
<circle cx="84" cy="70" r="5" fill="#5B8DEF"/><circle cx="100" cy="62" r="4" fill="#8E6BBF"/>
</svg>
<div style="text-align:left;">
<div style="font-size:64px;font-weight:800;line-height:1.05;letter-spacing:-1px;color:#008C95;">Evo<span style="color:#5B3FA0;">Pro</span>Trace</div>
<div style="font-size:19px;color:#4A5568;margin-top:6px;">Residue-level interaction maps, sequence conservation and 3D structure in one place</div>
<div style="margin-top:10px;">
<span style="background:#fff;border:1px solid #D9E2EC;border-radius:20px;padding:3px 12px;font-size:14px;color:#2D3748;margin-right:6px;">🧬 Interaction maps</span>
<span style="background:#fff;border:1px solid #D9E2EC;border-radius:20px;padding:3px 12px;font-size:14px;color:#2D3748;margin-right:6px;">🔬 Conservation</span>
<span style="background:#fff;border:1px solid #D9E2EC;border-radius:20px;padding:3px 12px;font-size:14px;color:#2D3748;">🧪 3D viewer</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 1. UPLOAD PDB STRUCTURES
# --------------------------------------------------

st.subheader("1. Upload PDB structures")

files = st.file_uploader(
    "Upload one or more PDB files",
    type=["pdb"],
    accept_multiple_files=True
)

n_files = len(files) if files else 0

if n_files:
    st.caption(f"{n_files} structure(s) uploaded.")

# --------------------------------------------------
# 2. CHOOSE THE ANALYSIS
# --------------------------------------------------

st.subheader("2. Choose the analysis")

mode_label = st.radio(
    "What do you want to map?",
    list(MODES.values()),
    horizontal=True
)

mode = [key for key, text in MODES.items() if text == mode_label][0]

chain2 = None
name2 = None
lig_resname = ""
lig_chain = ""
min_sep = 3

col1, col2 = st.columns(2)

with col1:
    chain1 = st.text_input(
        "First chain ID" if mode == "pp" else "Protein chain ID",
        "A"
    ).strip()
    name1 = st.text_input(
        "First protein name" if mode == "pp" else "Protein name",
        "Nanobody" if mode == "pp" else "Protein"
    )

with col2:
    if mode == "pp":
        chain2 = st.text_input("Second chain ID", "B").strip()
        name2 = st.text_input("Second protein name", "Target")

    elif mode == "ligand":
        lig_resname = st.text_input(
            "Ligand name in the PDB file (for example FMN)",
            ""
        ).strip().upper()
        lig_chain = st.text_input(
            "Ligand chain ID (optional)",
            ""
        ).strip()
        name2 = lig_resname or "Ligand"

    else:
        min_sep = st.number_input(
            "Ignore pairs that are closer than this in the sequence",
            min_value=1,
            max_value=50,
            value=3,
            help="Residues next to each other in the chain always touch. "
                 "This hides those trivial pairs."
        )

if mode == "ligand":
    st.caption(
        "The ligand is read from HETATM lines. Columns of the map are "
        "ligand atoms, matched by atom name across structures."
    )

# --------------------------------------------------
# 3. INTERACTION SETTINGS
# --------------------------------------------------

st.subheader("3. Interaction settings")

col1, col2, col3 = st.columns(3)

with col1:
    hbond = st.number_input("Hydrogen bond (Å)", value=3.5)
    salt = st.number_input("Salt bridge (Å)", value=4.0)

with col2:
    picat = st.number_input("Pi-cation (Å)", value=6.0)
    hydro = st.number_input("Hydrophobic (Å)", value=4.0)

with col3:
    pistack = st.number_input("Pi-stacking (Å)", value=5.0)
    contact = st.number_input("Contact (Å)", value=4.0)

col1, col2 = st.columns(2)

with col1:
    min_complexes = st.number_input(
        "Minimum complexes",
        min_value=1,
        max_value=max(1, n_files),
        value=max(1, n_files),
        key=f"min_complexes_{n_files}",
        help="A residue pair is drawn only if it is found in at least "
             "this many structures. It follows the number of uploaded "
             "files, and you can lower it."
    )

with col2:
    require_contact = st.checkbox(
        "Show a pair only if the residues also touch (contact cutoff)",
        value=False
    )

# --------------------------------------------------
# 4. RESIDUE LABELS AND DOMAINS (OPTIONAL)
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    pistack_angle = st.number_input(
        "Pi-stacking angle limit (degrees, 0 = off)",
        min_value=0.0,
        max_value=90.0,
        value=30.0,
        help="Rings count as stacked only if their flat faces are "
             "parallel (tilt up to this value) or, if the box on the "
             "right is ticked, T-shaped (tilt of 90 minus this value "
             "or more)."
    )

with col2:
    pistack_tshape = st.checkbox(
        "Also count T-shaped (perpendicular) rings",
        value=True
    )

st.subheader("4. Residue labels and domains (optional)")

label_files = {}
generated_downloads = []


def get_labels(key, title, chain):
    with st.expander(title):
        label_up = st.file_uploader(
            "Label table (CSV with residue_number, residue_name, "
            "residue_label, domain)",
            type=["csv"],
            key=key + "_labels"
        )
        domain_up = st.file_uploader(
            "or a domain table (CSV with Sub_Domain, Domain, "
            "Start_Resid, End_Resid): labels are made for you",
            type=["csv"],
            key=key + "_domains"
        )

        if label_up is not None:
            label_files[key] = (label_up.name, label_up.getvalue())

        elif domain_up is not None:
            pdb_bytes = files[0].getvalue() if files else None
            handle, path = tempfile.mkstemp(suffix=".csv")
            os.close(handle)
            ok, message = make_label_table(
                domain_up.getvalue(), pdb_bytes, chain, path
            )
            if ok:
                with open(path, "rb") as f:
                    data = f.read()
                label_files[key] = ("labels.csv", data)
                st.success(message)
                generated_downloads.append((key, data))
            else:
                st.error(message)
            os.remove(path)


get_labels("side1", f"Labels for {name1 or 'first protein'}", chain1)

if mode == "pp":
    get_labels("side2", f"Labels for {name2 or 'second protein'}", chain2)

use_labels_text = st.radio(
    "Text on the map axes",
    ["Domain labels (for example H1.3)", "Residue letter + number"],
    horizontal=True
)
use_labels = (
    "residue_label" if use_labels_text.startswith("Domain")
    else "residue_number"
)

for key, data in generated_downloads:
    st.download_button(
        f"Download the made labels ({key})",
        data=data,
        file_name=f"labels_{key}.csv",
        mime="text/csv",
        key="dl_" + key
    )

# --------------------------------------------------
# 5. GENERATE INTERACTION MAP
# --------------------------------------------------

if st.button("Generate interaction map", type="primary"):

    if not files:
        st.error("Please upload at least one PDB file.")

    elif not chain1:
        st.error("Please enter the chain ID.")

    elif mode == "pp" and (not chain2 or chain1 == chain2):
        st.error("Please enter two different chain IDs.")

    elif mode == "ligand" and not lig_resname:
        st.error("Please enter the ligand name (for example FMN).")

    else:
        side1 = {
            "chain": chain1,
            "name": name1,
            "use_labels": use_labels
        }

        if mode == "pp":
            side2 = {
                "chain": chain2,
                "name": name2,
                "use_labels": use_labels
            }
        elif mode == "ligand":
            side2 = {
                "resname": lig_resname,
                "chain": lig_chain,
                "name": name2
            }
        else:
            side2 = None

        with st.spinner("Analyzing structures. Please wait..."):
            st.session_state["map_result"] = run_map(
                [(f.name, f.getvalue()) for f in files],
                mode,
                side1,
                side2,
                {
                    "contact": contact,
                    "hbond": hbond,
                    "salt_bridge": salt,
                    "pi_cation": picat,
                    "hydrophobic": hydro,
                    "pi_stacking": pistack,
                    "pi_stacking_angle": (
                        pistack_angle if pistack_angle > 0 else None
                    ),
                    "pi_stacking_t_shaped": pistack_tshape
                },
                min_complexes,
                min_sep=min_sep,
                require_contact=require_contact,
                label_files=label_files
            )

result = st.session_state.get("map_result")

if result is not None:

    if not result["ok"]:
        st.error("Analysis could not be completed.")
        st.code(result["error"])

    else:
        st.success("Analysis completed.")

        if result["png"]:
            st.subheader("Interaction map")
            st.image(result["png"], use_container_width=True)
        else:
            st.warning(
                "No map was generated. Lower the 'Minimum complexes' "
                "setting, or check the chain IDs."
            )

        for filename in [
            "interaction_pairs.csv",
            "interaction_summary.csv"
        ]:
            if filename in result["tables"]:
                st.subheader(filename)
                st.dataframe(
                    pd.read_csv(BytesIO(result["tables"][filename]))
                )

        st.download_button(
            "Download all results (ZIP)",
            data=result["zip"],
            file_name="interaction_map_results.zip",
            mime="application/zip"
        )



# --------------------------------------------------
# 5. CONSERVATION ANALYSIS
# --------------------------------------------------

st.divider()
st.header("Conservation-Aware Protein Interaction Analysis")

conservation_fasta_files = st.file_uploader(
    "Upload multiple FASTA files containing related protein sequences",
    type=["fasta", "fa", "faa"],
    accept_multiple_files=True
)

sequences = {}

if conservation_fasta_files:

    for uploaded_file in conservation_fasta_files:

        fasta_text = uploaded_file.getvalue().decode("utf-8")
        current_name = None

        for line in fasta_text.splitlines():

            line = line.strip()

            if line.startswith(">"):
                current_name = f"{uploaded_file.name} | {line[1:]}"
                sequences[current_name] = ""

            elif current_name is not None:
                sequences[current_name] += line.upper()

    if len(sequences) < 2:
        st.warning("Please upload at least two protein sequences.")

    else:
        st.success(
            f"Successfully loaded {len(sequences)} protein sequences."
        )

        for name, sequence in sequences.items():
            st.write(f"**{name}:** {len(sequence)} amino acids")

        st.subheader("Select related protein sequences")

        selected_sequences = st.multiselect(
            "Choose sequences for conservation analysis",
            options=list(sequences.keys())
        )

        if len(selected_sequences) < 2:
            st.info("Select at least two related protein sequences.")

        else:

            if st.button("Align selected sequences"):

                try:
                    chosen = {
                        name: sequences[name]
                        for name in selected_sequences
                    }

                    aligned, method = align_sequences(chosen)

                    st.session_state["msa"] = aligned
                    st.session_state["msa_sequences"] = chosen
                    st.session_state["msa_method"] = method

                except Exception as error:
                    st.error(f"Alignment failed: {error}")


# --------------------------------------------------
# 6. DISPLAY ALIGNMENT AND CONSERVATION SCORES
# --------------------------------------------------

if "msa" in st.session_state:

    aligned = st.session_state["msa"]

    columns = len(next(iter(aligned.values())))

    st.success(
        f"Aligned {len(aligned)} sequences, "
        f"{columns} columns "
        f"({st.session_state['msa_method']})."
    )

    lines = []

    for i, (name, seq) in enumerate(aligned.items(), 1):
        lines.append(f"seq{i:<3} {seq}")

    st.code("\n".join(lines), language=None)

    for i, name in enumerate(aligned, 1):
        st.caption(f"seq{i} = {name}")

    st.download_button(
        "Download alignment (FASTA)",
        data=to_fasta(aligned),
        file_name="alignment.fasta",
        mime="text/plain"
    )

    scores = conservation_scores(aligned)
    score_table = pd.DataFrame(scores)

    st.subheader("Conservation per alignment column")

    st.bar_chart(
        score_table.set_index("column")["conservation"]
    )

    st.dataframe(score_table)

    st.download_button(
        "Download conservation scores (CSV)",
        data=score_table.to_csv(index=False),
        file_name="conservation_scores.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# 7. 3D STRUCTURE AND CONSERVATION VIEWER
# --------------------------------------------------

st.divider()
st.subheader("6. 3D Structure and Conservation Viewer")

if files:

    selected_file = st.selectbox(
        "Select a PDB structure",
        files,
        format_func=lambda x: x.name
    )

    pdb_text = selected_file.getvalue().decode(
        "utf-8",
        errors="replace"
    )

    pdb_conservation = {}
    sequence_scores = None
    conservation_sequence_name = None

    # Select FASTA sequence for PDB mapping
    if (
        "msa" in st.session_state
        and "msa_sequences" in st.session_state
    ):

        msa_names = list(st.session_state["msa"].keys())

        conservation_sequence_name = st.selectbox(
            "Select the FASTA sequence matching the first PDB chain",
            msa_names
        )

        aligned = st.session_state["msa"]
        scores = conservation_scores(aligned)

        sequence_scores = residue_conservation(
            aligned,
            scores,
            conservation_sequence_name
        )

    cutoff = st.slider(
        "Interaction contact cutoff (Å)",
        2.0,
        8.0,
        4.0,
        0.1
    )

    atoms = {chain1: []}

    if mode == "pp":
        atoms[chain2] = []

    pdb_residues = []
    seen_residues = set()

    # ----------------------------------------------
    # Parse PDB ATOM records
    # ----------------------------------------------

    for line in pdb_text.splitlines():

        if not line.startswith("ATOM  "):
            continue

        if line[16] not in (" ", "A"):
            continue

        chain = line[21].strip()

        if chain not in atoms:
            continue

        atom_name = line[12:16].strip()
        element = line[76:78].strip().upper()

        if not element:
            element = "".join(
                c for c in atom_name if c.isalpha()
            )[:1].upper()

        if element in ("H", "D"):
            continue

        try:

            atom = {
                "xyz": [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54])
                ],
                "resi": int(line[22:26]),
                "resname": line[17:20].strip()
            }

            atoms[chain].append(atom)

            if chain == chain1:

                residue_key = (
                    atom["resi"],
                    atom["resname"]
                )

                if residue_key not in seen_residues:
                    pdb_residues.append(residue_key)
                    seen_residues.add(residue_key)

        except (ValueError, IndexError):
            continue

    # ----------------------------------------------
    # Map conservation scores to PDB residues
    # ----------------------------------------------

    if (
        sequence_scores is not None
        and pdb_residues
        and conservation_sequence_name is not None
    ):

        try:
            pdb_conservation = map_conservation_to_pdb(
                st.session_state["msa_sequences"][
                    conservation_sequence_name
                ],
                sequence_scores,
                pdb_residues
            )

            st.success(
                f"Mapped conservation scores to "
                f"{len(pdb_conservation)} PDB residues."
            )

        except Exception as error:
            st.warning(f"Conservation mapping failed: {error}")

    # ----------------------------------------------
    # Create 3D viewer
    # ----------------------------------------------

    viewer = py3Dmol.view(width=900, height=600)

    viewer.addModel(pdb_text, "pdb")

    viewer.setStyle(
        {"chain": chain1},
        {"cartoon": {"color": "blue"}}
    )

    if mode == "pp":
        viewer.setStyle(
            {"chain": chain2},
            {"cartoon": {"color": "orange"}}
        )

    # ----------------------------------------------
    # Color residues by conservation
    # ----------------------------------------------

    for residue_number, score in pdb_conservation.items():

        if score >= 0.8:
            color = "green"

        elif score >= 0.5:
            color = "yellow"

        else:
            color = "red"

        viewer.addStyle(
            {
                "chain": chain1,
                "resi": residue_number
            },
            {
                "cartoon": {
                    "color": color
                }
            }
        )

    # ----------------------------------------------
    # Calculate protein-protein contacts
    # ----------------------------------------------

    if mode == "pp" and atoms[chain1] and atoms[chain2]:

        coords1 = np.array(
            [a["xyz"] for a in atoms[chain1]]
        )

        coords2 = np.array(
            [a["xyz"] for a in atoms[chain2]]
        )

        distances = cdist(coords1, coords2)
        indices = np.where(distances <= cutoff)

        residue_pairs = {}

        for i, j in zip(*indices):

            a = atoms[chain1][i]
            b = atoms[chain2][j]

            key = (a["resi"], b["resi"])
            distance = distances[i, j]

            if (
                key not in residue_pairs
                or distance < residue_pairs[key]["distance"]
            ):

                residue_pairs[key] = {
                    "distance": distance,
                    "a": a,
                    "b": b
                }

        # ------------------------------------------
        # Identify highly conserved interacting residues
        # ------------------------------------------

        conserved_interacting = []

        for (resi1, resi2), pair in residue_pairs.items():

            score = pdb_conservation.get(resi1)

            if score is not None and score >= 0.8:

                conserved_interacting.append({
                    "Residue": f"{pair['a']['resname']}{resi1}",
                    "Partner residue": f"{pair['b']['resname']}{resi2}",
                    "Conservation": round(score, 3)
                })

        # ------------------------------------------
        # Highlight interacting residues
        # ------------------------------------------

        labelled_residues = set()

        for (resi1, resi2), pair in residue_pairs.items():

            viewer.addStyle(
                {"chain": chain1, "resi": resi1},
                {"stick": {"color": "yellow"}}
            )

            viewer.addStyle(
                {"chain": chain2, "resi": resi2},
                {"stick": {"color": "yellow"}}
            )

            if (chain1, resi1) not in labelled_residues:

                viewer.addLabel(
                    f"{pair['a']['resname']}{resi1}",
                    {
                        "position": dict(zip(
                            ["x", "y", "z"],
                            pair["a"]["xyz"]
                        )),
                        "backgroundColor": "blue",
                        "fontColor": "white",
                        "fontSize": 11
                    }
                )

                labelled_residues.add((chain1, resi1))

            if (chain2, resi2) not in labelled_residues:

                viewer.addLabel(
                    f"{pair['b']['resname']}{resi2}",
                    {
                        "position": dict(zip(
                            ["x", "y", "z"],
                            pair["b"]["xyz"]
                        )),
                        "backgroundColor": "orange",
                        "fontColor": "white",
                        "fontSize": 11
                    }
                )

                labelled_residues.add((chain2, resi2))

            viewer.addLine({
                "start": dict(zip(
                    ["x", "y", "z"],
                    pair["a"]["xyz"]
                )),
                "end": dict(zip(
                    ["x", "y", "z"],
                    pair["b"]["xyz"]
                )),
                "color": "yellow",
                "dashed": True,
                "linewidth": 2
            })

        st.success(
            f"Found {len(residue_pairs)} interacting residue pairs."
        )

        st.subheader("Highly conserved interacting residues")

        if conserved_interacting:

            st.dataframe(
                pd.DataFrame(conserved_interacting),
                use_container_width=True
            )

        else:
            st.info(
                "No highly conserved interacting residues were identified."
            )

    elif mode == "pp":
        st.warning(
            "Could not find atoms in both selected chains. "
            "Check the chain IDs."
        )

    else:
        st.info(
            "Contacts in the 3D viewer are shown for protein vs protein "
            "only. The structure is shown without contact lines."
        )

    # ----------------------------------------------
    # Display 3D structure
    # ----------------------------------------------

    viewer.zoomTo()

    components.html(
        viewer._make_html(),
        height=620,
        scrolling=False
    )

    st.caption(
        "Conservation colors: green = high, yellow = moderate, "
        "red = low. Yellow sticks and dashed lines indicate contacts."
    )

else:
    st.info("Upload a PDB file to view its 3D structure.")
