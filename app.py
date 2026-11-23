import streamlit as st
import tempfile
import subprocess
import json
import os
import sys
import zipfile
from io import BytesIO

import pandas as pd
import numpy as np
import py3Dmol
import streamlit.components.v1 as components
from scipy.spatial.distance import cdist

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
<div style="
background-color:#F3EEFA;
border-radius:12px;
padding:12px 0;
    display:flex;
    align-items:center;
    justify-content:center;
    gap:8px;
    padding:4px 0 10px;
">
    <span style="
        color:#008C95;
        font-size:52px;
        font-weight:700;
    ">
        Evo<span style="font-size:52px;">Pro</span>Trace
    </span>
    <span style="font-size:44px;">🧬</span>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 1. UPLOAD PDB STRUCTURES
# --------------------------------------------------

# --------------------------------------------------
# 1. UPLOAD PDB STRUCTURES
# --------------------------------------------------

st.subheader("1. Upload PDB structures")

files = st.file_uploader(
    "Upload one or more PDB files",
    type=["pdb"],
    accept_multiple_files=True
)

# --------------------------------------------------
# 2. CONFIGURE PROTEIN CHAINS
# --------------------------------------------------

st.subheader("2. Configure protein chains")

col1, col2 = st.columns(2)

with col1:
    chain1 = st.text_input("First chain ID", "A").strip()
    name1 = st.text_input("First protein name", "Nanobody")

with col2:
    chain2 = st.text_input("Second chain ID", "B").strip()
    name2 = st.text_input("Second protein name", "Target")

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
    contact = st.number_input("Contact (Å)", value=4.0)
    min_complexes = st.number_input(
        "Minimum complexes",
        min_value=1,
        value=2
    )

# --------------------------------------------------
# 4. GENERATE INTERACTION MAP
# --------------------------------------------------

if st.button("Generate interaction map", type="primary"):

    if not files:
        st.error("Please upload at least one PDB file.")

    elif not chain1 or not chain2:
        st.error("Please enter both chain IDs.")

    elif chain1 == chain2:
        st.error("Please use two different chain IDs.")

    else:
        script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "interaction_map.py"
        )

        if not os.path.exists(script):
            st.error("interaction_map.py was not found.")

        else:
            with st.spinner("Analyzing structures. Please wait..."):

                with tempfile.TemporaryDirectory() as temp:

                    structures = os.path.join(temp, "structures")
                    output = os.path.join(temp, "results")

                    os.makedirs(structures)

                    for file in files:
                        with open(
                            os.path.join(structures, file.name), "wb"
                        ) as f:
                            f.write(file.getvalue())

                    config = {
                        "structures_dir": "structures",
                        "file_pattern": "*.pdb",
                        "ranking": None,
                        "side1": {
                            "chain": chain1,
                            "name": name1,
                            "labels": None
                        },
                        "side2": {
                            "chain": chain2,
                            "name": name2,
                            "labels": None
                        },
                        "thresholds": {
                            "contact": contact,
                            "hbond": hbond,
                            "salt_bridge": salt,
                            "pi_cation": picat,
                            "hydrophobic": hydro,
                            "pi_stacking": 5.5
                        },
                        "min_complexes": int(min_complexes),
                        "output_dir": "results"
                    }

                    config_path = os.path.join(temp, "config.json")

                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=4)

                    result = subprocess.run(
                        [
                            sys.executable,
                            script,
                            "--config",
                            config_path,
                            "--out",
                            output
                        ],
                        capture_output=True,
                        text=True
                    )

                    if result.returncode != 0:
                        st.error("Analysis could not be completed.")
                        st.code(result.stderr or result.stdout)

                    elif os.path.exists(output):

                        st.success("Analysis completed.")

                        png = os.path.join(
                            output, "interaction_map.png"
                        )

                        if os.path.exists(png):
                            st.subheader("Interaction map")
                            st.image(png, use_container_width=True)
                        else:
                            st.warning(
                                "No map was generated. Check the "
                                "minimum complexes setting."
                            )

                        for filename in [
                            "interaction_pairs.csv",
                            "interaction_summary.csv"
                        ]:
                            path = os.path.join(output, filename)

                            if os.path.exists(path):
                                st.subheader(filename)
                                st.dataframe(pd.read_csv(path))

                        memory_file = BytesIO()

                        with zipfile.ZipFile(
                            memory_file, "w", zipfile.ZIP_DEFLATED
                        ) as archive:

                            for filename in os.listdir(output):
                                path = os.path.join(output, filename)

                                if os.path.isfile(path):
                                    archive.write(path, filename)

                        st.download_button(
                            "Download all results (ZIP)",
                            data=memory_file.getvalue(),
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
st.subheader("4. 3D Structure and Conservation Viewer")

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

    atoms = {
        chain1: [],
        chain2: []
    }

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

    if atoms[chain1] and atoms[chain2]:

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

    else:
        st.warning(
            "Could not find atoms in both selected chains. "
            "Check the chain IDs."
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
