from pathlib import Path
import math
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
import py3Dmol
from stmol import showmol 

st.set_page_config(page_title="MIC-Scope", page_icon="🧪", layout="wide")

APP_DIR = Path(__file__).resolve().parent
AD_THRESHOLD = 0.40  # Exploratory threshold; not a validated confidence boundary.

MODEL_FILENAME = "ATCC25923_EXACT_ONLY_SCAFFOLD_EXTRATREES.joblib"
FEATURES_FILENAME = "feature_columns.csv"
TRAIN_FILENAME = "ATCC25923_SCAFFOLD_TRAIN.csv"


def get_setting(name, default=None):
    """Read configuration from Streamlit secrets first, then environment."""
    try:
        value = st.secrets.get(name, None)
        if value not in (None, ""):
            return value
    except Exception:
        pass
    return os.environ.get(name, default)


def resolve_artifact(filename):
    """
    Deployment:
      - Put the artifact in the Hugging Face model repo and set HF_REPO_ID.
    Local development:
      - Put the artifact in ./assets/ or set MODEL_DIR.
    """
    model_dir = Path(get_setting("MODEL_DIR", str(APP_DIR / "assets")))
    local_path = model_dir / filename
    if local_path.is_file():
        return str(local_path)

    repo_id = get_setting("HF_REPO_ID")
    if repo_id:
        token = get_setting("HF_TOKEN")
        return hf_hub_download(
            repo_id=str(repo_id),
            filename=filename,
            repo_type="model",
            token=str(token) if token else None,
        )

    raise FileNotFoundError(
        f"Could not find {filename}. For deployment, upload it to a Hugging Face "
        "model repository and configure HF_REPO_ID in Streamlit secrets. "
        f"For local use, place it in {model_dir}."
    )


@st.cache_resource(show_spinner="Loading the model and reference data...")
def load_resources():
    model_path = resolve_artifact(MODEL_FILENAME)
    features_path = resolve_artifact(FEATURES_FILENAME)
    train_path = resolve_artifact(TRAIN_FILENAME)

    model = joblib.load(model_path)
    feature_df = pd.read_csv(features_path)
    if "feature" not in feature_df.columns:
        raise ValueError("feature_columns.csv must contain a 'feature' column.")
    feature_names = feature_df["feature"].astype(str).tolist()

    if len(feature_names) != 2068:
        raise ValueError(f"Expected 2068 features, found {len(feature_names)}.")
    if getattr(model, "n_features_in_", len(feature_names)) != len(feature_names):
        raise ValueError(
            f"Model expects {getattr(model, 'n_features_in_', 'unknown')} features; "
            f"feature file has {len(feature_names)}."
        )

    train = pd.read_csv(train_path)
    smiles_col = next(
        (c for c in ["SMILES", "smiles", "canonical_smiles", "Canonical_SMILES"]
         if c in train.columns),
        None,
    )
    if smiles_col is None:
        raise ValueError(
            "Training CSV has no recognizable SMILES column. "
            f"Available columns: {list(train.columns)}"
        )

    train_smiles, train_fps = [], []
    for value in train[smiles_col].dropna().astype(str):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            continue
        train_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        train_fps.append(AllChem.GetMorganFingerprint(mol, 2, useChirality=True))

    if not train_fps:
        raise ValueError("No valid training SMILES found in the training CSV.")

    return model, feature_names, set(train_smiles), train_fps


def make_features(mol, feature_names):
    values = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    bits = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, bits)
    for i, bit in enumerate(bits):
        values[f"Morgan_R2_{i}"] = int(bit)

    values.update({
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "MolMR": Crippen.MolMR(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "NumRings": Lipinski.RingCount(mol),
        "NumAromaticRings": Lipinski.NumAromaticRings(mol),
        "NumAliphaticRings": Lipinski.NumAliphaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "NumHeavyAtoms": Lipinski.HeavyAtomCount(mol),
        "NumHeteroatoms": Lipinski.NumHeteroatoms(mol),
        "NumSaturatedRings": Lipinski.NumSaturatedRings(mol),
        "NumAromaticHeterocycles": rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        "NumAliphaticHeterocycles": rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
        "NumSaturatedHeterocycles": rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
        "NumSpiroAtoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "NumBridgeheadAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
    })

    missing = [name for name in feature_names if name not in values]
    if missing:
        raise ValueError(f"Missing model features: {missing[:10]}")
    return pd.DataFrame([[values[n] for n in feature_names]], columns=feature_names)


def predict_one(smiles, resources):
    model, feature_names, train_smiles, train_fps = resources
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError("Invalid SMILES. Please check the structure.")

    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    X = make_features(mol, feature_names)
    pred_pmic = float(model.predict(X)[0])
    mw = float(Descriptors.MolWt(mol))
    pred_mic = 10 ** (3 + math.log10(mw) - pred_pmic)

    count_fp = AllChem.GetMorganFingerprint(mol, 2, useChirality=True)
    sims = DataStructs.BulkTanimotoSimilarity(count_fp, train_fps)
    max_sim = float(max(sims)) if sims else float("nan")

    return {
        "Canonical_SMILES": canonical,
        "Molecular_Weight_g_mol": mw,
        "Predicted_pMIC": pred_pmic,
        "Predicted_MIC_ug_mL": pred_mic,
        "Max_Count_Morgan_Tanimoto": max_sim,
        "Exact_Structure_In_Training": canonical in train_smiles,
        "Exploratory_AD_Threshold": AD_THRESHOLD,
        "Exploratory_AD_Status": "Inside" if max_sim >= AD_THRESHOLD else "Outside",
    }, mol


st.title("🧪 MIC-Scope")
st.subheader("AI-Based Antibacterial Activity Prediction")
st.markdown(
    "Predict antibacterial activity against **_Staphylococcus aureus_ ATCC 25923** "
    "using a scaffold-aware ExtraTrees model."
)
st.warning(
    "Research-use prototype only. Predictions are computational estimates, "
    "not a substitute for laboratory measurements. The AD threshold is exploratory."
)

with st.expander("About this model"):
    st.write(
        "Model target: molar pMIC. Predicted MIC is reported in µg/mL. "
        "A similarity threshold of 0.40 is shown as an exploratory applicability-domain "
        "indicator and is not a calibrated probability of correctness."
    )

tab_single, tab_batch = st.tabs(["Single compound", "Batch CSV"])

with tab_single:
    smiles = st.text_area(
        "Enter SMILES",
        value="C1=CC=C(C=C1)CN=C=S",
        help="Example: benzyl isothiocyanate (BITC).",
    )
    if st.button("Predict compound", type="primary", key="single_predict"):
        try:
            with st.spinner("Loading model and calculating prediction..."):
                resources = load_resources()
                result, mol = predict_one(smiles, resources)
            st.success("Prediction completed.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted pMIC", f"{result['Predicted_pMIC']:.4f}")
            c2.metric("Predicted MIC (µg/mL)", f"{result['Predicted_MIC_ug_mL']:.4g}")
            c3.metric("Molecular weight (g/mol)", f"{result['Molecular_Weight_g_mol']:.3f}")


            try:
             viewer = py3Dmol.view(width=450, height=300)
             viewer.addModel(Chem.MolToMolBlock(mol), "mol")
             viewer.setStyle({"stick": {}})
             viewer.zoomTo()
             showmol(viewer, height=300, width=450)
         except Exception as exc:
             st.info(f"Structure preview is unavailable: {exc}")
    
            c4, c5 = st.columns(2)
            c4.metric("Maximum Count Morgan Tanimoto", f"{result['Max_Count_Morgan_Tanimoto']:.4f}")
            if result["Exact_Structure_In_Training"]:
                c5.warning("Exact structure found in training data.")
                st.warning("This is not an independent external prediction.")
            elif result["Max_Count_Morgan_Tanimoto"] >= AD_THRESHOLD:
                c5.success("Inside exploratory AD threshold (≥ 0.40).")
            else:
                c5.warning("Outside exploratory AD threshold (< 0.40).")

            st.dataframe(pd.DataFrame([result]), width="stretch")
            st.download_button(
                "Download result CSV",
                pd.DataFrame([result]).to_csv(index=False).encode("utf-8"),
                file_name="mic_scope_prediction.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")
            st.info(
                "If this is a deployment, check that the three model artifacts are "
                "available and that scikit-learn matches the model-training environment."
            )

with tab_batch:
    st.write("Upload a CSV containing a column named `SMILES` or `smiles`.")
    uploaded = st.file_uploader("Upload compound CSV", type=["csv"])
    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded)
            smiles_col = next((c for c in ["SMILES", "smiles"] if c in input_df.columns), None)
            if smiles_col is None:
                st.error("The CSV must contain a SMILES or smiles column.")
            elif st.button("Predict batch", type="primary", key="batch_predict"):
                resources = load_resources()
                rows = []
                progress = st.progress(0)
                for i, value in enumerate(input_df[smiles_col].fillna("").astype(str)):
                    row = {"Input_SMILES": value}
                    try:
                        result, _ = predict_one(value, resources)
                        row.update(result)
                        row["Status"] = "Predicted"
                    except Exception as exc:
                        row["Status"] = f"Failed: {exc}"
                    rows.append(row)
                    progress.progress((i + 1) / max(len(input_df), 1))
                results_df = pd.DataFrame(rows)
                st.dataframe(results_df, width="stretch")
                st.download_button(
                    "Download batch predictions CSV",
                    results_df.to_csv(index=False).encode("utf-8"),
                    file_name="mic_scope_batch_predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")

st.caption("MIC-Scope | Research prototype | Scaffold-aware ExtraTrees")
