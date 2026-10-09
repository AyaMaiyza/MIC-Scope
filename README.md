# MIC-Scope

AI-based antibacterial activity prediction using a scaffold-aware ExtraTrees model.

## What the app does

- Accepts SMILES for a single compound or a batch CSV.
- Predicts molar pMIC and derives predicted MIC in µg/mL.
- Calculates maximum Count Morgan Tanimoto similarity to training compounds.
- Reports an exploratory applicability-domain (AD) status.
- Downloads prediction results as CSV.

## Important scientific limitations

- For research use only; predictions are not laboratory measurements.
- The Tanimoto threshold of 0.40 is exploratory, not a validated confidence boundary.
- Exact structures found in training are not independent external validation.
- Experimental MIC values must be compared only when assay conditions, organism and strain are appropriate.
- The model artifact must be loaded with compatible scikit-learn/joblib versions.

## Files required by the app

The app expects these three artifacts:

1. `ATCC25923_EXACT_ONLY_SCAFFOLD_EXTRATREES.joblib`
2. `feature_columns.csv`
3. `ATCC25923_SCAFFOLD_TRAIN.csv`

The 337 MB model should not be committed as a normal GitHub file. Store the three artifacts in a Hugging Face model repository or another reliable artifact host. This app supports Hugging Face Hub.

## Deploy with Streamlit Community Cloud

1. Push this repository to GitHub.
2. Upload the three artifacts to a Hugging Face **model** repository.
3. In Streamlit Community Cloud, deploy `app.py` from this GitHub repository.
4. In the app's Streamlit secrets, add:

```toml
HF_REPO_ID = "YOUR_HUGGINGFACE_USERNAME/YOUR_MODEL_REPO"
# If the Hugging Face repository is private, add:
# HF_TOKEN = "hf_your_read_token"
```

Artifact filenames in the Hugging Face repository must match the three names above.

## Version compatibility

Before public deployment, identify the exact `scikit-learn`, `numpy`, and `joblib` versions used to train/save the model, then pin compatible versions in `requirements.txt`. Do not assume a model serialized in one scikit-learn version is safely portable to any other version.

## Local development

Place the three artifacts in `assets/`, then run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

You can also set `MODEL_DIR` to a directory containing the three artifacts.
