"""Streamlit demo for the Fish Health Finding Classification assignment.
This predicts recorded historical etiological labels; it is not a live-fish diagnosis.
"""
from pathlib import Path
import re
import warnings
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Fish_Health_Database.csv"

st.set_page_config(page_title="Fish Health Finding Classifier", page_icon="🐟", layout="centered")
st.title("🐟 Fish Health Finding Classifier")
st.caption("Machine Learning Application · Random Forest")
st.warning(
    "Educational demonstration only. This predicts historical recorded findings from case context. "
    "It does not diagnose a live fish and should not guide veterinary or farm-health decisions."
)


def read_dataset(path):
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def clean_column_name(name):
    name = str(name).replace("\ufeff", "").strip().rstrip("/").strip()
    return re.sub(r"\s+", " ", name)


def find_column(columns, candidates):
    for candidate in candidates:
        for col in columns:
            if candidate in str(col).lower():
                return col
    return None


@st.cache_resource(show_spinner="Training the Random Forest model on the dataset…")
def train_model(csv_path, modified_time):
    df = read_dataset(csv_path)
    df.columns = [clean_column_name(c) for c in df.columns]
    target = find_column(df.columns, ["etiological description", "description etiologique"])
    species = find_column(df.columns, ["species name", "nom de l'espece"])
    life_stage = find_column(df.columns, ["life stage description", "description du stade biologique"])
    location = find_column(df.columns, ["location name", "nom du lieu"])
    date_col = find_column(df.columns, ["start date", "date de debut"])
    if target is None:
        raise ValueError("Could not find the ETIOLOGICAL DESCRIPTION column in the CSV.")

    df = df.dropna(how="all").copy()
    labels = df[target].astype("string").str.strip()
    df = df[labels.notna() & labels.ne("")].copy()
    df[target] = df[target].astype(str).str.strip()
    df = df[~df[target].isin(["nan", "None", "", "DESCRIPTION ETIOLOGIQUE"])].copy()
    counts = df[target].value_counts()
    df = df[df[target].isin(counts[counts >= 10].index)].copy()

    feature_cols = [c for c in [species, life_stage, location, date_col] if c is not None]
    X = df[feature_cols].copy()
    y = df[target].astype(str)
    if date_col is not None and date_col in X.columns:
        parsed = pd.to_datetime(X[date_col], errors="coerce")
        X["case_year"] = parsed.dt.year
        X["case_month"] = parsed.dt.month
        X.drop(columns=[date_col], inplace=True)
    X = X.loc[:, X.nunique(dropna=True) > 1]
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, class_weight="balanced_subsample", random_state=42,
            min_samples_leaf=2, n_jobs=-1
        ))
    ])
    model.fit(X, y)
    return model, df, feature_cols, species, life_stage, location, date_col, list(X.columns)


if not DATA_PATH.exists():
    st.error("Dataset not found. Create a `data` folder beside app.py and put the CSV inside it.")
    st.code("data/Fish_Health_Database.csv")
    st.markdown("[Download the official dataset](https://open.canada.ca/data/en/dataset/2ece9991-62aa-4b7a-bd7d-4f8f1052cd21)")
    st.stop()

try:
    model, df, feature_cols, species_col, stage_col, location_col, date_col, model_features = train_model(
        str(DATA_PATH), DATA_PATH.stat().st_mtime
    )
except Exception as exc:
    st.error(f"Could not prepare the model: {exc}")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Historical records", f"{len(df):,}")
m2.metric("Recorded categories", f"{df[find_column(df.columns, ['etiological description', 'description etiologique'])].nunique():,}")
m3.metric("Model", "Random Forest")
st.divider()
st.subheader("Enter case context")
st.write("Choose the closest matching values from the historical dataset.")


def options_for(col, limit=100):
    if col is None:
        return ["Unknown"]
    vals = df[col].dropna().astype(str).str.strip()
    vals = vals[vals.ne("")].value_counts().head(limit).index.tolist()
    return vals or ["Unknown"]

with st.form("prediction_form"):
    chosen_species = st.selectbox("Species name", options_for(species_col, 100)) if species_col else "Unknown"
    chosen_stage = st.selectbox("Life stage", options_for(stage_col, 50)) if stage_col else "Unknown"
    chosen_location = st.selectbox("Location name", options_for(location_col, 100)) if location_col else "Unknown"
    chosen_date = st.date_input("Case date", value=pd.Timestamp("2020-01-01").date()) if date_col else None
    submitted = st.form_submit_button("Predict recorded finding", type="primary")

if submitted:
    row = {}
    if species_col in feature_cols if species_col else False:
        row[species_col] = chosen_species
    if stage_col in feature_cols if stage_col else False:
        row[stage_col] = chosen_stage
    if location_col in feature_cols if location_col else False:
        row[location_col] = chosen_location
    if date_col is not None and date_col in feature_cols:
        # Match the training transformation: date becomes year and month.
        row["case_year"] = chosen_date.year
        row["case_month"] = chosen_date.month
    input_df = pd.DataFrame([{col: row.get(col, np.nan) for col in model_features}])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted recorded category: **{prediction}**")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = model.named_steps["classifier"].classes_
        top_indices = np.argsort(probabilities)[::-1][:5]
        result = pd.DataFrame({"Recorded category": classes[top_indices], "Model score": probabilities[top_indices]})
        result["Model score"] = (result["Model score"] * 100).round(1).astype(str) + "%"
        st.subheader("Top model outputs")
        st.dataframe(result, hide_index=True, width="stretch")
        st.caption("Scores are model outputs, not clinically validated probabilities of disease.")

with st.expander("About this model"):
    st.write("The model learns associations between species, life stage, location/date and recorded etiological descriptions in historical cases.")
    st.write("It may reflect sampling bias, class imbalance and location-specific patterns. Results should not be interpreted as causal or as a veterinary diagnosis.")
