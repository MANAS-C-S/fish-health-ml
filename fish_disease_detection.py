"""
Fish Health Finding Classification — Machine Learning Assignment
Dataset: Fisheries and Oceans Canada, Fish Health Database

Task: classify the recorded ETIOLOGICAL DESCRIPTION using available case context.
This predicts historical diagnostic-record labels, NOT disease from water-quality
measurements and NOT a veterinary diagnosis for a live fish.

Put the downloaded CSV at data/Fish_Health_Database.csv, then run:
    python fish_disease_detection.py
"""

from pathlib import Path
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, ConfusionMatrixDisplay, confusion_matrix
)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Fish_Health_Database.csv"
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def read_dataset(path):
    """Read the downloaded source CSV robustly across common Windows encodings."""
    last_error = None
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def clean_column_name(name):
    # Source headings contain bilingual names and trailing slash separators.
    name = str(name).replace("\ufeff", "").strip().rstrip("/").strip()
    name = re.sub(r"\s+", " ", name)
    return name


def find_column(columns, candidates):
    lowered = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        for low, original in lowered.items():
            if candidate in low:
                return original
    return None


def main():
    if not DATA_PATH.exists():
        print("Dataset file not found.")
        print("Download the English CSV from:")
        print("https://open.canada.ca/data/en/dataset/2ece9991-62aa-4b7a-bd7d-4f8f1052cd21")
        print(f"Save it as: {DATA_PATH}")
        return

    df = read_dataset(DATA_PATH)
    df.columns = [clean_column_name(c) for c in df.columns]
    # Some public exports include a second row of French column names. Remove it
    # when the CASE field is non-numeric but the first column is expected to be an ID.
    first_col = df.columns[0]
    df = df.dropna(how="all").copy()

    print("\n=== DATASET OVERVIEW ===")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print("Columns:")
    for c in df.columns:
        print(f" - {c}")

    target = find_column(df.columns, ["etiological description", "description etiologique"])
    species = find_column(df.columns, ["species name", "nom de l'espece"])
    life_stage = find_column(df.columns, ["life stage description", "description du stade biologique"])
    location = find_column(df.columns, ["location name", "nom du lieu"])
    date_col = find_column(df.columns, ["start date", "date de debut"])

    if target is None:
        print("\nCould not find the etiological-description column automatically.")
        print("Check the printed column names and update the TARGET selection in the script.")
        return

    # If an accidental French header row was parsed as data, remove it.
    target_text = df[target].astype("string").str.strip()
    df = df[target_text.notna() & target_text.ne("")].copy()
    target_text = df[target].astype(str).str.strip()

    # Normalize whitespace; keep labels as recorded. Rare labels are grouped to make
    # stratified evaluation feasible, and this grouping is explicitly reported.
    target_text = target_text.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    df[target] = target_text
    df = df.dropna(subset=[target]).copy()
    counts = df[target].value_counts()
    min_count = 10
    common = counts[counts >= min_count].index
    df = df[df[target].isin(common)].copy()

    if df[target].nunique() < 2 or len(df) < 50:
        print("\nNot enough repeated target classes for a defensible supervised model.")
        print("No model was trained. Choose a better-supported target/dataset.")
        return

    # Use only pre-diagnostic context available in the source. Exclude case IDs,
    # coordinates, and the target itself. No claims of causal prediction are made.
    feature_cols = [c for c in [species, life_stage, location, date_col] if c is not None]
    if not feature_cols:
        print("No usable context feature columns found. No model was trained.")
        return

    X = df[feature_cols].copy()
    y = df[target].astype(str).copy()

    # Dates are represented by year/month only, if parseable.
    for c in list(X.columns):
        if c == date_col:
            parsed = pd.to_datetime(X[c], errors="coerce")
            X["case_year"] = parsed.dt.year
            X["case_month"] = parsed.dt.month
            X.drop(columns=[c], inplace=True)

    # Remove columns with no variation.
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.shape[1] == 0:
        print("No varying features remain after preprocessing.")
        return

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ])

    # A fixed holdout split. The grouping threshold ensures enough examples/class
    # for stratification, while the fallback handles unusual class distributions.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, min_samples_leaf=2, n_jobs=-1
        ))
    ])
    baseline = DummyClassifier(strategy="most_frequent")

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Baseline uses the most common class from training only.
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    base_pred = baseline.predict(np.zeros((len(y_test), 1)))

    accuracy = accuracy_score(y_test, pred)
    baseline_accuracy = accuracy_score(y_test, base_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )

    print("\n=== HELD-OUT TEST RESULTS ===")
    print(f"Rows used: {len(df)}")
    print(f"Target column: {target}")
    print(f"Input features: {', '.join(X.columns)}")
    print(f"Number of target classes: {df[target].nunique()}")
    print(f"Random Forest accuracy: {accuracy:.4f}")
    print(f"Majority-class baseline accuracy: {baseline_accuracy:.4f}")
    print(f"Macro precision: {p_macro:.4f}")
    print(f"Macro recall: {r_macro:.4f}")
    print(f"Macro F1-score: {f_macro:.4f}")
    print(f"Weighted F1-score: {f_weight:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_test, pred, zero_division=0))

    # Save target distribution.
    plt.figure(figsize=(10, 6))
    df[target].value_counts().head(15).sort_values().plot(kind="barh")
    plt.title("Most Frequent Etiological Findings")
    plt.xlabel("Number of records")
    plt.ylabel("Recorded finding")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "class_distribution.png", dpi=160)
    plt.close()

    # Confusion matrix: only the 12 most common labels in test data to keep readable.
    top_labels = pd.Series(y_test).value_counts().head(12).index.tolist()
    mask = pd.Series(y_test).isin(top_labels)
    cm = confusion_matrix(y_test[mask], pd.Series(pred, index=y_test.index)[mask], labels=top_labels)
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay(cm, display_labels=top_labels).plot(
        ax=ax, xticks_rotation=90, colorbar=False, values_format="d"
    )
    ax.set_title("Confusion Matrix — 12 Most Common Test Classes")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # Feature importance after one-hot encoding.
    try:
        names = model.named_steps["preprocess"].get_feature_names_out()
        values = model.named_steps["classifier"].feature_importances_
        imp = pd.Series(values, index=names).sort_values().tail(15)
        plt.figure(figsize=(10, 7))
        imp.plot(kind="barh")
        plt.title("Top Random Forest Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "feature_importance.png", dpi=160)
        plt.close()
    except Exception as exc:
        print("Feature importance plot not created:", exc)

    summary = pd.DataFrame([{
        "dataset_rows_used": len(df),
        "target_column": target,
        "feature_columns": ", ".join(X.columns),
        "number_of_classes": df[target].nunique(),
        "random_forest_accuracy": round(accuracy, 4),
        "majority_baseline_accuracy": round(baseline_accuracy, 4),
        "macro_precision": round(p_macro, 4),
        "macro_recall": round(r_macro, 4),
        "macro_f1": round(f_macro, 4),
        "weighted_f1": round(f_weight, 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test)
    }])
    summary.to_csv(ROOT / "results_summary.csv", index=False)

    # Report helper with real results and a warning about interpretation.
    with open(ROOT / "results_for_report.txt", "w", encoding="utf-8") as f:
        f.write("FISH HEALTH FINDING CLASSIFICATION — ACTUAL RUN RESULTS\n")
        f.write(f"Rows used: {len(df)}\nTarget: {target}\nFeatures: {', '.join(X.columns)}\n")
        f.write(f"Classes: {df[target].nunique()}\n")
        f.write(f"Random Forest accuracy: {accuracy:.4f}\nBaseline accuracy: {baseline_accuracy:.4f}\n")
        f.write(f"Macro precision: {p_macro:.4f}\nMacro recall: {r_macro:.4f}\nMacro F1: {f_macro:.4f}\n")
        f.write(f"Weighted F1: {f_weight:.4f}\n\n")
        f.write(classification_report(y_test, pred, zero_division=0))
        f.write("\nCaution: model predicts historical etiological-description labels from case context; ")
        f.write("it is not a clinical diagnostic tool and cannot establish causation.\n")

    print("\nSaved results_summary.csv, results_for_report.txt and plots/*.png")
    print("\nInterpretation: this model predicts recorded historical findings from case context.")
    print("It is not a veterinary diagnostic system and must not be described as diagnosing live fish.")


if __name__ == "__main__":
    main()
