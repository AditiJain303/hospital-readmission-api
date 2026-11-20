"""
FastAPI backend for the 30-day readmission model.

Run from this folder:
    uvicorn app:app --reload
Then open http://127.0.0.1:8000/docs for the interactive API page.

The saved file random_forest_readmission.pkl contains:
    {"model", "threshold", "tier_cutoffs", "actions"}
The model expects ENGINEERED features, so every request goes through
prepare_features(), which repeats the feature engineering from Random_forest.py.
If you change the feature engineering there, change it here too.
"""

import io
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = "random_forest_readmission.pkl"
print("LOADED APP FROM:", __file__)

# --- Load the model once at start-up -----------------------------------------
saved = joblib.load(MODEL_PATH)
model = saved["model"]
THRESHOLD = float(saved["threshold"])
TIER_CUTOFFS = saved["tier_cutoffs"]
ACTIONS = saved["actions"]
MODEL_COLUMNS = list(model.feature_names_in_)   # exact columns used in training

MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


# --- Feature engineering (same as Random_forest.py) ---------------------------
def icd9_group(code):
    if pd.isna(code):
        return "Unknown"
    code = str(code)
    if code[0] in ("V", "E"):
        return "Other"
    value = float(code)
    if 390 <= value <= 459 or value == 785:
        return "Circulatory"
    if 460 <= value <= 519 or value == 786:
        return "Respiratory"
    if 520 <= value <= 579 or value == 787:
        return "Digestive"
    if int(value) == 250:
        return "Diabetes"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or value == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw patient rows (same columns as diabetic_data.csv) into model input."""
    df = raw.copy().replace("?", np.nan)

    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col + "_grp"] = df[col].map(icd9_group) if col in df else "Unknown"

    df["total_previous_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["has_previous_admission"] = (df["number_inpatient"] > 0).astype(int)

    for med in MEDICATION_COLUMNS:          # drugs not supplied = not prescribed
        if med not in df:
            df[med] = "No"
        df[med] = df[med].fillna("No")
    df["n_meds_changed"] = df[MEDICATION_COLUMNS].isin(["Up", "Down"]).sum(axis=1)
    df["n_meds_active"] = (df[MEDICATION_COLUMNS] != "No").sum(axis=1)

    for col in ["admission_type_id", "admission_source_id", "discharge_disposition_id"]:
        df[col] = df[col].astype(str)

    missing = [c for c in MODEL_COLUMNS if c not in df]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    return df[MODEL_COLUMNS]


def assign_tier(probability: float) -> str:
    if probability >= TIER_CUTOFFS["high"]:
        return "High"
    if probability >= TIER_CUTOFFS["medium"]:
        return "Medium"
    return "Low"


def score(raw: pd.DataFrame) -> List[dict]:
    try:
        X = prepare_features(raw)
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error))
    probabilities = model.predict_proba(X)[:, 1]
    results = []
    for p in probabilities:
        tier = assign_tier(p)
        results.append({
            "risk_score": round(float(p), 3),
            "predicted_readmission_30d": bool(p >= THRESHOLD),
            "risk_tier": tier,
            "recommended_action": ACTIONS[tier],
        })
    return results


# --- Request schema ------------------------------------------------------------
class Patient(BaseModel):
    patient_id: Optional[str] = Field(None, description="Your own reference, returned as-is")
    age: str = Field(..., examples=["[70-80)"])
    admission_type_id: int = Field(..., examples=[1])
    discharge_disposition_id: int = Field(..., examples=[1])
    admission_source_id: int = Field(..., examples=[7])
    time_in_hospital: int = Field(..., ge=1, examples=[5])
    medical_specialty: Optional[str] = Field(None, examples=["InternalMedicine"])
    num_lab_procedures: int = Field(..., ge=0, examples=[45])
    num_procedures: int = Field(..., ge=0, examples=[1])
    num_medications: int = Field(..., ge=0, examples=[18])
    number_outpatient: int = Field(..., ge=0, examples=[0])
    number_emergency: int = Field(..., ge=0, examples=[0])
    number_inpatient: int = Field(..., ge=0, examples=[2])
    diag_1: Optional[str] = Field(None, examples=["428"])
    diag_2: Optional[str] = Field(None, examples=["250.01"])
    diag_3: Optional[str] = Field(None, examples=["401"])
    number_diagnoses: int = Field(..., ge=0, examples=[9])
    medications: Dict[str, str] = Field(
        default_factory=dict,
        description="Drug name -> 'No' / 'Steady' / 'Up' / 'Down'. Drugs left out count as 'No'.",
        examples=[{"insulin": "Up", "metformin": "Steady"}],
    )


# --- API -----------------------------------------------------------------------
app = FastAPI(
    title="30-Day Readmission Risk API",
    description="Random forest model for diabetic patients. Returns a risk score, "
                "risk tier and recommended follow-up. For care-team use, not patients.",
    version="1.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "threshold": THRESHOLD,
        "tier_cutoffs": TIER_CUTOFFS,
    }


def patients_to_frame(patients: List[Patient]) -> pd.DataFrame:
    rows = []
    for p in patients:
        row = p.model_dump(exclude={"medications", "patient_id"})
        unknown = set(p.medications) - set(MEDICATION_COLUMNS)
        if unknown:
            raise HTTPException(422, f"Unknown medication(s): {sorted(unknown)}")
        row.update(p.medications)
        rows.append(row)
    return pd.DataFrame(rows)


@app.post("/predict")
def predict(patient: Patient):
    result = score(patients_to_frame([patient]))[0]
    return {"patient_id": patient.patient_id, **result}


@app.post("/predict/batch")
def predict_batch(patients: List[Patient]):
    results = score(patients_to_frame(patients))
    output = [{"patient_id": p.patient_id, **r} for p, r in zip(patients, results)]
    return sorted(output, key=lambda r: r["risk_score"], reverse=True)


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Upload a CSV with the same columns as diabetic_data.csv.
    Returns every row with its score and tier, highest risk first."""
    try:
        raw = pd.read_csv(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Could not read the file as CSV.")
    results = score(raw)
    id_col = next((c for c in ["patient_nbr", "encounter_id"] if c in raw), None)
    output = [
        {"patient_id": str(raw[id_col].iloc[i]) if id_col else str(i), **r}
        for i, r in enumerate(results)
    ]
    output.sort(key=lambda r: r["risk_score"], reverse=True)
    summary = pd.Series([r["risk_tier"] for r in output]).value_counts().to_dict()
    return {"rows": len(output), "tier_counts": summary, "results": output}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Overall factor importance (shown as a chart on the web page) -------------
def _original_name(feature: str, columns: List[str]) -> str:
    name = feature.split("__", 1)[1]
    for col in sorted(columns, key=len, reverse=True):
        if name == col or name.startswith(col + "_"):
            return col
    return name


_pre = model.named_steps["preprocessor"]
_clf = model.named_steps["classifier"]
_importance: Dict[str, float] = {}
for _feat, _imp in zip(_pre.get_feature_names_out(), _clf.feature_importances_):
    _key = _original_name(_feat, MODEL_COLUMNS)
    _importance[_key] = _importance.get(_key, 0.0) + float(_imp)


@app.get("/feature-importance")
def feature_importance(top: int = 10):
    top = max(1, min(top, 30))
    ranked = sorted(_importance.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return [
        {"feature": k, "label": k.replace("_grp", "").replace("_", " "), "importance": round(v, 4)}
        for k, v in ranked
    ]