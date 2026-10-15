
from pathlib import Path
import json, torch
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from src.models.mlp import HeartDiseaseMLP

MODEL_PATH=PROJECT_ROOT/"models"/"fedprox_dp_best.pt"
RAW_PATH=PROJECT_ROOT/"data"/"raw"/"cleveland_raw.csv"
TRAIN_INDEX_PATH=PROJECT_ROOT/"data"/"splits"/"train_indices.csv"

FEATURE_NAMES=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
CONTINUOUS=["age","trestbps","chol","thalach","oldpeak","ca"]
CATEGORICAL=["cp","restecg","slope","thal"]
BINARY=["sex","fbs","exang"]

def _fit_preprocessing():
    raw=pd.read_csv(RAW_PATH, na_values=["?"])
    idx=pd.read_csv(TRAIN_INDEX_PATH)["index"].astype(int).tolist()
    train=raw.iloc[idx].copy()
    for c in FEATURE_NAMES: train[c]=pd.to_numeric(train[c],errors="coerce")
    ca_median=float(train["ca"].median())
    thal_mode=float(train["thal"].mode().iloc[0])
    train["ca"]=train["ca"].fillna(ca_median); train["thal"]=train["thal"].fillna(thal_mode)
    for c in CATEGORICAL: train[c]=train[c].astype(int)
    enc=pd.get_dummies(train,columns=CATEGORICAL,prefix=CATEGORICAL,drop_first=False,dtype=float)
    feature_cols=[c for c in enc.columns if c!="num"]
    means={c:float(enc[c].mean()) for c in CONTINUOUS}
    stds={c:float(enc[c].std(ddof=0)) for c in CONTINUOUS}
    return ca_median,thal_mode,feature_cols,means,stds

CA_MEDIAN,THAL_MODE,FEATURE_NAMES_PROCESSED,MEANS,STDS=_fit_preprocessing()

class ModelService:
    def __init__(self):
        self.model=None
        self.version="fedprox-dp-best-v1"
        self.load()
    def load(self):
        if not MODEL_PATH.exists(): raise FileNotFoundError(f"Model checkpoint missing: {MODEL_PATH}")
        ckpt=torch.load(MODEL_PATH,map_location="cpu")
        self.model=HeartDiseaseMLP(input_dim=int(ckpt.get("input_dim",22)))
        self.model.load_state_dict(ckpt["state_dict"]); self.model.eval()
        self.config=ckpt
    def transform_patient(self, patient:dict):
        row={k:patient[k] for k in FEATURE_NAMES}
        if row["ca"] is None: row["ca"]=CA_MEDIAN
        if row["thal"] is None: row["thal"]=THAL_MODE
        df=pd.DataFrame([row])
        for c in FEATURE_NAMES: df[c]=pd.to_numeric(df[c],errors="coerce")
        for c in CATEGORICAL: df[c]=df[c].astype(int)
        enc=pd.get_dummies(df,columns=CATEGORICAL,prefix=CATEGORICAL,drop_first=False,dtype=float)
        enc=enc.reindex(columns=FEATURE_NAMES_PROCESSED,fill_value=0.0)
        for c in CONTINUOUS:
            enc[c]=(enc[c].astype(float)-MEANS[c])/(STDS[c] if STDS[c] else 1.0)
        return enc.values.astype(np.float32)
    def predict(self, patient:dict):
        x=self.transform_patient(patient)
        with torch.no_grad(): prob=float(self.model(torch.tensor(x)).numpy().ravel()[0])
        return prob, int(prob>=0.5), x
    def predict_array(self,x):
        with torch.no_grad(): return self.model(torch.tensor(np.asarray(x,dtype=np.float32))).numpy().ravel()
