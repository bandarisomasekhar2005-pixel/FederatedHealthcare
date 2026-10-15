
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from ..database import get_db, Hospital, Experiment, ExperimentMetric, TrainingRound, PrivacySetting, Prediction, Explanation, AuditLog
from ..services.training import start_experiment
from ..services.seed import seed
from ..ml.model_service import ModelService
from ..explainability.shap_service import SHAPService
from ..security.secure_aggregation import demo as secure_demo
from ..security.he_demo import demo as he_demo

router=APIRouter(prefix="/api")
executor=ThreadPoolExecutor(max_workers=1)
model_service=ModelService()
shap_service=SHAPService(model_service)

class PatientInput(BaseModel):
    age: float=Field(...,ge=1,le=120); sex:int=Field(...,ge=0,le=1); cp:int=Field(...,ge=1,le=4)
    trestbps:float=Field(...,ge=50,le=260); chol:float=Field(...,ge=50,le=700); fbs:int=Field(...,ge=0,le=1)
    restecg:int=Field(...,ge=0,le=2); thalach:float=Field(...,ge=50,le=260); exang:int=Field(...,ge=0,le=1)
    oldpeak:float=Field(...,ge=0,le=10); slope:int=Field(...,ge=1,le=3); ca:float=Field(...,ge=0,le=3); thal:int=Field(...,ge=3,le=7)

class ExperimentStart(BaseModel):
    algorithm:str=Field(...,pattern="^(FedAvg|FedProx|FedAvg \\+ DP|FedProx \\+ DP)$")
    rounds:int=Field(20,ge=1,le=50); local_epochs:int=Field(4,ge=1,le=20)
    privacy_enabled:bool=False; secure_aggregation_enabled:bool=False; he_enabled:bool=False

@router.get("/health")
def health(db:Session=Depends(get_db)):
    return {"status":"ok","database":"connected","model_loaded":True,"project":"Privacy-Preserving Federated Learning for Healthcare"}

@router.get("/hospitals")
def hospitals(db:Session=Depends(get_db)):
    return [h.__dict__ | {"_sa_instance_state":None} for h in db.query(Hospital).order_by(Hospital.id).all()]

@router.get("/hospitals/{id}")
def hospital(id:int,db:Session=Depends(get_db)):
    h=db.get(Hospital,id)
    if not h: raise HTTPException(404,"Hospital not found")
    return {k:v for k,v in h.__dict__.items() if k!="_sa_instance_state"}

def _metric_dict(m): return {"accuracy":m.accuracy,"precision":m.precision,"recall":m.recall,"f1":m.f1,"roc_auc":m.roc_auc,
                              "specificity":m.specificity,"sensitivity":m.sensitivity}
@router.get("/experiments")
def experiments(db:Session=Depends(get_db)):
    out=[]
    for e in db.query(Experiment).order_by(Experiment.id.desc()).all():
        out.append({"id":e.id,"experiment_name":e.experiment_name,"algorithm":e.algorithm,"status":e.status,
                    "privacy_enabled":e.privacy_enabled,"secure_aggregation_enabled":e.secure_aggregation_enabled,
                    "he_enabled":e.he_enabled,"rounds":e.rounds,"local_epochs":e.local_epochs,
                    "created_at":e.started_at or e.completed_at,"metrics":_metric_dict(e.metrics[-1]) if e.metrics else None})
    return out

@router.get("/experiments/{id}")
def experiment(id:int,db:Session=Depends(get_db)):
    e=db.get(Experiment,id)
    if not e: raise HTTPException(404,"Experiment not found")
    return {"id":e.id,"algorithm":e.algorithm,"status":e.status,"error":e.error,"rounds":e.rounds,
            "local_epochs":e.local_epochs,"metrics":_metric_dict(e.metrics[-1]) if e.metrics else None,
            "training_rounds":[{"round":r.round_number,"global_loss":r.global_loss,"average_client_loss":r.average_client_loss,
                               "client_drift":r.client_drift} for r in e.rounds_data]}

@router.get("/results")
def results(db:Session=Depends(get_db)):
    rows=[]
    for e in db.query(Experiment).filter(Experiment.status=="completed").order_by(Experiment.id).all():
        if e.metrics: rows.append({"algorithm":e.algorithm,**_metric_dict(e.metrics[-1])})
    return rows

@router.get("/results/comparison")
def comparison(db:Session=Depends(get_db)):
    return results(db)

@router.get("/training/rounds")
def training_rounds(db:Session=Depends(get_db)):
    rows=db.query(TrainingRound).order_by(TrainingRound.experiment_id,TrainingRound.round_number).all()
    return [{"experiment_id":r.experiment_id,"round":r.round_number,"global_loss":r.global_loss,
             "average_client_loss":r.average_client_loss,"client_drift":r.client_drift,"accuracy":r.accuracy} for r in rows]

@router.get("/privacy")
def privacy(db:Session=Depends(get_db)):
    p=db.query(PrivacySetting).order_by(PrivacySetting.id.desc()).first()
    return {"enabled":True,"mechanism":"Update-Level Differential Privacy","clipping_bound":p.clipping_bound if p else 1.0,
            "noise_multiplier":p.noise_multiplier if p else 0.25,"delta":p.delta if p else 1e-5,
            "epsilon_estimate":p.epsilon_estimate if p else 86.667,
            "note":"This is an update-level accounting estimate, not per-example DP-SGD."}

@router.get("/security")
def security():
    return {"secure_aggregation":{"status":"prototype","label":"Secure Aggregation Prototype","demo":secure_demo()},
            "homomorphic_encryption":{"status":"demonstration","label":"Homomorphic Encryption Demonstration","demo":he_demo()},
            "audit_logging":True}

@router.post("/predictions")
def predict(patient:PatientInput,db:Session=Depends(get_db)):
    prob,pred,x=model_service.predict(patient.model_dump())
    p=Prediction(model_version=model_service.version,prediction=pred,probability=prob)
    db.add(p); db.add(AuditLog(event="PREDICTION",description=f"Prediction {p.id or 'new'} generated using {model_service.version}")); db.commit(); db.refresh(p)
    explanation=shap_service.explain(x)
    for row in explanation["all_features"]:
        db.add(Explanation(prediction_id=p.id,method="SHAP",feature=row["feature"],contribution=row["contribution"]))
    db.commit()
    return {"prediction_id":p.id,"model_version":model_service.version,"prediction":pred,"probability":prob,
            "risk":"Higher estimated risk" if pred else "Lower estimated risk",
            "explanation":explanation,
            "disclaimer":"Research/educational prototype; not a clinically validated diagnostic tool."}

@router.get("/explanations/{prediction_id}")
def explanation(prediction_id:int,db:Session=Depends(get_db)):
    p=db.get(Prediction,prediction_id)
    if not p: raise HTTPException(404,"Prediction not found")
    rows=db.query(Explanation).filter(Explanation.prediction_id==prediction_id).order_by(Explanation.id).all()
    return {"prediction_id":prediction_id,"method":"SHAP","features":[{"feature":r.feature,"contribution":r.contribution} for r in rows]}

@router.post("/experiments/start")
def start(req:ExperimentStart,db:Session=Depends(get_db)):
    privacy=req.privacy_enabled or "DP" in req.algorithm
    mu=0.01 if "FedProx" in req.algorithm else 0.0
    e=Experiment(experiment_name=f"Interactive {req.algorithm} — {datetime.utcnow().isoformat()}",
                 algorithm=req.algorithm,privacy_enabled=privacy,secure_aggregation_enabled=req.secure_aggregation_enabled,
                 he_enabled=req.he_enabled,mu=mu,rounds=req.rounds,local_epochs=req.local_epochs,batch_size=16,
                 learning_rate=0.01,status="queued")
    db.add(e); db.commit(); db.refresh(e)
    executor.submit(start_experiment,e.id)
    return {"experiment_id":e.id,"status":"queued"}

@router.get("/experiments/status/{id}")
def experiment_status(id:int,db:Session=Depends(get_db)):
    e=db.get(Experiment,id)
    if not e: raise HTTPException(404,"Experiment not found")
    latest=db.query(TrainingRound).filter(TrainingRound.experiment_id==id).order_by(TrainingRound.round_number.desc()).first()
    return {"id":id,"status":e.status,"current_round":latest.round_number if latest else 0,"total_rounds":e.rounds,
            "global_loss":latest.global_loss if latest else None,"error":e.error}

@router.get("/model/current")
def current_model():
    return {"version":model_service.version,"algorithm":"FedProx + Differential Privacy","path":"models/fedprox_dp_best.pt",
            "input_dim":22,"loaded":model_service.model is not None}

@router.get("/architecture")
def architecture():
    return {"nodes":["Simulated Hospital Clients","Local Training","FedAvg / FedProx","Update-Level DP","Secure Aggregation Prototype",
                     "Global Model","Evaluation","Prediction + SHAP","Web Dashboard"]}

@router.get("/audit")
def audit(db:Session=Depends(get_db)):
    return [{"event":a.event,"description":a.description,"timestamp":a.timestamp} for a in db.query(AuditLog).order_by(AuditLog.id.desc()).limit(100)]
