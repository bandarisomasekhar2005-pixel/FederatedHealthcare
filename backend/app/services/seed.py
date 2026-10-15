
from datetime import datetime
from pathlib import Path
import pandas as pd
from ..database import SessionLocal, Hospital, Experiment, ExperimentMetric, PrivacySetting, AuditLog, ModelVersion, init_db

HOSPITALS=[
 {"name":"Hospital 1","sample_count":108,"class_0_count":62,"class_1_count":46,"disease_rate":46/108,"training_sample_count":84},
 {"name":"Hospital 2","sample_count":51,"class_0_count":12,"class_1_count":39,"disease_rate":39/51,"training_sample_count":33},
 {"name":"Hospital 3","sample_count":49,"class_0_count":20,"class_1_count":29,"disease_rate":29/49,"training_sample_count":36},
 {"name":"Hospital 4","sample_count":95,"class_0_count":70,"class_1_count":25,"disease_rate":25/95,"training_sample_count":89},
]
RESULTS=[
 ("FedAvg",0.8361,0.7812,0.8929,0.8333,0.9080,0.7879,0.8929),
 ("FedProx",0.8361,0.8000,0.8571,0.8276,0.9351,0.8182,0.8571),
 ("FedAvg + DP",0.8361,0.7812,0.8929,0.8333,0.9394,0.7879,0.8929),
 ("FedProx + DP",0.8689,0.8333,0.8929,0.8621,0.9426,0.8485,0.8929),
]
def seed():
    init_db(); db=SessionLocal()
    try:
        if db.query(Hospital).count()==0:
            for h in HOSPITALS: db.add(Hospital(**h))
        if db.query(Experiment).count()==0:
            for algo,acc,prec,rec,f1,auc,spec,sens in RESULTS:
                e=Experiment(experiment_name=f"Existing reported result — {algo}",algorithm=algo,
                    privacy_enabled="DP" in algo,secure_aggregation_enabled=False,he_enabled=False,
                    mu=0.01 if "FedProx" in algo else 0.0,rounds=20,local_epochs=4,batch_size=16,
                    learning_rate=0.01,status="completed",completed_at=datetime.utcnow())
                db.add(e); db.flush()
                db.add(ExperimentMetric(experiment_id=e.id,accuracy=acc,precision=prec,recall=rec,f1=f1,
                                        roc_auc=auc,specificity=spec,sensitivity=sens))
                if "DP" in algo:
                    db.add(PrivacySetting(experiment_id=e.id,clipping_bound=1.0,noise_multiplier=0.25,
                                          delta=1e-5,epsilon_estimate=86.667))
        if db.query(ModelVersion).count()==0:
            db.add(ModelVersion(model_path="models/fedprox_dp_best.pt",version="fedprox-dp-best-v1"))
        db.add(AuditLog(event="SEED",description="Database seeded from verified project artifacts and reported experiment results."))
        db.commit()
    finally: db.close()
