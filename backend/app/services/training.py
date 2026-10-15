
from datetime import datetime
from pathlib import Path
import sys, threading, copy, numpy as np, torch
from ..database import SessionLocal, Experiment, TrainingRound, ExperimentMetric, ModelVersion, AuditLog, PrivacySetting
PROJECT_ROOT=Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from src.federated.client import HospitalClient
from src.federated.server import FederatedServer
from src.federated.dp import DifferentialPrivacyEngine

_lock=threading.Lock()

def start_experiment(experiment_id:int):
    with _lock:
        db=SessionLocal()
        try:
            exp=db.get(Experiment,experiment_id)
            if not exp: return
            exp.status="running"; exp.started_at=datetime.utcnow(); db.commit()
            split=PROJECT_ROOT/"data"/"splits"
            clients=[HospitalClient(f"Hospital_{i}",split/f"hospital_{i}_train.csv",22) for i in range(1,5)]
            dp=None
            if exp.privacy_enabled:
                dp=DifferentialPrivacyEngine(clip_norm=1.0,noise_multiplier=0.25,delta=1e-5)
            torch.manual_seed(42); np.random.seed(42)
            server=FederatedServer(input_dim=22)
            res=server.run_rounds(clients,exp.rounds,exp.local_epochs,exp.batch_size,exp.learning_rate,
                                  exp.mu,dp_engine=dp,verbose=False)
            for h in res["history"]:
                db.add(TrainingRound(experiment_id=exp.id,round_number=h["round"],
                                      global_loss=h["avg_client_loss"],average_client_loss=h["avg_client_loss"],
                                      client_drift=None,status="completed"))
            if dp:
                db.add(PrivacySetting(experiment_id=exp.id,clipping_bound=1.0,noise_multiplier=0.25,
                                      delta=1e-5,epsilon_estimate=dp.compute_effective_epsilon(exp.rounds)))
            # evaluate on the held-out set only after all rounds
            import pandas as pd
            test=pd.read_csv(split/"global_test.csv")
            X=test.drop(columns=["target"]).values.astype(np.float32); y=test["target"].values.astype(int)
            metrics,_,_=server.evaluate_final_model(X,y)
            db.add(ExperimentMetric(experiment_id=exp.id,accuracy=metrics["Accuracy"],precision=metrics["Precision"],
                                     recall=metrics["Sensitivity (Recall)"],f1=metrics["F1-Score"],
                                     roc_auc=metrics["ROC-AUC"],specificity=metrics["Specificity"],
                                     sensitivity=metrics["Sensitivity (Recall)"]))
            model_path=PROJECT_ROOT/"models"/f"experiment_{exp.id}.pt"
            torch.save({"state_dict":server.global_model.state_dict(),"input_dim":22,
                        "algorithm":exp.algorithm,"mu":exp.mu,"rounds":exp.rounds,
                        "local_epochs":exp.local_epochs,"batch_size":exp.batch_size,
                        "learning_rate":exp.learning_rate},model_path)
            db.add(ModelVersion(experiment_id=exp.id,model_path=str(model_path.relative_to(PROJECT_ROOT)),
                                version=f"experiment-{exp.id}"))
            exp.status="completed"; exp.completed_at=datetime.utcnow(); db.add(AuditLog(event="TRAINING_COMPLETED",
                description=f"Experiment {exp.id} completed with held-out evaluation only after federated training."))
            db.commit()
        except Exception as exc:
            db.rollback()
            try:
                exp=db.get(Experiment,experiment_id)
                if exp: exp.status="failed"; exp.error=str(exc); db.add(AuditLog(event="TRAINING_FAILED",description=str(exc))); db.commit()
            finally: db.close()
            return
        finally:
            db.close()
