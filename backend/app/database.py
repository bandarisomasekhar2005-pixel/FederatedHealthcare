
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "federated_healthcare.db"
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Hospital(Base):
    __tablename__ = "hospitals"
    id=Column(Integer, primary_key=True)
    name=Column(String(100), nullable=False)
    sample_count=Column(Integer, nullable=False)
    class_0_count=Column(Integer, nullable=False)
    class_1_count=Column(Integer, nullable=False)
    disease_rate=Column(Float, nullable=False)
    training_sample_count=Column(Integer, nullable=True)
    created_at=Column(DateTime, default=datetime.utcnow)

class Experiment(Base):
    __tablename__="experiments"
    id=Column(Integer, primary_key=True)
    experiment_name=Column(String(200), nullable=False)
    algorithm=Column(String(100), nullable=False)
    privacy_enabled=Column(Boolean, default=False)
    secure_aggregation_enabled=Column(Boolean, default=False)
    he_enabled=Column(Boolean, default=False)
    mu=Column(Float, default=0.0)
    rounds=Column(Integer, default=20)
    local_epochs=Column(Integer, default=4)
    batch_size=Column(Integer, default=16)
    learning_rate=Column(Float, default=0.01)
    status=Column(String(30), default="queued")
    started_at=Column(DateTime, nullable=True)
    completed_at=Column(DateTime, nullable=True)
    error=Column(Text, nullable=True)
    metrics=relationship("ExperimentMetric", back_populates="experiment", cascade="all, delete-orphan")
    rounds_data=relationship("TrainingRound", back_populates="experiment", cascade="all, delete-orphan")

class ExperimentMetric(Base):
    __tablename__="experiment_metrics"
    id=Column(Integer, primary_key=True)
    experiment_id=Column(Integer, ForeignKey("experiments.id"))
    accuracy=Column(Float); precision=Column(Float); recall=Column(Float); f1=Column(Float)
    roc_auc=Column(Float); specificity=Column(Float); sensitivity=Column(Float)
    created_at=Column(DateTime, default=datetime.utcnow)
    experiment=relationship("Experiment", back_populates="metrics")

class TrainingRound(Base):
    __tablename__="training_rounds"
    id=Column(Integer, primary_key=True)
    experiment_id=Column(Integer, ForeignKey("experiments.id"))
    round_number=Column(Integer)
    global_loss=Column(Float)
    average_client_loss=Column(Float)
    client_drift=Column(Float, nullable=True)
    accuracy=Column(Float, nullable=True)
    status=Column(String(30), default="completed")
    created_at=Column(DateTime, default=datetime.utcnow)
    experiment=relationship("Experiment", back_populates="rounds_data")

class ModelVersion(Base):
    __tablename__="model_versions"
    id=Column(Integer, primary_key=True)
    experiment_id=Column(Integer, nullable=True)
    model_path=Column(String(500), nullable=False)
    version=Column(String(100), nullable=False)
    created_at=Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__="predictions"
    id=Column(Integer, primary_key=True)
    model_version=Column(String(100), nullable=False)
    prediction=Column(Integer, nullable=False)
    probability=Column(Float, nullable=False)
    created_at=Column(DateTime, default=datetime.utcnow)
    explanation=relationship("Explanation", back_populates="prediction", cascade="all, delete-orphan")

class Explanation(Base):
    __tablename__="explanations"
    id=Column(Integer, primary_key=True)
    prediction_id=Column(Integer, ForeignKey("predictions.id"))
    method=Column(String(50), nullable=False)
    feature=Column(String(100), nullable=False)
    contribution=Column(Float, nullable=False)
    created_at=Column(DateTime, default=datetime.utcnow)
    prediction=relationship("Prediction", back_populates="explanation")

class PrivacySetting(Base):
    __tablename__="privacy_settings"
    id=Column(Integer, primary_key=True)
    experiment_id=Column(Integer, nullable=True)
    clipping_bound=Column(Float)
    noise_multiplier=Column(Float)
    delta=Column(Float)
    epsilon_estimate=Column(Float)

class AuditLog(Base):
    __tablename__="audit_logs"
    id=Column(Integer, primary_key=True)
    event=Column(String(100), nullable=False)
    description=Column(Text, nullable=False)
    timestamp=Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db=SessionLocal()
    try: yield db
    finally: db.close()
