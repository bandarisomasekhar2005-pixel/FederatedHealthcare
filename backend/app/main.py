
import os, sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from .database import init_db
from .services.seed import seed
from .api.routes import router

app=FastAPI(title="Federated Healthcare Research API",version="1.0.0",
            description="Research/educational prototype for privacy-preserving federated heart disease prediction.")
origins=os.getenv("CORS_ORIGINS","http://localhost:5173").split(",")
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
app.include_router(router)

@app.on_event("startup")
def startup():
    init_db(); seed()

@app.get("/")
def root(): return {"message":"Federated Healthcare API","docs":"/docs","health":"/api/health"}
