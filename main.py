from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, Depends
import joblib
from pydantic import BaseModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load("student_risk_model_0.1-1.0.joblib")
    yield
    app.state.model = None

app = FastAPI(lifespan=lifespan)

def get_model():
    return app.state.model

class PredictRequest(BaseModel):
    average_score: float
    assessments_completed: int
    performance_trend: float
    max_consecutive_misses: int
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
def predict_risk(
        request: PredictRequest,
        model=Depends(get_model)
):
    features = [[
        request.average_score,
        request.assessments_completed,
        request.performance_trend,
        request.max_consecutive_misses
    ]]
    risk_score = model.predict(features)
    return {"risk_score": risk_score[0]}

