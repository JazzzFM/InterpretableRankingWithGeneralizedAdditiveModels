from datetime import timedelta
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import os, pandas as pd, mlflow.pyfunc
import logging
import time
from typing import Optional

from src.auth import (
    authenticate_user, create_access_token, get_current_active_user,
    fake_users_db, User, ACCESS_TOKEN_EXPIRE_MINUTES
)
from src.secrets_manager import get_model_uri, get_mlflow_tracking_uri

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load model configuration from secrets
MODEL_URI = get_model_uri()
MLFLOW_TRACKING_URI = get_mlflow_tracking_uri()

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="Credit GAM API", 
    version="1.0.0",
    description="Secure API for credit scoring using GAM models",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
)

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8050").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Trusted hosts (prevent Host header attacks)
trusted_hosts = os.getenv("TRUSTED_HOSTS", "localhost,127.0.0.1,credit-gam-api").split(",")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Load model with retry logic
model = None
max_retries = 3
retry_delay = 5

for attempt in range(max_retries):
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        logger.info(f"Model loaded successfully from {MODEL_URI}")
        break
    except Exception as e:
        logger.warning(f"Attempt {attempt + 1} failed to load model: {e}")
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
        else:
            logger.error("Failed to load model after all attempts")
            model = None

class CreditRequest(BaseModel):
    Age: float = Field(..., ge=18, le=100, description="Age of applicant")
    CreditAmount: float = Field(..., ge=100, le=100000, description="Credit amount requested")
    Duration: float = Field(..., ge=1, le=72, description="Duration in months")

class Token(BaseModel):
    access_token: str
    token_type: str

class PredictionResponse(BaseModel):
    prob_default: float = Field(..., description="Probability of default (0-1)")
    decision: str = Field(..., description="Approval decision: 'approve' or 'review'")
    model_version: Optional[str] = Field(None, description="Model version used")
    request_id: Optional[str] = Field(None, description="Unique request identifier")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    timestamp: str

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return access token."""
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    from datetime import datetime
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/score", response_model=PredictionResponse)
async def score(
    request: CreditRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Score a credit application (protected endpoint)."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available"
        )
    
    try:
        # Create DataFrame for prediction
        df = pd.DataFrame([request.dict()])
        
        # Make prediction
        prediction_start = time.time()
        prob_default = float(model.predict(df)[0])
        prediction_time = time.time() - prediction_start
        
        # Apply business logic
        decision = "approve" if prob_default < 0.25 else "review"
        
        response = PredictionResponse(
            prob_default=prob_default,
            decision=decision,
            model_version=MODEL_URI,
            request_id=f"{current_user.username}_{int(time.time())}"
        )
        
        logger.info(f"Prediction made for user {current_user.username}: {prob_default:.3f} -> {decision}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/score/batch")
async def score_batch(
    requests: list[CreditRequest],
    current_user: User = Depends(get_current_active_user)
):
    """Score multiple credit applications (protected endpoint)."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available"
        )
    
    if len(requests) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 requests per batch"
        )
    
    try:
        results = []
        for i, req in enumerate(requests):
            df = pd.DataFrame([req.dict()])
            prob_default = float(model.predict(df)[0])
            decision = "approve" if prob_default < 0.25 else "review"
            
            results.append(PredictionResponse(
                prob_default=prob_default,
                decision=decision,
                model_version=MODEL_URI,
                request_id=f"{current_user.username}_batch_{int(time.time())}_{i}"
            ))
        
        logger.info(f"Batch prediction completed for user {current_user.username}: {len(results)} requests")
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model/info")
async def model_info(current_user: User = Depends(get_current_active_user)):
    """Get model information (protected endpoint)."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available"
        )
    
    return {
        "model_uri": MODEL_URI,
        "model_type": "LogisticGAM",
        "framework": "PyGAM",
        "loaded_at": time.ctime(),
        "tracking_uri": MLFLOW_TRACKING_URI
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)