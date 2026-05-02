"""
FastAPI Prediction API

REST API for real-time churn prediction and next-best-offer recommendations.

Endpoints:
  GET  /              → API info
  GET  /health        → Health check
  POST /predict/churn → Churn probability + risk level + health score
  POST /predict/nbo   → Best offer recommendation with conversion probability
  GET  /docs          → Auto-generated Swagger UI (built-in)

Shows interviewers: you can deploy models, not just build notebooks.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.churn_pipeline import FEATURE_COLUMNS, load_churn_artifact

# ===== APP SETUP =====
app = FastAPI(
    title="Telco Customer Intelligence API",
    description="""
    🚀 **ML-Powered Telco Customer Intelligence API**
    
    Real-time predictions for churn risk and personalized offer recommendations,
    powered by Gradient Boosting models trained on 7,000+ real customer records.
    
    **Endpoints:**
    - `/predict/churn` — Get churn probability, risk level, and health score
    - `/predict/nbo` — Get personalized next-best-offer recommendation
    - `/health` — API health check
    
    Built with FastAPI • Trained on IBM Telco Churn Data (Kaggle)
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ===== DATA MODELS =====
class CustomerInput(BaseModel):
    """Customer features for prediction."""
    age: int = Field(..., ge=18, le=100, description="Customer age")
    annual_income: float = Field(50000, ge=10000, description="Annual income ($)")
    tenure_months: int = Field(..., ge=0, le=100, description="Months as customer")
    monthly_data_gb: float = Field(10, ge=0, description="Monthly data usage (GB)")
    monthly_minutes: float = Field(500, ge=0, description="Monthly call minutes")
    ott_usage_hours: float = Field(5, ge=0, description="OTT streaming hours/month")
    arpu: float = Field(..., ge=0, description="Average Revenue Per User ($)")
    data_allowance_gb: float = Field(15, ge=0, description="Data plan allowance (GB)")
    minutes_allowance: float = Field(1000, ge=0, description="Minutes allowance")
    satisfaction_score: float = Field(..., ge=1, le=10, description="Satisfaction (1-10)")
    num_complaints_12m: int = Field(0, ge=0, description="Complaints in last 12 months")
    support_tickets_12m: int = Field(0, ge=0, description="Support tickets in 12 months")
    payment_score: float = Field(700.0, ge=300, le=850, description="Payment reliability score (300-850)")
    late_payments_12m: int = Field(0, ge=0, description="Late payments in 12 months")
    campaigns_exposed: int = Field(0, ge=0, description="Marketing campaigns exposed to")
    total_clicks: int = Field(0, ge=0, description="Total campaign clicks")
    total_conversions: int = Field(0, ge=0, description="Total campaign conversions")
    monthly_web_sessions: float = Field(5, ge=0, description="Monthly web sessions")
    monthly_app_sessions: float = Field(3, ge=0, description="Monthly app sessions")
    avg_session_duration_min: float = Field(10, ge=0, description="Avg session duration (min)")
    self_service_transactions: int = Field(2, ge=0, description="Self-service transactions")
    plan_type: str = Field("Standard", description="Plan type: Basic, Standard, or Premium")
    gender: str = Field("Male", description="Gender: Male or Female")

    model_config = {"json_schema_extra": {
        "examples": [{
            "age": 35, "annual_income": 55000, "tenure_months": 18,
            "monthly_data_gb": 12, "monthly_minutes": 600, "ott_usage_hours": 8,
            "arpu": 72, "data_allowance_gb": 15, "minutes_allowance": 1000,
            "satisfaction_score": 4.5, "num_complaints_12m": 3,
            "support_tickets_12m": 5, "payment_score": 650, "late_payments_12m": 2,
            "campaigns_exposed": 3, "total_clicks": 5, "total_conversions": 1,
            "monthly_web_sessions": 8, "monthly_app_sessions": 4,
            "avg_session_duration_min": 12, "self_service_transactions": 3,
            "plan_type": "Basic", "gender": "Male"
        }]
    }}


class ChurnResponse(BaseModel):
    """Churn prediction response."""
    churn_probability: float
    risk_level: str
    health_score: float
    primary_risk_factor: str
    recommendation: str
    confidence: str


class NBOResponse(BaseModel):
    """Next Best Offer response."""
    best_offer: str
    offer_description: str
    conversion_probability: float
    monthly_price_increase: float
    expected_annual_value: float
    customer_segment: str


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    data_source: str
    total_training_records: int
    model_auc: float


# ===== MODEL MANAGEMENT =====
class ModelManager:
    """Manages ML models for the API."""
    
    def __init__(self):
        self.churn_pipeline = None
        self.feature_columns = FEATURE_COLUMNS
        self.is_loaded = False
        self.training_records = 0
        self.model_auc = 0.0
        self.metrics = {}
    
    def load_artifact(self):
        """Load the pre-trained churn pipeline artifact for serving."""
        try:
            artifact = load_churn_artifact()
        except FileNotFoundError as exc:
            print(f"Model artifact not loaded: {exc}")
            self.is_loaded = False
            return
        
        self.churn_pipeline = artifact["pipeline"]
        self.feature_columns = artifact.get("feature_columns", FEATURE_COLUMNS)
        self.metrics = artifact.get("metrics", {})
        self.training_records = int(self.metrics.get("training_records", 0))
        self.model_auc = float(self.metrics.get("roc_auc", 0.0))
        self.is_loaded = True
        print(f"Model artifact loaded. ROC-AUC: {self.model_auc:.4f} | Records: {self.training_records:,}")
    
    def predict_churn(self, customer: CustomerInput) -> dict:
        """Predict churn for a single customer."""
        if not self.is_loaded:
            raise RuntimeError("Model artifact not loaded. Run `python train_model.py` first.")

        customer_df = pd.DataFrame([customer.model_dump()])
        customer_df = customer_df[self.feature_columns]
        churn_prob = float(self.churn_pipeline.predict_proba(customer_df)[0][1])
        
        # Risk level
        if churn_prob >= 0.7:
            risk_level = "Critical"
        elif churn_prob >= 0.5:
            risk_level = "High"
        elif churn_prob >= 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Health score (0-100, higher = healthier)
        health_score = round((1 - churn_prob) * 60 + 
                            (customer.satisfaction_score / 10) * 25 +
                            min(customer.tenure_months / 72, 1) * 15, 1)
        health_score = min(max(health_score, 0), 100)
        
        # Primary risk factor
        risk_factors = {
            'Low Satisfaction': (10 - customer.satisfaction_score) * 2,
            'Short Tenure': max(0, (24 - customer.tenure_months)) * 0.3,
            'High Complaints': customer.num_complaints_12m * 3,
            'Late Payments': customer.late_payments_12m * 2.5,
            'Low Engagement': max(0, (5 - customer.monthly_web_sessions - customer.monthly_app_sessions)) * 1.5
        }
        primary_factor = max(risk_factors, key=risk_factors.get)
        
        # Recommendation
        recommendations = {
            'Low Satisfaction': 'Proactive outreach call + satisfaction survey',
            'Short Tenure': 'Welcome bundle + onboarding support',
            'High Complaints': 'Priority support + issue resolution follow-up',
            'Late Payments': 'Flexible payment plan + auto-pay incentive',
            'Low Engagement': 'App engagement campaign + feature education'
        }
        
        return {
            'churn_probability': round(churn_prob, 4),
            'risk_level': risk_level,
            'health_score': health_score,
            'primary_risk_factor': primary_factor,
            'recommendation': recommendations.get(primary_factor, 'General retention program'),
            'confidence': f"Model ROC-AUC: {self.model_auc:.3f}"
        }
    
    def predict_nbo(self, customer: CustomerInput) -> dict:
        """Predict next best offer for a customer."""
        # Rule-based NBO based on customer profile
        offers = []
        
        if customer.plan_type == 'Basic':
            offers.append({
                'name': 'Upgrade to Standard Plan',
                'description': 'More data, more minutes, better value',
                'monthly_price_increase': 20,
                'score': 0.6 + (customer.annual_income / 100000) * 0.2
            })
        
        if customer.plan_type in ['Basic', 'Standard']:
            offers.append({
                'name': 'Upgrade to Premium Plan',
                'description': 'Unlimited everything + priority support',
                'monthly_price_increase': 30,
                'score': 0.3 + (customer.annual_income / 100000) * 0.3
            })
        
        if customer.ott_usage_hours < 10:
            offers.append({
                'name': 'Streaming Bundle',
                'description': 'TV + Movies streaming package',
                'monthly_price_increase': 15,
                'score': 0.4 + (1 - customer.age / 80) * 0.3
            })
        
        if customer.support_tickets_12m >= 3:
            offers.append({
                'name': 'Premium Tech Support',
                'description': 'Dedicated support with priority queue',
                'monthly_price_increase': 8,
                'score': 0.5 + min(customer.support_tickets_12m / 10, 0.3)
            })
        
        usage_ratio = customer.monthly_data_gb / max(customer.data_allowance_gb, 1)
        if usage_ratio > 0.8:
            offers.append({
                'name': 'Data Boost Package',
                'description': 'Double your data allowance',
                'monthly_price_increase': 10,
                'score': 0.5 + min(usage_ratio - 0.8, 0.3)
            })
        
        if customer.tenure_months >= 24:
            offers.append({
                'name': 'Loyalty Rewards Bundle',
                'description': 'Exclusive perks for loyal customers',
                'monthly_price_increase': 5,
                'score': 0.3 + min(customer.tenure_months / 72, 0.4)
            })
        
        if not offers:
            offers.append({
                'name': 'No Offer Recommended',
                'description': 'Customer profile does not match any current offers',
                'monthly_price_increase': 0,
                'score': 0
            })
        
        # Pick the best offer
        best = max(offers, key=lambda x: x['score'])
        conversion_prob = min(max(best['score'] * 0.3, 0.05), 0.35)
        expected_value = conversion_prob * best['monthly_price_increase'] * 12
        
        # Customer segment
        if customer.arpu >= 80 and customer.satisfaction_score >= 8:
            segment = "Champion"
        elif customer.arpu >= 60:
            segment = "High Value"
        elif customer.arpu >= 40:
            segment = "Mid Value"
        else:
            segment = "Growth Opportunity"
        
        return {
            'best_offer': best['name'],
            'offer_description': best['description'],
            'conversion_probability': round(conversion_prob, 4),
            'monthly_price_increase': best['monthly_price_increase'],
            'expected_annual_value': round(expected_value, 2),
            'customer_segment': segment
        }


# ===== INITIALIZE =====
model_manager = ModelManager()


@app.on_event("startup")
async def startup():
    """Load models on API startup."""
    model_manager.load_artifact()


# ===== ENDPOINTS =====
@app.get("/", tags=["Info"])
async def root():
    """API information and available endpoints."""
    return {
        "name": "Telco Customer Intelligence API",
        "version": "1.0.0",
        "description": "ML-powered churn prediction and offer recommendations",
        "endpoints": {
            "/health": "GET - Health check",
            "/predict/churn": "POST - Churn prediction",
            "/predict/nbo": "POST - Next best offer",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        data_source="IBM Telco Customer Churn (Kaggle)",
        total_training_records=model_manager.training_records,
        model_auc=round(model_manager.model_auc, 4)
    )


@app.post("/predict/churn", response_model=ChurnResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerInput):
    """
    Predict churn probability for a customer.
    
    Returns churn probability (0-1), risk level, health score (0-100),
    primary risk factor, and a recommended retention action.
    """
    try:
        result = model_manager.predict_churn(customer)
        return ChurnResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/nbo", response_model=NBOResponse, tags=["Predictions"])
async def predict_nbo(customer: CustomerInput):
    """
    Recommend the best offer for a customer.
    
    Returns the best offer, conversion probability, expected revenue,
    and customer segment classification.
    """
    try:
        result = model_manager.predict_nbo(customer)
        return NBOResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
