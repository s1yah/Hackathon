# 🛡️ AI-Powered Fraudulent Product Detection System
### Shopee Hackathon — Implementation Guide

---

## 📐 System Architecture Overview

```
[Seller Submits Product Listing]
         │
         ▼
[Async Job Queue (Celery + Redis)]
         │
    ┌────┴────┐
    ▼         ▼
[NLP Engine] [Pricing Anomaly Detector]
    │         │
    └────┬────┘
         ▼
[Seller Trust Score Engine]
         │
         ▼
[Decision Engine]
    │         │         │
    ▼         ▼         ▼
[APPROVED] [SUSPENDED] [REJECTED]
               │
               ▼
    [Moderator Dashboard (HITL)]
```

> [!IMPORTANT]
> The entire pipeline runs **asynchronously** — the seller gets an instant acknowledgment, and the actual review happens in the background to avoid API bottlenecks.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Backend API | **Python + FastAPI** |
| Async Task Queue | **Celery + Redis** |
| NLP | **spaCy**, **HuggingFace Transformers**, **scikit-learn** |
| Pricing Anomaly | **scikit-learn** (Isolation Forest / Z-Score) |
| Database | **PostgreSQL** (listings, sellers, decisions) |
| Seller Trust Score | **Custom weighted scoring (Python)** |
| Moderator Dashboard | **React + Axios** (or simple Flask + Jinja2) |
| Deployment | **Docker + Docker Compose** |

---

## 📦 Recommended Datasets

| Dataset | Purpose | Source |
|---|---|---|
| Amazon Product Reviews / Flipkart Dataset | Fake/deceptive product description training | Kaggle |
| Fake Product Review Dataset (Yelp/Amazon) | NLP deceptive language model training | UCI ML Repo / Kaggle |
| Shopee Open Data (if available via API) | Real listing price benchmarks | Shopee Open Platform |
| E-commerce Price Dataset | Market price baselines for anomaly detection | Kaggle |
| Fake vs Genuine Product Dataset | Binary classification labels | Kaggle |

> [!TIP]
> Search Kaggle for: `"fake product detection"`, `"deceptive reviews NLP"`, `"e-commerce fraud dataset"`. For price data, scrape or use Shopee's API sandbox if the hackathon provides access.

---

## 🔢 Phase 1: Project Setup

### Step 1 — Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install core dependencies
pip install fastapi uvicorn celery redis sqlalchemy psycopg2-binary
pip install transformers spacy scikit-learn pandas numpy
pip install torch  # for HuggingFace models

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### Step 2 — Project Structure

```
shopee-fraud-detection/
├── api/
│   ├── main.py              # FastAPI entry point
│   ├── routes/
│   │   ├── listings.py      # POST /submit-listing
│   │   └── moderator.py     # Moderator dashboard API
│   └── models/              # Pydantic request/response schemas
├── workers/
│   ├── celery_app.py        # Celery configuration
│   ├── nlp_worker.py        # NLP scanning task
│   ├── pricing_worker.py    # Pricing anomaly task
│   └── decision_worker.py   # Final verdict engine
├── ml/
│   ├── nlp_model.py         # NLP classifier
│   ├── pricing_model.py     # Anomaly detector
│   └── trust_score.py       # Seller trust calculator
├── db/
│   ├── database.py          # DB connection
│   └── models.py            # SQLAlchemy ORM models
├── dashboard/               # React or Flask moderator UI
├── docker-compose.yml
└── requirements.txt
```

### Step 3 — Database Schema (PostgreSQL)

```sql
-- Sellers table
CREATE TABLE sellers (
  id UUID PRIMARY KEY,
  name VARCHAR(255),
  verification_status VARCHAR(50),  -- 'verified', 'unverified', 'new'
  trust_score FLOAT DEFAULT 50.0,
  total_listings INT DEFAULT 0,
  flagged_listings INT DEFAULT 0,
  successful_sales INT DEFAULT 0,
  account_age_days INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Product Listings table
CREATE TABLE listings (
  id UUID PRIMARY KEY,
  seller_id UUID REFERENCES sellers(id),
  title VARCHAR(500),
  description TEXT,
  price FLOAT,
  category VARCHAR(100),
  status VARCHAR(50) DEFAULT 'pending',  -- 'approved', 'suspended', 'rejected'
  nlp_flag_score FLOAT,
  price_anomaly_score FLOAT,
  final_decision VARCHAR(50),
  moderator_reviewed BOOLEAN DEFAULT FALSE,
  submitted_at TIMESTAMP DEFAULT NOW(),
  decided_at TIMESTAMP
);

-- Moderator Actions table
CREATE TABLE moderator_actions (
  id UUID PRIMARY KEY,
  listing_id UUID REFERENCES listings(id),
  moderator_id VARCHAR(100),
  action VARCHAR(50),  -- 'approved', 'rejected'
  notes TEXT,
  actioned_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🧠 Phase 2: NLP Engine (Major Point 1)

### Step 4 — Train a Deceptive Text Classifier

The NLP model detects fraudulent/misleading language in product titles and descriptions.

**Approach: Fine-tune a pre-trained transformer (DistilBERT)**

```python
# ml/nlp_model.py
from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizer
import torch

class NLPFraudDetector:
    def __init__(self, model_path="distilbert-base-uncased-finetuned-sst-2-english"):
        # For hackathon: use zero-shot classification as a fast prototype
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        self.fraud_labels = [
            "authentic genuine product",
            "deceptive fake counterfeit product"
        ]
        
        # Keyword flags for rule-based boosting
        self.red_flag_keywords = [
            "100% original", "guaranteed authentic", "factory price",
            "replica", "AAA quality", "same as original", "free shipping no return",
            "limited time", "act now", "guaranteed win"
        ]

    def score(self, title: str, description: str) -> float:
        """Returns a fraud probability score between 0 and 1."""
        text = f"{title}. {description}"
        
        # Transformer-based score
        result = self.classifier(text, self.fraud_labels)
        fraud_prob = result['scores'][result['labels'].index("deceptive fake counterfeit product")]
        
        # Rule-based keyword boosting
        keyword_hits = sum(1 for kw in self.red_flag_keywords if kw.lower() in text.lower())
        keyword_boost = min(keyword_hits * 0.05, 0.3)  # max 30% boost
        
        return min(fraud_prob + keyword_boost, 1.0)
```

> [!TIP]
> For the hackathon, **zero-shot classification** with `facebook/bart-large-mnli` gives you a working prototype without needing labeled training data. If you have time, fine-tune on a Kaggle fake product dataset for higher accuracy.

---

## 📊 Phase 3: Pricing Anomaly Detector (Major Point 1)

### Step 5 — Build the Price Anomaly Model

```python
# ml/pricing_model.py
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd

class PricingAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=42)
        # In production: load from DB by category
        self.category_baselines = {}  # { category: (mean_price, std_price) }
    
    def fit(self, price_data: pd.DataFrame):
        """Train on historical legitimate price data per category."""
        for category, group in price_data.groupby('category'):
            prices = group['price'].values.reshape(-1, 1)
            self.model.fit(prices)
            self.category_baselines[category] = {
                'mean': prices.mean(),
                'std': prices.std(),
                'q1': np.percentile(prices, 25),
                'q3': np.percentile(prices, 75)
            }

    def score(self, price: float, category: str) -> float:
        """Returns anomaly score between 0 (normal) and 1 (highly anomalous)."""
        if category not in self.category_baselines:
            return 0.3  # Neutral score for unknown categories
        
        baseline = self.category_baselines[category]
        mean, std = baseline['mean'], baseline['std']
        
        if std == 0:
            return 0.0
        
        # Z-score based anomaly
        z_score = abs((price - mean) / std)
        
        # Normalize to 0-1 range (z-score of 3+ is considered very anomalous)
        anomaly_score = min(z_score / 3.0, 1.0)
        
        # Extra flag: suspiciously cheap (possible fake/counterfeit)
        if price < baseline['q1'] * 0.3:
            anomaly_score = min(anomaly_score + 0.3, 1.0)
        
        return anomaly_score
```

---

## 🏆 Phase 4: Seller Trust Score Engine (Major Point 2)

### Step 6 — Dynamic Seller Trust Scoring

```python
# ml/trust_score.py

class SellerTrustScorer:
    """
    Trust score: 0 (untrusted) to 100 (fully trusted).
    Thresholds adjust NLP and pricing sensitivity accordingly.
    """
    
    WEIGHTS = {
        'account_age': 0.20,        # Older accounts are more trusted
        'verification_status': 0.30, # Verified sellers get large boost
        'successful_sales_rate': 0.25,
        'flag_rate': 0.25            # Penalize previously flagged listings
    }
    
    def calculate(self, seller: dict) -> float:
        age_days = seller.get('account_age_days', 0)
        verification = seller.get('verification_status', 'new')
        total_listings = seller.get('total_listings', 1)
        flagged = seller.get('flagged_listings', 0)
        sales = seller.get('successful_sales', 0)
        
        # Age score (max after 365 days)
        age_score = min(age_days / 365.0, 1.0) * 100
        
        # Verification score
        verification_score = {
            'verified': 100,
            'unverified': 40,
            'new': 10
        }.get(verification, 10)
        
        # Sales success rate
        sales_rate = (sales / total_listings) if total_listings > 0 else 0
        sales_score = sales_rate * 100
        
        # Flag penalty
        flag_rate = (flagged / total_listings) if total_listings > 0 else 0
        flag_score = (1 - flag_rate) * 100
        
        # Weighted total
        trust_score = (
            age_score       * self.WEIGHTS['account_age'] +
            verification_score * self.WEIGHTS['verification_status'] +
            sales_score     * self.WEIGHTS['successful_sales_rate'] +
            flag_score      * self.WEIGHTS['flag_rate']
        )
        
        return round(min(max(trust_score, 0), 100), 2)
    
    def get_thresholds(self, trust_score: float) -> dict:
        """
        Returns dynamic NLP and pricing thresholds based on trust score.
        High-trust sellers → more lenient. Low-trust → stricter.
        """
        if trust_score >= 75:
            return {'nlp_threshold': 0.75, 'price_threshold': 0.85}
        elif trust_score >= 40:
            return {'nlp_threshold': 0.60, 'price_threshold': 0.70}
        else:  # New or risky accounts
            return {'nlp_threshold': 0.45, 'price_threshold': 0.55}
```

---

## ⚖️ Phase 5: Decision Engine (Major Point 3)

### Step 7 — Verdict Logic

```python
# workers/decision_worker.py

def make_decision(nlp_score: float, price_score: float, 
                  trust_score: float, thresholds: dict) -> str:
    """
    Returns: 'approved', 'suspended' (human review), or 'rejected'
    """
    nlp_thresh   = thresholds['nlp_threshold']
    price_thresh = thresholds['price_threshold']
    
    # High confidence fraud → auto-reject
    if nlp_score >= 0.85 and price_score >= 0.80:
        return 'rejected'
    
    # Clearly fraudulent text alone (very high NLP)
    if nlp_score >= 0.90:
        return 'rejected'
    
    # Both signals triggered beyond their thresholds → human review
    if nlp_score >= nlp_thresh and price_score >= price_thresh:
        return 'suspended'
    
    # Single signal triggered → human review (ambiguous)
    if nlp_score >= nlp_thresh or price_score >= price_thresh:
        return 'suspended'
    
    # Passes all checks
    return 'approved'
```

---

## 🔄 Phase 6: Async Pipeline with Celery (Major Point 1)

### Step 8 — Task Queue Setup

```python
# workers/celery_app.py
from celery import Celery

celery_app = Celery(
    'fraud_detection',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# workers/pipeline_worker.py
from celery_app import celery_app
from ml.nlp_model import NLPFraudDetector
from ml.pricing_model import PricingAnomalyDetector
from ml.trust_score import SellerTrustScorer
from workers.decision_worker import make_decision
from db.database import get_db

nlp_detector    = NLPFraudDetector()
price_detector  = PricingAnomalyDetector()
trust_scorer    = SellerTrustScorer()

@celery_app.task
def process_listing(listing_id: str):
    db = get_db()
    listing = db.query_listing(listing_id)
    seller  = db.query_seller(listing.seller_id)
    
    # Step 1: Score NLP
    nlp_score = nlp_detector.score(listing.title, listing.description)
    
    # Step 2: Score Pricing
    price_score = price_detector.score(listing.price, listing.category)
    
    # Step 3: Get seller trust
    trust_score = trust_scorer.calculate(seller)
    thresholds  = trust_scorer.get_thresholds(trust_score)
    
    # Step 4: Make decision
    decision = make_decision(nlp_score, price_score, trust_score, thresholds)
    
    # Step 5: Update DB
    db.update_listing(listing_id, {
        'nlp_flag_score': nlp_score,
        'price_anomaly_score': price_score,
        'status': decision,
        'final_decision': decision
    })
    
    return {"listing_id": listing_id, "decision": decision}
```

### Step 9 — FastAPI Submission Endpoint

```python
# api/routes/listings.py
from fastapi import APIRouter
from workers.pipeline_worker import process_listing

router = APIRouter()

@router.post("/submit-listing")
async def submit_listing(payload: ListingPayload):
    # 1. Save to DB with status='pending'
    listing_id = save_listing_to_db(payload)
    
    # 2. Dispatch to async queue (non-blocking!)
    process_listing.delay(listing_id)
    
    # 3. Instantly return to seller
    return {
        "listing_id": listing_id,
        "status": "pending",
        "message": "Your listing is under review. You'll be notified shortly."
    }
```

---

## 🖥️ Phase 7: Moderator Dashboard (Major Point 3 — HITL)

### Step 10 — Dashboard API Endpoints

```python
# api/routes/moderator.py
@router.get("/moderator/queue")
async def get_suspended_listings():
    """Returns all listings in 'suspended' state for human review."""
    return db.query_listings(status='suspended')

@router.post("/moderator/decision/{listing_id}")
async def moderator_decision(listing_id: str, action: str, notes: str):
    """Moderator approves or rejects a suspended listing."""
    db.update_listing(listing_id, {'status': action})
    db.log_moderator_action(listing_id, action, notes)
    # Update seller trust score based on outcome
    update_seller_trust(listing_id, action)
    return {"listing_id": listing_id, "final_status": action}
```

### Step 11 — Dashboard UI (Key Screens)

Build a React (or simple HTML+JS) dashboard with these views:

| Screen | Purpose |
|---|---|
| **Queue View** | List of all `suspended` listings, sorted by risk score |
| **Listing Detail View** | Shows title, description, images, NLP score, price score, trust score, and seller history |
| **Decision Panel** | Approve / Reject buttons + notes field |
| **Analytics View** | Total flagged vs approved, accuracy metrics, seller trust distribution |

---

## 🐳 Phase 8: Containerization

### Step 12 — Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on: [redis, postgres]
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000

  worker:
    build: .
    depends_on: [redis, postgres]
    command: celery -A workers.celery_app worker --loglevel=info

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: fraud_db
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: secret
    ports:
      - "5432:5432"
```

---

## 📅 Recommended Hackathon Timeline

| Day | Task |
|---|---|
| **Day 1** | Setup project structure, DB schema, FastAPI skeleton, Docker Compose |
| **Day 2** | Build NLP engine (zero-shot first, train if time allows). Build pricing anomaly detector. |
| **Day 3** | Implement seller trust score + decision engine. Wire up Celery async pipeline. |
| **Day 4** | Build moderator dashboard UI. End-to-end integration testing. |
| **Day 5** | Polish, demo data generation, presentation slides, rehearsal |

---

## 🎯 Demo Strategy

For the hackathon demo, prepare **3 test scenarios**:

1. **Obvious Fake**: Low price, keywords like "replica AAA", new unverified seller → **Auto-Rejected** instantly
2. **Ambiguous Listing**: Slightly suspicious description, mid-range price, semi-trusted seller → Routed to **Moderator Dashboard**  
3. **Legitimate Listing**: Normal price, clear description, verified seller → **Auto-Approved** quickly

> [!NOTE]
> Generate synthetic demo data using Python's `Faker` library combined with your red-flag keyword lists. This ensures a reliable, repeatable demo even without a live Shopee API.

---

## 🚀 Stretch Goals (If Time Permits)

- [ ] **Image Analysis**: Use CLIP or Google Vision API to detect stock photo reuse or watermarked fake product images
- [ ] **Seller Network Graph**: Use NetworkX to detect fraud rings (multiple fake accounts sharing the same behavioral pattern)
- [ ] **Real-time Notifications**: WebSockets to push moderator queue updates live
- [ ] **Explainability**: Show moderators exactly *why* a listing was flagged (SHAP values or attention maps)
- [ ] **A/B Threshold Testing**: Admin panel to tune NLP/price thresholds and see accuracy impact

---

## 📚 Key Libraries Reference

```
fastapi           # REST API framework
celery            # Async task queue
redis             # Message broker for Celery
sqlalchemy        # ORM for PostgreSQL
transformers      # HuggingFace NLP models
spacy             # NLP preprocessing
scikit-learn      # Isolation Forest, Z-score, classical ML
pandas / numpy    # Data manipulation
docker-compose    # Containerization
faker             # Synthetic demo data generation
```
