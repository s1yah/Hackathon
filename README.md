# 🛡️ Shopee Sentinel — AI-Powered Fraudulent Product Detection System

> An intelligent, multi-layered security pipeline and Human-In-The-Loop (HITL) moderator workstation that automatically inspects e-commerce product listings for counterfeit language, pricing anomalies, and seller fraud.

---

## 📌 Project Overview (What is this?)

When sellers post products on online marketplaces like Shopee, some listings may be scams, fake/replica items, or suspiciously priced. 

**Shopee Sentinel** acts as an **AI Security Team** that checks product listings before they go live:
* 🟢 **Auto-Approved**: Clean, genuine items with normal prices and verified sellers.
* 🔴 **Auto-Rejected**: Clear scams (e.g., $15 iPhone listing + fake keywords + unverified seller).
* 🟡 **Suspended for Review**: Borderline items where the AI isn't 100% sure. These are sent to a **Moderator Dashboard** for human review.

---

## 🧠 Technologies & AI Models Used

* **Programming Language**: Python 3.10+
* **Backend API**: FastAPI + Uvicorn
* **Database**: SQLite (local dev/demo) / PostgreSQL (production) via SQLAlchemy ORM
* **Frontend**: HTML5, CSS3 (Modern Glassmorphism Dark Mode), JavaScript (Vanilla)
* **Containerization**: Docker & Docker Compose

### 🤖 AI & Machine Learning Models
1. **NLP Text Inspector** ([ml/nlp_model.py](file:///c:/Users/maric/Downloads/Hackathon/ml/nlp_model.py)):
   * Uses **BART (`facebook/bart-large-mnli`)** for zero-shot text classification to spot deceptive language.
   * Combined with TF-IDF classification trained on 40,000+ product reviews and red-flag keyword heuristics (e.g., *"AAA quality"*, *"100% original"*, *"no return"*).
2. **Pricing Anomaly Detector** ([ml/pricing_model.py](file:///c:/Users/maric/Downloads/Hackathon/ml/pricing_model.py)):
   * Uses Z-Score statistical analysis fitted on category market datasets ([Plan/materials/product_prices.csv](file:///c:/Users/maric/Downloads/Hackathon/Plan/materials/product_prices.csv)) to flag extreme price deviations and deep discount scams.
3. **Seller Trust Engine** ([ml/trust_score.py](file:///c:/Users/maric/Downloads/Hackathon/ml/trust_score.py)):
   * Computes a dynamic reputation score ($0\text{--}100$) based on account age, verification status, and past sales.

---

## 🚀 Getting Started (How to Run)

### 1. Install Dependencies
Ensure you have Python installed, then install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Seed the Database & Fit Data Baselines
Run the setup script to initialize the database schema and fit category price baselines from the dataset CSV files:

```bash
python scripts/seed_demo.py
```

### 3. Start the Web Server & Moderator Dashboard
Launch the FastAPI server:

```bash
python -m uvicorn api.main:app --port 8000
```

### 4. Open the App in Your Browser
Open your browser and navigate to:
* 🖥️ **Moderator Dashboard**: `http://localhost:8000/`
* 📑 **API Documentation (Swagger)**: `http://localhost:8000/docs`

---

## 🧪 Running AI Training & Inspection Scripts

### 1. Train the NLP Fraud Model
To train the NLP deceptive text model on 40,000+ labeled dataset examples:

```bash
python scripts/train_nlp_model.py
```
* **Output**: Trains a classifier achieving **89.44% Accuracy** and saves model files to `ml/saved_models/`.

### 2. Run CLI Test Examples
To test how the AI scores different sample products (Fake, Ambiguous, Genuine) directly in your terminal:

```bash
python scripts/check_examples.py
```

---

## 📂 Project Directory Structure

```
Hackathon/
├── api/                   # FastAPI routes and request/response models
│   ├── main.py            # Main application entry point
│   └── routes/            # Listings, moderator queue & demo API routes
├── db/                    # Database models and session management
│   ├── database.py        # SQLAlchemy database connection setup
│   └── models.py          # Seller, Listing, and ModeratorAction ORM models
├── ml/                    # Machine Learning components
│   ├── nlp_model.py       # Deceptive text classifier (BART + heuristics)
│   ├── pricing_model.py   # Price Z-score anomaly detector
│   └── trust_score.py     # Dynamic seller trust scorer
├── workers/               # Async task execution & decision engine logic
│   ├── decision_worker.py # Traffic light decision engine (Approve/Suspend/Reject)
│   └── pipeline_worker.py # Pipeline execution manager
├── dashboard/             # Web Moderator Dashboard UI (HTML, CSS, JS)
│   └── index.html         # Interactive workstation interface
├── scripts/               # Training, seeding, and demo test scripts
│   ├── train_nlp_model.py # NLP dataset training script
│   ├── seed_demo.py       # Database seeding & baseline fitting script
│   └── check_examples.py  # CLI test inspection script
├── Plan/materials/        # Training datasets (CSVs) & hackathon guides
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Multi-service orchestration (Postgres, Redis, API)
└── requirements.txt       # Python package dependencies
```

---

## 🎯 Hackathon Demo Features

In the live **Moderator Dashboard** (`http://localhost:8000/`), you can test **3 built-in demo buttons**:
1. 🔴 **1. Obvious Fake**: Instantly triggers a $15 counterfeit iPhone listing $\rightarrow$ Auto-Rejected.
2. 🟡 **2. Ambiguous Listing**: Triggers a mid-range refurbished headphone listing $\rightarrow$ Suspended and routed to the Human Moderator Queue for inspection.
3. 🟢 **3. Legitimate Listing**: Triggers an official Sony store listing $\rightarrow$ Auto-Approved.

## Datasets Used

1. Fake Reviews Dataset - https://www.kaggle.com/datasets/muqaddasejaz/fake-reviews-dataset
3. E-commerce Product Prices - https://www.kaggle.com/datasets/steve1215rogg/e-commerce-dataset
4. Amazon E-commerce Products & Reviews Dataset (Backup for Testing) - https://www.kaggle.com/datasets/lazylad99/amazon-e-commerce-product-and-review-dataset

* Note: The datasets were cleaned before training. 
