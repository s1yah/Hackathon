import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import logging

from db.database import init_db
from api.routes import listings, moderator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize DB tables
init_db()

app = FastAPI(
    title="Shopee Fraudulent Product Detection System",
    description="AI-Powered Async Listing Inspection & Human-in-the-Loop Moderator Dashboard",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount router endpoints
app.include_router(listings.router)
app.include_router(moderator.router)

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")
if os.path.exists(DASHBOARD_DIR):
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def get_dashboard(request: Request):
    index_path = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>Fraud Detection System API is running! Dashboard HTML not found.</h2>")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Shopee Fraud Detection Engine"}
