# 🌾 KrishiSahay — AI-Powered Agricultural Query Resolution

> A Generative AI system for answering Indian farmers' queries on crops, pests, fertilizers & government schemes — in multiple languages.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 RAG Pipeline | FAISS semantic search + IBM Watson Granite LLM |
| 🌐 Multilingual | English, Hindi, Telugu, Tamil |
| 📱 Offline PWA | Works offline with service workers & cached answers |
| 🎤 Voice Input | Hands-free queries using Web Speech API |
| 🏛️ Gov. Schemes | PM-KISAN, PMFBY, KCC, PMKSY, eNAM and more |
| ⚡ Fallback Mode | Works without IBM Watson using knowledge base only |
| 🌤️ Weather Widget | Real-time temperature via Open-Meteo API |
| 📊 Feedback System | Per-query thumbs up/down with stats tracking |

---

## 🚀 Quick Start

### Option 1: Manual Setup (Recommended)

```bash
# Clone the repo
git clone https://github.com/RadhikaChundu/cropcode.git
cd cropcode

# ── Backend ──────────────────────────────────────────────
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# ── Frontend (open a new terminal) ───────────────────────
cd ../frontend
python -m http.server 3000
# Open http://localhost:3000
```

### Option 2: Docker Compose

```bash
cp backend/.env.example backend/.env
# Optionally edit .env with your IBM Watson credentials
docker-compose up --build
# Open http://localhost:3000
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Python 3.10+ |
| Vector Search | FAISS + Sentence-Transformers (`all-MiniLM-L6-v2`) |
| LLM | IBM Watson Granite-13b / Fallback KB |
| Frontend | Vanilla JS PWA (no framework) |
| Offline | Service Workers + localStorage caching |
| Styling | Pure CSS with glassmorphism & animations |

---

## 🗂️ Project Structure

```
krishisahay/
├── backend/
│   ├── main.py              # FastAPI app with RAG pipeline
│   ├── requirements.txt     # Python dependencies
│   ├── .env.example         # Environment variable template
│   └── Dockerfile
├── frontend/
│   ├── index.html           # Single-page PWA app
│   ├── manifest.json        # PWA manifest
│   └── sw.js                # Service worker for offline support
├── docker-compose.yml
├── nginx.conf
└── README.md
```

---

## 🌍 IBM Watson Setup (Optional)

Without Watson, the app returns answers directly from its built-in knowledge base (15 articles covering crops, pests, fertilizers, schemes, soil, irrigation, and market).

To enable IBM Watson Granite LLM:

1. Create an [IBM Cloud account](https://cloud.ibm.com)
2. Create a **Watson Machine Learning** instance
3. Get an **API Key** from IAM settings
4. Create a **Project** and copy the Project ID
5. Add to `backend/.env`:
   ```env
   IBM_API_KEY=your_ibm_cloud_api_key
   IBM_PROJECT_ID=your_project_id
   ```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check + FAISS status |
| `/query` | POST | **Main AI query endpoint** |
| `/schemes` | GET | Government schemes list |
| `/categories` | GET | Knowledge base categories |
| `/search?q=...` | GET | Search knowledge base |
| `/stats` | GET | Usage statistics |
| `/feedback` | POST | Submit query feedback |
| `/docs` | GET | Interactive API docs (Swagger) |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How to grow wheat in Rabi season?", "language": "en"}'
```

---

## 📋 Knowledge Base Categories

- 🌾 **Crops** — Wheat, Rice, Maize, seasonal recommendations
- 🐛 **Pests** — Aphids, Fall Armyworm, yellowing leaves
- 🧪 **Fertilizers** — Urea, DAP, SSP dosage & application
- 🏛️ **Schemes** — PM-KISAN, PMFBY, KCC, PMKSY, eNAM
- 🌍 **Soil** — Soil health improvement, composting
- 💧 **Irrigation** — Drip irrigation, water conservation
- 📊 **Market** — MSP, mandi prices, eNAM platform
- ☁️ **Weather** — Monsoon farming, flood & drought management

---

## 📞 Helplines

| Service | Number |
|---------|--------|
| Kisan Call Center | 1800-180-1551 |
| PM-KISAN Helpline | 155261 / 011-23381092 |
| PMFBY Helpline | 1800-180-1551 |

---

## 📄 License

MIT © 2025 KrishiSahay
