# 🌾 KrishiSahay — AI-Powered Agricultural Query Resolution

> Generative AI system for answering farmers' queries on crops, pests, fertilizers & government schemes

## ✨ Features
- 🤖 RAG Pipeline: FAISS semantic search + IBM Watson LLM
- 🌐 Multilingual: English, Hindi, Telugu, Tamil
- 📱 PWA: Works offline with cached answers
- 🎤 Voice Input: Hands-free queries for farmers
- 🏛️ Government Schemes: PM-KISAN, PMFBY, KCC, eNAM
- ⚡ Fallback: Works without IBM Watson (knowledge base only)

## 🚀 Quick Start

### Option 1: Manual Setup (Fastest)
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (in new terminal)
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

### Option 2: Docker
```bash
cp backend/.env.example backend/.env
# Edit .env with your IBM credentials (optional)
docker-compose up --build
# Open http://localhost:3000
```

## 🔧 Tech Stack
| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Python 3.10 |
| Vector Search | FAISS + Sentence-Transformers |
| LLM | IBM Watson Granite / Fallback KB |
| Frontend | Vanilla JS PWA |
| Offline | Service Workers + localStorage |

## 🌍 IBM Watson Setup (Optional)
1. Create IBM Cloud account → Watson Machine Learning instance
2. Get API Key from IAM settings
3. Create a Project → copy Project ID
4. Add to `backend/.env`:
   ```
   IBM_API_KEY=your_key
   IBM_PROJECT_ID=your_project_id
   ```

## 📡 API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| /query | POST | Main AI query endpoint |
| /schemes | GET | Government schemes list |
| /categories | GET | Knowledge base categories |
| /search?q=... | GET | Search knowledge base |
| /health | GET | API health check |
| /stats | GET | Usage statistics |

## 📞 Helplines
- Kisan Call Center: 1800-180-1551
- PM-KISAN Helpline: 155261
- PMFBY Helpline: 1800-180-1551
