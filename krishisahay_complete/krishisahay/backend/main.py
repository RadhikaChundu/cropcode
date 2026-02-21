"""
KrishiSahay - GenAI Agricultural Query Resolution System
FastAPI Backend with RAG Pipeline
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json, os, hashlib, time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available - using keyword search")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    build_faiss_index()
    yield

app = FastAPI(title="KrishiSahay API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

KNOWLEDGE_BASE = [
    {"id":"kb_001","category":"crops","question":"Best crops for summer season?","answer":"For summer: Maize (25-35°C), Moong dal (60-75 days), Sunflower, Groundnut (drought-tolerant), Sesame, Cucumber/bitter gourd/ridge gourd. Ensure adequate irrigation.","keywords":["summer","crops","season","kharif","hot"],"tags":["crops","seasons"]},
    {"id":"kb_002","category":"crops","question":"How to grow wheat successfully?","answer":"Wheat: Sow Oct-Nov (Rabi). Loamy soil pH 6-7.5. Seed rate 100-125 kg/ha. Fertilizer: 120N:60P:40K kg/ha. 4-6 irrigations (critical at crown root initiation). Varieties: HD-2967, WH-711, PBW-343. Harvest April-May at 12-14% grain moisture. Yield: 40-50 quintals/ha.","keywords":["wheat","rabi","sowing","gehu","cultivation"],"tags":["crops","rabi","grains"]},
    {"id":"kb_003","category":"crops","question":"Rice paddy cultivation best practices","answer":"Rice: Kharif season. Nursery 21-30 days before transplant. 2-3 seedlings/hill, 20x15cm spacing. 5cm standing water during vegetative stage. Fertilizer: 120:60:60 NPK. Varieties: Swarna, MTU-7029, Pusa Basmati-1121. Watch: BPH, Stem Borer, Blast. Yield 50-60 Q/ha.","keywords":["rice","paddy","dhan","kharif","transplanting"],"tags":["crops","kharif"]},
    {"id":"kb_004","category":"pests","question":"How to control aphids on crops?","answer":"Aphid control: 1) Neem oil 5ml + soap 1ml/liter weekly spray 2) Strong water jet dislodges aphids 3) Yellow sticky traps 4) Chemical: Imidacloprid 0.3ml/liter or Thiamethoxam 0.5g/liter 5) Encourage ladybirds by avoiding broad pesticides. Spray early morning or evening.","keywords":["aphids","insects","pest","mahu","spray"],"tags":["pests","insects"]},
    {"id":"kb_005","category":"pests","question":"How to manage fall armyworm in maize?","answer":"FAW in Maize: Check for window-pane damage, frass in whorls. Bt spray 2g/liter, Trichogramma 1.5 lakh eggs/ha, Pheromone traps 5/ha. Chemical: Emamectin benzoate 0.4g/liter or Spinetoram 0.5ml/liter into whorl. Critical: V4-V8 stage.","keywords":["armyworm","maize","corn","FAW","makka","worm"],"tags":["pests","maize"]},
    {"id":"kb_006","category":"pests","question":"What causes yellow leaves and how to treat?","answer":"Yellow leaves causes: 1) Nitrogen deficiency - bottom leaves yellow → urea 20g/L foliar spray 2) Iron deficiency - interveinal yellowing on young leaves → FeSO4 5g/L 3) Magnesium - veins yellowing → Epsom salt 20g/L 4) Waterlogging - improve drainage 5) Virus - mosaic pattern - remove plants 6) Overwatering - reduce frequency.","keywords":["yellow","chlorosis","nutrients","deficiency","leaves"],"tags":["pests","nutrients"]},
    {"id":"kb_007","category":"fertilizers","question":"How much urea to apply to wheat?","answer":"Urea for wheat: Total 260 kg/ha in 3 splits - 1) At sowing: 87 kg/ha with DAP 2) Crown root initiation (21 days): 87 kg/ha + first irrigation 3) Tillering (45 days): 87 kg/ha. Never apply in standing water. Use neem-coated urea for better efficiency.","keywords":["urea","wheat","fertilizer","nitrogen","khad"],"tags":["fertilizers","wheat"]},
    {"id":"kb_008","category":"fertilizers","question":"Difference between DAP and SSP fertilizer?","answer":"DAP: 18%N + 46%P, ~Rs1350/50kg. Best for cereals. 2.5 bags/ha. SSP: 16%P + 11%S + 19%Ca, ~Rs400/50kg. Best for oilseeds/pulses needing sulphur. 7-8 bags/ha. Use DAP for wheat/rice/maize. Use SSP for groundnut/mustard/soybean - they need sulphur and SSP is more economical.","keywords":["DAP","SSP","phosphorus","fertilizer","phosphate"],"tags":["fertilizers"]},
    {"id":"kb_009","category":"schemes","question":"PM-KISAN scheme details and how to apply?","answer":"PM-KISAN: Rs6,000/year in 3 installments of Rs2,000. All small/marginal farmers with cultivable land eligible. Apply: pmkisan.gov.in or CSC center. Need: Aadhaar, bank passbook, land records. Check status at pmkisan.gov.in. Helpline: 155261 / 011-23381092.","keywords":["PM-KISAN","scheme","government","6000","samman nidhi","subsidy"],"tags":["schemes","government"]},
    {"id":"kb_010","category":"schemes","question":"Pradhan Mantri Fasal Bima Yojana crop insurance","answer":"PMFBY Crop Insurance: Premium: Kharif 2%, Rabi 1.5%, Annual crops 5% of sum insured. Covers natural calamities, pests, diseases, post-harvest losses. Apply at pmfby.gov.in or bank within 2 weeks of sowing. Claim within 72 hours of damage. Helpline: 1800-180-1551.","keywords":["PMFBY","crop insurance","fasal bima","insurance","premium"],"tags":["schemes","insurance"]},
    {"id":"kb_011","category":"schemes","question":"Kisan Credit Card KCC and how to get?","answer":"KCC: Credit at 7% p.a. (4% effective with subvention for timely repayment). Limit: Rs50,000-3,00,000 based on land. Covers seeds, fertilizers, pesticides, equipment. Apply at any bank with Aadhaar, PAN, land records. PM-KISAN beneficiaries get KCC without income proof. Helpline: 1800-180-1111.","keywords":["KCC","kisan credit card","loan","credit","bank"],"tags":["schemes","credit"]},
    {"id":"kb_012","category":"weather","question":"How does monsoon affect farming and precautions?","answer":"Monsoon farming: Pre-monsoon - prepare land, stock inputs. Early (June-July) - sow kharif after 50mm rain. Active monsoon - drainage channels, watch fungal diseases. Late (Sep) - plan rabi timeline. Flood: Use Swarna Sub1 rice, raise nursery beds. Drought: Mulching, drought-tolerant varieties, micro-irrigation.","keywords":["monsoon","rain","kharif","flood","drought","barsaat"],"tags":["weather","seasons"]},
    {"id":"kb_013","category":"soil","question":"How to improve soil health naturally?","answer":"Soil improvement: 1) Vermicompost 5-10 tonnes/ha 2) Green manuring with dhaincha (adds 80-100 kg N/ha) 3) Crop rotation - cereals with legumes 4) Biofertilizers - Rhizobium for pulses, Azospirillum for cereals 5) Mulching - conserves moisture, adds organic matter 6) Reduce tillage. NEVER burn crop residue - destroys soil organisms.","keywords":["soil","health","organic","compost","fertile","mitti"],"tags":["soil","organic"]},
    {"id":"kb_014","category":"irrigation","question":"Drip irrigation - what is it and is it suitable?","answer":"Drip irrigation: Saves 40-60% water, delivers to root zone. Best for: vegetables, fruits, sugarcane, cotton. Cost Rs40,000-80,000/ha. Subsidy: 55-75% under PMKSY for small farmers. Benefits: 50% water saving, 25% fertilizer saving, 20-50% yield increase. Apply at state Horticulture dept or pmksy.gov.in.","keywords":["drip","irrigation","water","sprinkler","PMKSY","sinchai"],"tags":["irrigation","water"]},
    {"id":"kb_015","category":"market","question":"How to get good price for crop and where to sell?","answer":"Best crop prices: 1) MSP - sell to APMC/FCI for guaranteed price (check agricoop.nic.in) 2) eNAM - online competitive bidding at enam.gov.in 3) FPO - join for collective bargaining 4) Direct marketing - local consumers, restaurants 5) Contract farming with companies 6) Cold storage - sell when prices rise. Track prices: Agmarknet.gov.in, Kisan Suvidha app.","keywords":["market","price","MSP","mandi","sell","eNAM","bazar"],"tags":["market","prices"]},
]

SCHEMES = [
    {"name":"PM-KISAN","benefit":"₹6,000/year","category":"Financial Aid","url":"pmkisan.gov.in","color":"#16a34a"},
    {"name":"PMFBY","benefit":"Crop Insurance","category":"Insurance","url":"pmfby.gov.in","color":"#2563eb"},
    {"name":"Kisan Credit Card","benefit":"Low-interest Credit","category":"Credit","url":"","color":"#9333ea"},
    {"name":"PMKSY","benefit":"55-75% Irrigation Subsidy","category":"Infrastructure","url":"pmksy.gov.in","color":"#0891b2"},
    {"name":"eNAM","benefit":"Online Crop Market","category":"Market","url":"enam.gov.in","color":"#d97706"},
    {"name":"Soil Health Card","benefit":"Free Soil Testing","category":"Soil","url":"soilhealth.dac.gov.in","color":"#854d0e"},
    {"name":"PKVY","benefit":"Organic Farming Support","category":"Organic","url":"","color":"#15803d"},
    {"name":"RKVY","benefit":"Agriculture Development Fund","category":"Development","url":"rkvy.nic.in","color":"#dc2626"},
]

faiss_index = None
index_items = []
embedder = None

def build_faiss_index():
    global faiss_index, index_items, embedder
    if not FAISS_AVAILABLE:
        return
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [f"{i['question']} {i['answer']}" for i in KNOWLEDGE_BASE]
        embs = embedder.encode(texts).astype(np.float32)
        faiss.normalize_L2(embs)
        faiss_index = faiss.IndexFlatIP(embs.shape[1])
        faiss_index.add(embs)
        index_items = KNOWLEDGE_BASE
        print(f"FAISS index built: {len(KNOWLEDGE_BASE)} items")
    except Exception as e:
        print(f"FAISS build failed: {e}")

def keyword_search(query, top_k=3):
    qwords = set(query.lower().split())
    scores = []
    for item in KNOWLEDGE_BASE:
        score = 0
        searchable = f"{item['question']} {item['answer']} {' '.join(item['keywords'])}".lower()
        for w in qwords:
            if len(w) > 2 and w in searchable:
                score += 1
            for kw in item['keywords']:
                if w in kw.lower():
                    score += 2
        scores.append((score, item))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [it for sc, it in scores[:top_k] if sc > 0]

def semantic_search(query, top_k=3):
    if faiss_index is None or embedder is None:
        return keyword_search(query, top_k)
    try:
        qvec = embedder.encode([query]).astype(np.float32)
        faiss.normalize_L2(qvec)
        scores, indices = faiss_index.search(qvec, top_k)
        results = [index_items[idx] for i, idx in enumerate(indices[0]) if idx >= 0 and scores[0][i] > 0.15]
        return results if results else keyword_search(query, top_k)
    except:
        return keyword_search(query, top_k)

def generate_answer(query, context_items):
    if not context_items:
        return {"answer":"I couldn't find specific information. Please call Kisan Call Center: 1800-180-1551 for expert advice.","sources":[],"method":"no_match"}
    
    IBM_KEY = os.getenv("IBM_API_KEY")
    IBM_PID = os.getenv("IBM_PROJECT_ID")
    
    if IBM_KEY and IBM_PID and REQUESTS_AVAILABLE:
        try:
            ctx = "\n".join([f"Q: {i['question']}\nA: {i['answer']}" for i in context_items])
            prompt = f"You are KrishiSahay, an agricultural assistant for Indian farmers.\nContext:\n{ctx}\n\nFarmer's question: {query}\n\nAnswer practically in 3-5 sentences:"
            tr = requests.post("https://iam.cloud.ibm.com/identity/token",
                data={"apikey":IBM_KEY,"grant_type":"urn:ibm:params:oauth:grant-type:apikey"},
                headers={"Content-Type":"application/x-www-form-urlencoded"})
            token = tr.json().get("access_token")
            if token:
                r = requests.post("https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29",
                    json={"model_id":"ibm/granite-13b-instruct-v2","input":prompt,"parameters":{"decoding_method":"greedy","max_new_tokens":400},"project_id":IBM_PID},
                    headers={"Authorization":f"Bearer {token}","Content-Type":"application/json"})
                ans = r.json().get("results",[{}])[0].get("generated_text","").strip()
                if ans:
                    return {"answer":ans,"sources":[i['id'] for i in context_items],"method":"ibm_watson"}
        except Exception as e:
            print(f"Watson error: {e}")
    
    return {"answer":context_items[0]['answer'],"sources":[i['id'] for i in context_items],"method":"knowledge_base"}

def detect_language(text):
    if any('\u0900' <= c <= '\u097F' for c in text): return 'hi'
    if any('\u0C00' <= c <= '\u0C7F' for c in text): return 'te'
    if any('\u0B80' <= c <= '\u0BFF' for c in text): return 'ta'
    return 'en'

class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "en"
    category: Optional[str] = None

class FeedbackRequest(BaseModel):
    query_id: str
    rating: int
    comment: Optional[str] = None

feedback_store = []

# Startup is handled via the lifespan context manager above

@app.get("/")
def root():
    return {"message":"KrishiSahay API","version":"1.0.0","status":"healthy"}

@app.get("/health")
def health():
    return {"status":"ok","faiss_available":FAISS_AVAILABLE,"faiss_loaded":faiss_index is not None,"kb_size":len(KNOWLEDGE_BASE),"ibm_configured":bool(os.getenv("IBM_API_KEY"))}

@app.post("/query")
def handle_query(req: QueryRequest):
    start = time.time()
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    detected_lang = detect_language(req.query)
    results = semantic_search(req.query, top_k=3)
    if req.category and req.category != "all":
        cat_r = [r for r in results if r.get('category') == req.category]
        if cat_r: results = cat_r
    response = generate_answer(req.query, results)
    query_id = hashlib.md5(f"{req.query}{time.time()}".encode()).hexdigest()[:8]
    return {
        "query_id": query_id, "query": req.query, "answer": response["answer"],
        "sources": response["sources"], "method": response["method"],
        "language": req.language, "detected_language": detected_lang,
        "related": [{"question":r["question"],"category":r["category"],"id":r["id"]} for r in results[:2]],
        "processing_time": round(time.time()-start, 3),
        "category": results[0]["category"] if results else "general"
    }

@app.get("/schemes")
def get_schemes():
    return {"schemes":SCHEMES,"total":len(SCHEMES)}

@app.get("/categories")
def get_categories():
    cats = {}
    for i in KNOWLEDGE_BASE:
        cats[i['category']] = cats.get(i['category'],0)+1
    return {"categories":[{"name":k,"count":v} for k,v in cats.items()]}

@app.get("/kb/{item_id}")
def get_kb_item(item_id: str):
    for item in KNOWLEDGE_BASE:
        if item['id'] == item_id: return item
    raise HTTPException(404, "Not found")

@app.get("/search")
def search_kb(q: str, limit: int = 5):
    return {"query":q,"results":semantic_search(q, top_k=limit)}

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    feedback_store.append({"query_id":req.query_id,"rating":req.rating,"comment":req.comment,"timestamp":time.time()})
    return {"status":"recorded","total":len(feedback_store)}

@app.get("/stats")
def get_stats():
    ratings = [f["rating"] for f in feedback_store]
    return {"kb_items":len(KNOWLEDGE_BASE),"schemes":len(SCHEMES),"feedback":len(feedback_store),"avg_rating":round(sum(ratings)/len(ratings),2) if ratings else 0,"faiss_enabled":FAISS_AVAILABLE and faiss_index is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
