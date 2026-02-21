"""
KrishiSahay RAG Engine
Retrieval-Augmented Generation pipeline using FAISS + LLM
"""

import os
import json
import faiss
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Knowledge Base — Agricultural Q&A Data
# ─────────────────────────────────────────────────────────────

AGRICULTURAL_KNOWLEDGE = [
    # CROPS
    {
        "id": "crop_001",
        "category": "crops",
        "subcategory": "rice",
        "title": "Rice Cultivation Best Practices",
        "content": "Rice (Oryza sativa) grows best in temperatures between 20-35°C. It needs 1000-2000mm of rainfall or equivalent irrigation. Transplant seedlings at 3-4 leaf stage, 20-25 days old. Maintain 2-5cm water level during vegetative stage. Apply nitrogen in 3 splits: 50% basal, 25% at tillering, 25% at panicle initiation. Common varieties: IR-64, Swarna, BPT-5204, Jaya.",
        "tags": ["rice", "paddy", "cultivation", "irrigation", "nitrogen"]
    },
    {
        "id": "crop_002",
        "category": "crops",
        "subcategory": "wheat",
        "title": "Wheat Farming Guide",
        "content": "Wheat grows best in cool weather (15-20°C) during grain filling. Sow in October-November in North India. Use 100-125 kg/ha seed rate. Apply 120:60:40 NPK kg/ha. Varieties: HD-2967, PBW-343, WH-711. Irrigate at crown root initiation (21 DAS), tillering (42 DAS), jointing (65 DAS), flowering (90 DAS), and grain filling (105 DAS). Yellow rust (Puccinia striiformis) is the main fungal disease.",
        "tags": ["wheat", "rabi", "winter crop", "NPK", "irrigation"]
    },
    {
        "id": "crop_003",
        "category": "crops",
        "subcategory": "cotton",
        "title": "Cotton Cultivation",
        "content": "Cotton requires 180-200 frost-free days, 21-27°C temperature and well-drained black soil. Bt cotton hybrids dominate in India. Space at 90×60 cm for dryland, 90×45 cm for irrigated. Apply 120:60:60 NPK. Pink bollworm (Pectinophora gossypiella) is the major pest. Use pheromone traps at 5/ha. Harvest when 60% bolls are open.",
        "tags": ["cotton", "Bt cotton", "bollworm", "kharif", "black soil"]
    },
    {
        "id": "crop_004",
        "category": "crops",
        "subcategory": "maize",
        "title": "Maize Growing Guide",
        "content": "Maize (Zea mays) needs warm climate (18-27°C), well-drained loamy soil, pH 5.8-7.0. Sow at 60×25 cm spacing for hybrids. Apply 120:60:40 NPK. Critical irrigation at knee-high, tasseling, and silking stages. Fall Armyworm (Spodoptera frugiperda) is an invasive pest — spray Emamectin benzoate 5% SG @ 0.4g/L at early infestation.",
        "tags": ["maize", "corn", "fall armyworm", "hybrid", "NPK"]
    },
    {
        "id": "crop_005",
        "category": "crops",
        "subcategory": "tomato",
        "title": "Tomato Farming",
        "content": "Tomato grows in 18-27°C, needs well-drained fertile soil pH 6.0-7.0. Transplant 25-30 day old seedlings at 75×60 cm spacing. Apply 150:75:75 NPK kg/ha. Support with stakes at 30cm height. Early blight (Alternaria solani) and late blight (Phytophthora infestans) are major diseases. Spray Mancozeb 75WP @ 2g/L preventively. Use drip irrigation for water efficiency.",
        "tags": ["tomato", "vegetables", "blight", "drip irrigation", "staking"]
    },
    # PESTS
    {
        "id": "pest_001",
        "category": "pests",
        "subcategory": "insects",
        "title": "Rice Brown Planthopper (BPH) Management",
        "content": "Brown Planthopper (Nilaparvata lugens) is the most destructive rice pest causing 'hopper burn'. Symptoms: circular yellowing patches. Management: avoid excessive nitrogen, maintain proper spacing, use BPH-resistant varieties (IR-36, Jaya). Chemical: Imidacloprid 17.8% SL @ 0.25ml/L or Buprofezin 25% SC @ 1ml/L. Avoid broad-spectrum pesticides that kill natural enemies. Apply in evening when hoppers are active.",
        "tags": ["BPH", "planthopper", "rice pest", "hopper burn", "imidacloprid"]
    },
    {
        "id": "pest_002",
        "category": "pests",
        "subcategory": "insects",
        "title": "Aphid Management in Crops",
        "content": "Aphids (Aphididae family) suck sap and transmit viruses. They colonize tender shoots and undersides of leaves. Yellowing, stunting, honeydew, and sooty mold are symptoms. Management: conserve natural enemies (ladybird beetles, lacewings), use yellow sticky traps, spray with neem oil 5ml/L or insecticidal soap. Chemical: Dimethoate 30EC @ 1.7ml/L. Economic threshold: 10-15 aphids/leaf tip. Avoid spraying during flowering.",
        "tags": ["aphid", "sucking pest", "neem oil", "virus vector", "natural enemies"]
    },
    {
        "id": "pest_003",
        "category": "pests",
        "subcategory": "disease",
        "title": "Powdery Mildew Management",
        "content": "Powdery mildew is caused by various fungal species (Erysiphales order). Symptoms: white powdery coating on leaves, stems, fruits. Favored by moderate temperature (20-25°C) and high humidity without wetness. Management: avoid overhead irrigation, improve air circulation, remove infected parts. Spray Wettable Sulfur 80WP @ 3g/L or Hexaconazole 5EC @ 1ml/L at first sign. Repeat every 10-14 days. Use resistant varieties.",
        "tags": ["powdery mildew", "fungal disease", "sulfur", "hexaconazole", "vegetables"]
    },
    {
        "id": "pest_004",
        "category": "pests",
        "subcategory": "disease",
        "title": "Blast Disease in Rice",
        "content": "Rice blast (Pyricularia oryzae) causes diamond-shaped lesions on leaves, neck, and node. Neck blast is most damaging — causes 'dead head' (white empty panicle). Favored by: excess nitrogen, cool nights (20-25°C), heavy dews. Management: use resistant varieties, balanced fertilization, avoid late planting. Fungicide: Tricyclazole 75WP @ 0.6g/L or Isoprothiolane 40EC @ 1.5ml/L. Apply at boot stage preventively.",
        "tags": ["rice blast", "neck blast", "tricyclazole", "Pyricularia", "fungicide"]
    },
    # FERTILIZERS
    {
        "id": "fert_001",
        "category": "fertilizers",
        "subcategory": "NPK",
        "title": "NPK Fertilizer Guide",
        "content": "NPK stands for Nitrogen (N), Phosphorus (P), Potassium (K) — the three primary macronutrients. Nitrogen (Urea 46%N): promotes vegetative growth, dark green color. Deficiency: yellowing of older leaves. Phosphorus (DAP, SSP): root development, flowering. Deficiency: purpling of leaves. Potassium (MOP 60%K2O): disease resistance, fruit quality. Deficiency: leaf scorch at margins. Soil test before applying to optimize dosage and avoid waste.",
        "tags": ["NPK", "urea", "DAP", "MOP", "macronutrients", "soil test"]
    },
    {
        "id": "fert_002",
        "category": "fertilizers",
        "subcategory": "organic",
        "title": "Organic Fertilizers and Compost",
        "content": "Organic fertilizers improve soil health long-term. FYM (Farmyard Manure): apply 10-15 tonnes/ha, NPK ratio approx 0.5:0.25:0.5%. Vermicompost: richer at 2:1:1.5%, apply 2-4 tonnes/ha. Green manure crops (Dhaincha, Sunhemp): incorporate before flowering, adds 80-100 kg N/ha. Biofertilizers: Rhizobium for legumes (fixes 50-200 kg N/ha), PSB (Phosphate Solubilizing Bacteria) releases locked phosphorus. Always cure FYM for 3 months before use.",
        "tags": ["organic", "compost", "FYM", "vermicompost", "biofertilizer", "Rhizobium"]
    },
    {
        "id": "fert_003",
        "category": "fertilizers",
        "subcategory": "micronutrients",
        "title": "Micronutrient Deficiency and Management",
        "content": "Zinc deficiency is most common in Indian soils — causes 'khaira disease' in rice (brown rusty spots, stunted growth). Apply Zinc Sulphate 25kg/ha basal or 0.5% foliar spray. Iron deficiency: interveinal chlorosis on young leaves — spray Ferrous Sulphate 0.5% twice. Boron deficiency in oilseeds and vegetables — apply Borax 10kg/ha or spray 0.2% solution. Micronutrient deficiency is common in alkaline and waterlogged soils.",
        "tags": ["zinc", "iron", "boron", "micronutrients", "khaira", "deficiency"]
    },
    # GOVERNMENT SCHEMES
    {
        "id": "scheme_001",
        "category": "schemes",
        "subcategory": "income_support",
        "title": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
        "content": "PM-KISAN provides income support of ₹6,000 per year to all landholding farmer families in 3 installments of ₹2,000 every 4 months. Eligibility: All landholding farmer families with cultivable land. Exclusions: institutional landholders, income tax payers, retired government pensioners >₹10,000/month. How to apply: Visit nearest CSC (Common Service Centre), register on pmkisan.gov.in, or use PM-KISAN app. Documents: Aadhaar, bank account linked to Aadhaar, land records.",
        "tags": ["PM-KISAN", "income support", "₹6000", "direct benefit transfer", "Aadhaar"]
    },
    {
        "id": "scheme_002",
        "category": "schemes",
        "subcategory": "insurance",
        "title": "PMFBY — Pradhan Mantri Fasal Bima Yojana",
        "content": "PMFBY is crop insurance scheme providing financial support to farmers suffering crop loss/damage due to unforeseen calamities. Premium: Kharif crops — 2%, Rabi crops — 1.5%, Annual commercial/horticulture — 5%. Government pays rest. Coverage: Pre-sowing to post-harvest losses. Claim: Report loss within 72 hours via crop insurance app or helpline 14447. Mandatory for loanee farmers, voluntary for non-loanee. Register at pmfby.gov.in before cutoff date.",
        "tags": ["PMFBY", "crop insurance", "premium", "compensation", "natural calamity"]
    },
    {
        "id": "scheme_003",
        "category": "schemes",
        "subcategory": "soil_health",
        "title": "Soil Health Card Scheme",
        "content": "Soil Health Card Scheme provides farmers a report card of soil nutrient status and recommendations for appropriate dosage of nutrients for improving soil health and fertility. Cards issued every 2 years. The card shows: 12 parameters including N, P, K, pH, EC, organic carbon, sulfur, zinc, iron, copper, manganese, boron. Get free soil testing at nearest Soil Testing Laboratory. Apply online at soilhealth.dac.gov.in or visit Agriculture Department.",
        "tags": ["soil health card", "soil test", "NPK recommendation", "soil testing lab"]
    },
    {
        "id": "scheme_004",
        "category": "schemes",
        "subcategory": "credit",
        "title": "Kisan Credit Card (KCC)",
        "content": "Kisan Credit Card provides short-term credit to farmers for agriculture needs at subsidized interest rate. Loan amount: Based on land holding, cropping pattern. Interest rate: 7% per annum (4% after interest subvention for timely repayment). Revolving credit valid for 5 years. Covers: Crop cultivation, post-harvest expenses, maintenance of farm assets, allied activities (fishery, animal husbandry). Apply at any bank branch, PACS, or Jan Dhan branch. Documents: Aadhaar, land records, passport photo.",
        "tags": ["KCC", "Kisan Credit Card", "loan", "interest subsidy", "credit", "bank"]
    },
    {
        "id": "scheme_005",
        "category": "schemes",
        "subcategory": "mechanization",
        "title": "Sub-Mission on Agricultural Mechanization (SMAM)",
        "content": "SMAM provides subsidy on farm machinery and equipment to small/marginal farmers. Subsidy: 40-50% for general farmers, 50-80% for SC/ST/small/marginal farmers. Eligible machinery: tractors, power tillers, threshers, reapers, sprayers, seeders, drip/sprinkler systems. Custom Hiring Centres (CHC) get up to 80% subsidy. Apply at state agriculture department or Agriculture Infrastructure Fund portal. Priority to FPOs and cooperatives.",
        "tags": ["SMAM", "farm machinery", "subsidy", "tractor", "custom hiring", "mechanization"]
    },
    # WEATHER & IRRIGATION
    {
        "id": "water_001",
        "category": "irrigation",
        "subcategory": "drip",
        "title": "Drip Irrigation System",
        "content": "Drip irrigation delivers water directly to root zone, saving 30-60% water vs flood irrigation. Components: main line, sub-main, lateral pipes, emitters/drippers. Flow rate: 2-8 LPH per dripper. Ideal for: vegetables, fruits, sugarcane, cotton. Subsidy available under PMKSY-PDMC (Per Drop More Crop): 55% for small/marginal, 45% for others. Fertigation through drip increases nutrient use efficiency by 30%. Clean filters every 2 weeks to prevent clogging.",
        "tags": ["drip irrigation", "water saving", "fertigation", "PMKSY", "subsidy"]
    },
    {
        "id": "water_002",
        "category": "irrigation",
        "subcategory": "sprinkler",
        "title": "Sprinkler Irrigation",
        "content": "Sprinkler irrigation mimics rainfall, suits uneven terrain and sandy soils. Types: portable, semi-portable, permanent systems. Application rate: 5-15mm/hour. Best for: wheat, groundnut, pulses, vegetables. Avoid during high wind (>16 km/h) and extreme heat. Sprinkler subsidy available under PMKSY: up to 55% for small farmers. Reduces water use by 25-40% vs flood irrigation. Can apply on slopes up to 12%.",
        "tags": ["sprinkler", "irrigation", "water saving", "wheat", "PMKSY"]
    },
]


class RAGEngine:
    """Core Retrieval-Augmented Generation Engine"""

    def __init__(self):
        self.model = None
        self.index = None
        self.documents = []
        self.initialized = False
        self._load()

    def _load(self):
        """Initialize embedding model and FAISS index"""
        try:
            logger.info("Loading Sentence Transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Model loaded")

            self.documents = AGRICULTURAL_KNOWLEDGE
            texts = [f"{d['title']}. {d['content']}" for d in self.documents]

            logger.info("Building FAISS index...")
            embeddings = self.model.encode(texts, show_progress_bar=False)
            embeddings = np.array(embeddings).astype('float32')

            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)

            self.initialized = True
            logger.info(f"✅ FAISS index built with {len(self.documents)} documents")

        except Exception as e:
            logger.error(f"RAG Engine initialization failed: {e}")
            self.initialized = False

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        """Retrieve top-k relevant documents for a query"""
        if not self.initialized:
            return []

        query_embedding = self.model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0.1:
                doc = self.documents[idx].copy()
                doc['relevance_score'] = float(score)
                results.append(doc)

        return results

    def get_context(self, query: str, top_k: int = 4) -> Tuple[str, List[Dict]]:
        """Get formatted context string and source documents"""
        docs = self.retrieve(query, top_k)
        if not docs:
            return "", []

        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"[Source {i}: {doc['title']}]\n{doc['content']}"
            )

        return "\n\n".join(context_parts), docs


# Singleton instance
_rag_engine: Optional[RAGEngine] = None

def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
