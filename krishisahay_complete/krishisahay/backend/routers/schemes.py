"""Government Schemes router"""
from fastapi import APIRouter
router = APIRouter()

SCHEMES = [
    {"id": "pm_kisan", "name": "PM-KISAN", "benefit": "₹6,000/year income support", "category": "income", "link": "https://pmkisan.gov.in"},
    {"id": "pmfby", "name": "PMFBY", "benefit": "Crop insurance at 2% premium", "category": "insurance", "link": "https://pmfby.gov.in"},
    {"id": "kcc", "name": "Kisan Credit Card", "benefit": "7% interest credit", "category": "credit", "link": "https://www.nabard.org"},
    {"id": "soil_health", "name": "Soil Health Card", "benefit": "Free soil testing", "category": "soil", "link": "https://soilhealth.dac.gov.in"},
    {"id": "smam", "name": "SMAM", "benefit": "40-80% subsidy on machinery", "category": "mechanization", "link": "https://agrimachinery.nic.in"},
    {"id": "pmksy", "name": "PMKSY", "benefit": "55% subsidy on drip/sprinkler", "category": "irrigation", "link": "https://pmksy.gov.in"},
]

@router.get("/schemes")
async def get_schemes(category: str = None):
    if category:
        return {"schemes": [s for s in SCHEMES if s["category"] == category]}
    return {"schemes": SCHEMES}
