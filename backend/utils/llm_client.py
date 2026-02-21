"""
KrishiSahay LLM Client
Supports IBM Watson ML + Ollama (LLaMA 3) fallback + rule-based fallback
"""

import os
import json
import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

IBM_API_KEY = os.getenv("IBM_API_KEY", "")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID", "")
IBM_REGION = os.getenv("IBM_REGION", "us-south")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

SYSTEM_PROMPT = """You are KrishiSahay, an expert agricultural advisor helping Indian farmers.
You have deep knowledge of crops, pests, fertilizers, irrigation, and government schemes.

Rules:
- Answer ONLY agriculture-related questions
- Be specific, practical, and actionable
- Use simple language farmers can understand
- Mention specific chemical names, dosages, and timings when relevant
- Always mention safety precautions for pesticide use
- If a government scheme is relevant, mention it
- If information is from the context, use it; if not, use general agricultural knowledge
- Keep answers concise but complete (150-250 words)
- End with one key takeaway or action item

IMPORTANT: Do not answer non-agricultural questions. Politely redirect to farming topics."""


def build_prompt(query: str, context: str, language: str = "en") -> str:
    lang_instruction = ""
    if language != "en":
        lang_names = {
            "hi": "Hindi", "te": "Telugu", "ta": "Tamil",
            "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati",
            "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi"
        }
        lang_name = lang_names.get(language, language)
        lang_instruction = f"\n\nIMPORTANT: Respond in {lang_name} language only."

    if context:
        return f"""Based on the following agricultural knowledge:

{context}

Farmer's Question: {query}

Please provide a helpful, practical answer for the farmer.{lang_instruction}"""
    else:
        return f"""Farmer's Question: {query}

Please provide a helpful, practical agricultural answer.{lang_instruction}"""


async def call_ibm_watson(query: str, context: str, language: str = "en") -> Optional[str]:
    """Call IBM Watson Machine Learning API"""
    if not IBM_API_KEY or not IBM_PROJECT_ID:
        return None

    try:
        # Get IAM token
        async with httpx.AsyncClient(timeout=30) as client:
            token_resp = await client.post(
                "https://iam.cloud.ibm.com/identity/token",
                data={
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                    "apikey": IBM_API_KEY
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            token = token_resp.json().get("access_token")
            if not token:
                return None

            # Call generation endpoint
            url = f"https://{IBM_REGION}.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
            payload = {
                "model_id": "ibm/granite-13b-instruct-v2",
                "input": build_prompt(query, context, language),
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": 400,
                    "min_new_tokens": 50,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop_sequences": ["\n\n\n"]
                },
                "project_id": IBM_PROJECT_ID
            }

            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            )
            result = resp.json()
            generated = result.get("results", [{}])[0].get("generated_text", "").strip()
            if generated:
                logger.info("✅ IBM Watson response received")
                return generated

    except Exception as e:
        logger.warning(f"IBM Watson call failed: {e}")

    return None


async def call_ollama(query: str, context: str, language: str = "en") -> Optional[str]:
    """Call local Ollama instance"""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "system": SYSTEM_PROMPT,
                    "prompt": build_prompt(query, context, language),
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 400}
                }
            )
            result = resp.json()
            answer = result.get("response", "").strip()
            if answer:
                logger.info("✅ Ollama response received")
                return answer
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
    return None


def rule_based_answer(query: str, context: str) -> str:
    """Smart rule-based fallback using retrieved context"""
    if not context:
        return (
            "I couldn't find specific information about your query in my knowledge base. "
            "Please consult your local Agricultural Extension Officer (Krishi Vigyan Kendra) "
            "or call the Kisan Call Centre at 1800-180-1551 (toll-free) for expert advice."
        )

    # Extract first relevant context
    lines = context.split('\n')
    answer_lines = []
    for line in lines:
        if line.startswith('[Source') or not line.strip():
            continue
        answer_lines.append(line.strip())
        if len(' '.join(answer_lines).split()) > 120:
            break

    base = ' '.join(answer_lines)[:600]
    return (
        f"{base}\n\n"
        "📞 For more help: Kisan Call Centre 1800-180-1551 (toll-free, 6AM-10PM)"
    )


async def generate_answer(query: str, context: str, language: str = "en") -> str:
    """Main answer generation — tries IBM Watson → Ollama → Rule-based"""

    # Try IBM Watson first
    answer = await call_ibm_watson(query, context, language)
    if answer:
        return answer

    # Try Ollama
    answer = await call_ollama(query, context, language)
    if answer:
        return answer

    # Fallback to rule-based
    logger.info("Using rule-based fallback")
    return rule_based_answer(query, context)
