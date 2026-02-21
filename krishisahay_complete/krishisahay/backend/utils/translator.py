"""
KrishiSahay Translation Utility
Multi-language support for Indian languages
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Language detection patterns (simple keyword-based for demo)
LANG_PATTERNS = {
    "hi": ["क्या", "कैसे", "मेरी", "फसल", "किसान", "बीमारी", "खाद", "सिंचाई", "है", "का"],
    "te": ["ఏమి", "ఎలా", "పంట", "రైతు", "వ్యాధి", "ఎరువు", "నీటి", "ఉంది"],
    "ta": ["என்ன", "எப்படி", "பயிர்", "விவசாயி", "நோய்", "உரம்", "நீர்ப்பாசனம்"],
    "mr": ["काय", "कसे", "पीक", "शेतकरी", "रोग", "खत", "सिंचन"],
    "bn": ["কী", "কীভাবে", "ফসল", "কৃষক", "রোগ", "সার", "সেচ"],
    "kn": ["ಏನು", "ಹೇಗೆ", "ಬೆಳೆ", "ರೈತ", "ರೋಗ", "ಗೊಬ್ಬರ"],
}

# Common agricultural term translations (EN → target)
TRANSLATIONS = {
    "hi": {
        # English terms → Hindi
        "crop": "फसल", "pest": "कीट", "disease": "बीमारी", "fertilizer": "खाद",
        "irrigation": "सिंचाई", "farmer": "किसान", "soil": "मिट्टी", "seed": "बीज",
        "yield": "उपज", "harvest": "कटाई", "nitrogen": "नाइट्रोजन",
        "spray": "छिड़काव", "water": "पानी", "government scheme": "सरकारी योजना",
        "subsidy": "सब्सिडी", "apply": "आवेदन करें", "register": "पंजीकरण करें",
        # Response templates
        "_hello": "नमस्ते! मैं KrishiSahay हूं, आपका कृषि सहायक। आपकी फसल के बारे में मुझे बताएं।",
        "_not_found": "माफ़ करें, इस विषय पर मुझे विशेष जानकारी नहीं मिली। कृपया किसान कॉल सेंटर 1800-180-1551 पर कॉल करें।",
    },
    "te": {
        "crop": "పంట", "pest": "చీడపురుగు", "disease": "వ్యాధి", "fertilizer": "ఎరువు",
        "_hello": "నమస్కారం! నేను KrishiSahay, మీ వ్యవసాయ సహాయకుడిని. మీ పంట గురించి చెప్పండి.",
        "_not_found": "క్షమించండి, ఈ విషయంపై నాకు సమాచారం దొరకలేదు. కిసాన్ కాల్ సెంటర్ 1800-180-1551కి కాల్ చేయండి.",
    },
    "ta": {
        "crop": "பயிர்", "pest": "பூச்சி", "disease": "நோய்", "fertilizer": "உரம்",
        "_hello": "வணக்கம்! நான் KrishiSahay, உங்கள் விவசாய உதவியாளர். உங்கள் பயிர் பற்றி சொல்லுங்கள்.",
        "_not_found": "மன்னிக்கவும், இந்த தலைப்பில் தகவல் கிடைக்கவில்லை. கிசான் கால் சென்டர் 1800-180-1551 அழைக்கவும்.",
    },
}

async def detect_language(text: str) -> str:
    """Detect language from text"""
    try:
        # Try langdetect if available
        from langdetect import detect
        lang = detect(text)
        return lang
    except Exception:
        pass

    # Fallback: check for known script patterns
    for lang, patterns in LANG_PATTERNS.items():
        for pattern in patterns:
            if pattern in text:
                return lang

    return "en"


async def translate_to_english(text: str, source_lang: str) -> str:
    """Translate text to English"""
    if source_lang == "en":
        return text

    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=source_lang, target='en').translate(text)
        if translated:
            return translated
    except Exception as e:
        logger.warning(f"Translation to English failed: {e}")

    return text  # Return original if translation fails


async def translate_from_english(text: str, target_lang: str) -> str:
    """Translate English response to target language"""
    if target_lang == "en":
        return text

    try:
        from deep_translator import GoogleTranslator
        # Split into chunks of 4500 chars (API limit is 5000)
        chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
        translated_chunks = []
        for chunk in chunks:
            t = GoogleTranslator(source='en', target=target_lang).translate(chunk)
            translated_chunks.append(t or chunk)
        return ' '.join(translated_chunks)
    except Exception as e:
        logger.warning(f"Translation from English failed: {e}")

    return text  # Return English if translation fails


def get_language_name(lang_code: str) -> str:
    names = {
        "en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil",
        "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati",
        "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi"
    }
    return names.get(lang_code, lang_code.upper())


SUPPORTED_LANGUAGES = [
    {"code": "en", "name": "English", "native": "English"},
    {"code": "hi", "name": "Hindi", "native": "हिंदी"},
    {"code": "te", "name": "Telugu", "native": "తెలుగు"},
    {"code": "ta", "name": "Tamil", "native": "தமிழ்"},
    {"code": "mr", "name": "Marathi", "native": "मराठी"},
    {"code": "bn", "name": "Bengali", "native": "বাংলা"},
    {"code": "gu", "name": "Gujarati", "native": "ગુજરાતી"},
    {"code": "kn", "name": "Kannada", "native": "ಕನ್ನಡ"},
    {"code": "ml", "name": "Malayalam", "native": "മലയാളം"},
    {"code": "pa", "name": "Punjabi", "native": "ਪੰਜਾਬੀ"},
]
