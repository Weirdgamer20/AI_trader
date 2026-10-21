from __future__ import annotations

import re
from typing import Any
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract


def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess image to maximize OCR legibility for chart fonts."""
    img = image.convert("RGB")
    gray = ImageOps.grayscale(img)
    enhanced = ImageEnhance.Contrast(gray).enhance(1.8)
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    return sharpened


def parse_chart_entities(text: str) -> dict[str, list[Any]]:
    """Parse candidate prices, timestamps, and currency markers from OCR text."""
    # Find price candidates (e.g. ₹82,10,520.50 or 8210520 or 65,00,000)
    price_pattern = re.compile(r"(?:[₹$]\s*)?([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]+)?|[0-9]{4,}(?:\.[0-9]+)?)")
    time_pattern = re.compile(r"\b([0-2]?[0-9]:[0-5][0-9](?::[0-5][0-9])?|\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b")

    prices: list[dict[str, Any]] = []
    seen_floats = set()
    for match in price_pattern.finditer(text):
        raw_num = match.group(1).replace(",", "")
        try:
            val = float(raw_num)
            # Filter reasonable BTC/INR price range (e.g. 10,000 to 1,000,000,000)
            if 10_000 <= val <= 1_000_000_000 and val not in seen_floats:
                seen_floats.add(val)
                prices.append({"raw": match.group(0).strip(), "value": val})
        except ValueError:
            continue

    timestamps = [m.group(0) for m in time_pattern.finditer(text)]

    currencies = []
    for cur in ["BTC", "INR", "USDT", "USD"]:
        if re.search(rf"\b{cur}\b", text, re.IGNORECASE):
            currencies.append(cur)

    return {
        "prices": prices,
        "timestamps": list(dict.fromkeys(timestamps)),
        "currencies": currencies,
    }


def extract_chart_data(uploaded_file) -> dict[str, Any]:
    """Extract visible text, confidence score, and parsed entities from a chart screenshot.

    CRITICAL BOUNDARY: OCR output is strictly for visual inspection and manual confirmation;
    it is never silently mixed into API-verified OHLCV datasets.
    """
    image = Image.open(uploaded_file)
    processed = preprocess_image(image)

    # Extract OCR data table including per-word confidence
    try:
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, config="--psm 6")
        confidences = [int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) >= 0]
        mean_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0.0
        raw_text = pytesseract.image_to_string(processed, config="--psm 6").strip()
        entities = parse_chart_entities(raw_text)
        is_available = True
        error_msg = None
    except Exception as exc:
        mean_conf = 0.0
        raw_text = ""
        entities = {"prices": [], "timestamps": [], "currencies": []}
        is_available = False
        error_msg = f"Tesseract OCR runtime not available or failed: {exc}"

    return {
        "is_available": is_available,
        "error": error_msg,
        "raw_text": raw_text,
        "confidence_pct": mean_conf,
        "parsed_prices": entities["prices"],
        "parsed_timestamps": entities["timestamps"],
        "detected_currencies": entities["currencies"],
        "disclaimer": "OCR data is unverified manual input and must not replace exchange candles.",
    }


def read_chart_screenshot(uploaded_file) -> str:
    """Backward-compatible helper returning raw text string."""
    res = extract_chart_data(uploaded_file)
    return res["raw_text"]
