import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

try:
    from .qwen_chatbot import QwenChatbot
    from .weather_api_client import fetch_weather_forecast, format_forecast
    from .weather_visualization import load_prediction_data, visualize_city_predictions, append_prediction_record
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from qwen_chatbot import QwenChatbot
    from weather_api_client import fetch_weather_forecast, format_forecast
    from weather_visualization import load_prediction_data, visualize_city_predictions, append_prediction_record

CITY_EXTRACTION_PATTERN = re.compile(r"\b(?:in|at|for)\s+([A-Za-z ]+?)(?:\?|\.|,|$)", re.I)
HOUR_PATTERN = re.compile(r"\b(?:at|around|by)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I)
DAYS_PATTERN = re.compile(r"\b(\d+)\s+day(?:s)?\b", re.I)


# use `load_prediction_data` from `weather_visualization` for strict parsing


def get_known_cities() -> list[str]:
    prediction_df = load_prediction_data()
    if prediction_df.empty:
        return []
    return list(prediction_df["city"].dropna().unique())


def extract_city(prompt: str, known_cities: Iterable[str]) -> Optional[str]:
    lower_prompt = prompt.lower()
    for city in sorted(set(known_cities), key=len, reverse=True):
        if city.lower() in lower_prompt:
            return city

    match = CITY_EXTRACTION_PATTERN.search(prompt)
    if match:
        return match.group(1).strip()

    return None


def find_city_in_prompt(prompt: str, prediction_df) -> Optional[str]:
    if prediction_df is None or prediction_df.empty:
        return None

    prompt_lower = prompt.lower()
    cities = prediction_df["city"].dropna().unique()
    for city in cities:
        if str(city).lower() in prompt_lower:
            return city
    return None


def extract_days(prompt: str) -> int:
    lower_prompt = prompt.lower()
    if "tomorrow" in lower_prompt:
        return 1
    if "next week" in lower_prompt:
        return 7
    if "weekend" in lower_prompt:
        return 2

    match = DAYS_PATTERN.search(prompt)
    if match:
        days = int(match.group(1))
        return max(1, min(days, 10))

    return 1


def extract_hour(prompt: str) -> int:
    lower_prompt = prompt.lower()
    if "morning" in lower_prompt:
        return 9
    if "afternoon" in lower_prompt:
        return 15
    if "evening" in lower_prompt:
        return 19
    if "night" in lower_prompt or "tonight" in lower_prompt:
        return 21

    match = HOUR_PATTERN.search(prompt)
    if match:
        hour = int(match.group(1))
        suffix = match.group(3)
        if suffix:
            suffix = suffix.lower()
            if suffix == "pm" and hour != 12:
                hour += 12
            if suffix == "am" and hour == 12:
                hour = 0
        return max(0, min(hour, 23))

    return 9


def build_prediction_record(city: str, prompt: str, forecast: dict, hour: int) -> dict:
    forecastday = forecast.get("forecast", {}).get("forecastday", [])
    if not forecastday:
        return {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "city": city,
            "prompt": prompt,
            "predicted_condition": "Unknown",
            "predicted_temp_c": None,
            "predicted_humidity_pct": None,
            "predicted_precip_mm": None,
        }

    day_info = forecastday[0]
    hour_info = next(
        (item for item in day_info.get("hour", []) if item.get("time") and item.get("time").endswith(f" {hour}:00")),
        None,
    )

    if hour_info:
        condition = hour_info.get("condition", {}).get("text", "Unknown")
        temp = hour_info.get("temp_c")
        humidity = hour_info.get("humidity")
        precip = hour_info.get("precip_mm")
    else:
        condition = day_info.get("day", {}).get("condition", {}).get("text", "Unknown")
        temp = day_info.get("day", {}).get("avgtemp_c")
        humidity = day_info.get("day", {}).get("avghumidity")
        precip = day_info.get("day", {}).get("totalprecip_mm")

    return {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "city": city,
        "prompt": prompt,
        "predicted_condition": condition,
        "predicted_temp_c": temp,
        "predicted_humidity_pct": humidity,
        "predicted_precip_mm": precip,
    }


def get_forecast_data(prompt: str, known_cities: Iterable[str]) -> tuple[Optional[str], Optional[int], Optional[dict], Optional[str]]:
    city = extract_city(prompt, known_cities)
    if not city:
        return None, None, None, None

    days = extract_days(prompt)
    hour = extract_hour(prompt)

    try:
        forecast = fetch_weather_forecast(location=city, days=days, hour=hour)
        return city, hour, forecast, None
    except Exception as exc:
        return city, hour, None, str(exc)


def fetch_actual_forecast(prompt: str, known_cities: Iterable[str]) -> Optional[str]:
    city, hour, forecast, error = get_forecast_data(prompt, known_cities)
    if not city:
        return None
    if error:
        return f"Unable to fetch actual forecast for {city}: {error}"

    return format_forecast(forecast, hour=hour)


def interactive_weather_chat() -> None:
    chatbot = QwenChatbot()
    prediction_df = load_prediction_data()
    known_cities = get_known_cities()

    print("Weather chatbot is ready. Ask weather questions and I will answer with the model plus live forecast data.")
    print("Type 'exit', 'quit', or 'bye' to stop.")
    print("")

    while True:
        try:
            prompt = input("User Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "bye"}:
            print("Goodbye.")
            break

        if not chatbot.is_weather_prompt(prompt):
            print("Bot: I can only answer weather-related questions. Please ask about weather forecasts, temperature, rain, wind, humidity, or similar topics.")
            continue

        start = time.time()
        response = chatbot.generate_response(prompt)
        duration_ms = round((time.time() - start) * 1000, 2)

        print(f"Bot: {response}")
        print(f"(Response generated in {duration_ms} ms)")

        forecast_city, forecast_hour, forecast_obj, forecast_error = get_forecast_data(prompt, known_cities)
        if forecast_city and forecast_obj is not None:
            forecast_text = format_forecast(forecast_obj, hour=forecast_hour)
            print("\n=== Live Forecast ===")
            print(forecast_text)
            print("====================\n")

            prediction_cities = {str(c).lower() for c in prediction_df["city"].dropna().unique()} if not prediction_df.empty else set()
            if forecast_city.lower() not in prediction_cities:
                print(f"No existing prediction rows for {forecast_city}. Saving a new prediction row.")
                record = build_prediction_record(forecast_city, prompt, forecast_obj, forecast_hour)
                append_prediction_record(record)
                prediction_df = load_prediction_data()
        elif forecast_city and forecast_error:
            print(f"Unable to fetch actual forecast for {forecast_city}: {forecast_error}")
        else:
            print("No specific city could be extracted from your question for a live forecast.")

        city = find_city_in_prompt(prompt, prediction_df)
        if not city:
            city = extract_city(prompt, known_cities)

        if city:
            print(f"Showing prediction plots for {city}...")
            visualize_city_predictions(city, prediction_df)
        else:
            print("No city from the prediction dataset was detected in the prompt; visualization is skipped.")


if __name__ == "__main__":
    interactive_weather_chat()
