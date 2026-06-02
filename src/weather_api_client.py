import json
import os
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT_DIR / ".env"
API_KEY_ENV_ALIASES = ["WEATHER_API_KEY", "WEATHERAPI_KEY"]
BASE_WEATHERAPI_URL = "https://api.weatherapi.com/v1/forecast.json"


def find_env_file() -> Optional[Path]:
    search_roots = [ROOT_DIR, Path.cwd(), Path(__file__).resolve().parent]

    for root in search_roots:
        for folder in [root] + list(root.parents):
            candidate = folder / ".env"
            if candidate.exists():
                return candidate
    return None


def load_env_file(env_path: Path = ENV_FILE) -> Dict[str, str]:
    if not env_path.exists():
        found = find_env_file()
        if found:
            env_path = found
        else:
            return {}

    values: Dict[str, str] = {}
    with env_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")

    for key, value in values.items():
        os.environ.setdefault(key, value)

    return values


def get_api_key() -> str:
    load_env_file()
    for key in API_KEY_ENV_ALIASES:
        api_key = os.getenv(key)
        if api_key:
            return api_key

    raise EnvironmentError(
        "Missing environment variable WEATHER_API_KEY or WEATHERAPI_KEY. "
        "Create a .env file with one of these keys set to your API key."
    )


def build_weather_api_url(location: str, days: int = 1, hour: int = 6) -> str:
    api_key = get_api_key()
    query = {
        "q": location,
        "days": days,
        "hour": hour,
        "key": api_key,
    }
    return f"{BASE_WEATHERAPI_URL}?{urlencode(query)}"


def fetch_weather_forecast(location: str, days: int = 1, hour: int = 6) -> Dict:
    url = build_weather_api_url(location, days=days, hour=hour)
    request = Request(url, headers={"User-Agent": "weather-api-client/1.0"})
    with urlopen(request, timeout=15) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def format_forecast(response: Dict, hour: int = 6) -> str:
    location = response.get("location", {})
    forecast = response.get("forecast", {}).get("forecastday", [])
    if not forecast:
        return "No forecast data available."

    day_info = forecast[0]
    hour_info = next(
        (item for item in day_info.get("hour", []) if item.get("time") and item.get("time").endswith(f" {hour}:00")),
        None,
    )

    location_name = location.get("name", "Unknown location")
    condition = day_info.get("day", {}).get("condition", {}).get("text", "Unknown")
    temp_c = day_info.get("day", {}).get("avgtemp_c")
    humidity = day_info.get("day", {}).get("avghumidity")
    precip_mm = day_info.get("day", {}).get("totalprecip_mm")

    forecast_text = [
        f"Weather forecast for {location_name}:",
        f"- Condition: {condition}",
        f"- Average temperature: {temp_c}°C",
        f"- Average humidity: {humidity}%",
        f"- Total precipitation: {precip_mm} mm",
    ]

    if hour_info:
        hour_condition = hour_info.get("condition", {}).get("text", "Unknown")
        hour_temp = hour_info.get("temp_c")
        forecast_text.extend([
            f"- Hour {hour}:00 condition: {hour_condition}",
            f"- Hour {hour}:00 temperature: {hour_temp}°C",
        ])

    return "\n".join(forecast_text)


def main() -> None:
    location = "Paris"
    days = 1
    hour = 6

    forecast_data = fetch_weather_forecast(location=location, days=days, hour=hour)
    print(format_forecast(forecast_data))


if __name__ == "__main__":
    main()
