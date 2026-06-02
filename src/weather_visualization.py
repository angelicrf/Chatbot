from pathlib import Path
import csv
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
TELEMETRY_FILE = ROOT_DIR / "data" / "telemetry.csv"
PREDICTION_DATA_FILE = ROOT_DIR / "data" / "prediction_weather_data.csv"


def load_weather_data(path: Path | str = TELEMETRY_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")
    df = pd.read_csv(path)
    raw_ts = df["timestamp"].astype(str)

    def _normalize_to_strict(s: str) -> str | None:
        s = s.strip()
        # already like YYYY-MM-DDTHH:MM:SSZ -> add microseconds
        m1 = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z$", s)
        if m1:
            return f"{m1.group(1)}.000000Z"
        # already has fractional seconds
        m2 = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d+)Z$", s)
        if m2:
            frac = m2.group(2)
            frac6 = (frac + "000000")[:6]
            return f"{m2.group(1)}.{frac6}Z"
        # try parsing flexibly
        try:
            from dateutil import parser

            parsed = parser.parse(s)
            # produce strict Z-terminated microsecond format in UTC
            if getattr(parsed, "tzinfo", None):
                parsed = parsed.astimezone(tz=None)
            else:
                parsed = parsed
            return parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        except Exception:
            return None

    normalized = raw_ts.apply(_normalize_to_strict)
    invalid = normalized[normalized.isna()]
    if not invalid.empty:
        raise ValueError(f"Found unparseable timestamp values in {path}: {invalid.to_list()}")

    df["timestamp"] = pd.to_datetime(normalized.tolist(), format="%Y-%m-%dT%H:%M:%S.%fZ", utc=True)
    return df


def load_prediction_data(path: Path | str = PREDICTION_DATA_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    raw_ts = df["timestamp"].astype(str)

    def _normalize_to_strict(s: str) -> str | None:
        s = s.strip()
        m1 = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z$", s)
        if m1:
            return f"{m1.group(1)}.000000Z"
        m2 = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d+)Z$", s)
        if m2:
            frac = m2.group(2)
            frac6 = (frac + "000000")[:6]
            return f"{m2.group(1)}.{frac6}Z"
        try:
            from dateutil import parser

            parsed = parser.parse(s)
            if getattr(parsed, "tzinfo", None):
                parsed = parsed.astimezone(tz=None)
            else:
                parsed = parsed
            return parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        except Exception:
            return None

    normalized = raw_ts.apply(_normalize_to_strict)
    invalid = normalized[normalized.isna()]
    if not invalid.empty:
        raise ValueError(f"Found unparseable timestamp values in {path}: {invalid.to_list()}")

    df["timestamp"] = pd.to_datetime(normalized.tolist(), format="%Y-%m-%dT%H:%M:%S.%fZ", utc=True)
    return df


def append_prediction_record(record: dict, path: Path | str = PREDICTION_DATA_FILE) -> None:
    path = Path(path)
    fieldnames = [
        "timestamp",
        "city",
        "prompt",
        "predicted_condition",
        "predicted_temp_c",
        "predicted_humidity_pct",
        "predicted_precip_mm",
    ]
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def visualize_city_predictions(city: str, df: pd.DataFrame) -> None:
    if df.empty:
        print("No prediction data available to visualize.")
        return

    city_df = df[df["city"].str.lower() == str(city).lower()].sort_values("timestamp")
    if city_df.empty:
        print(f"No prediction rows found for {city}.")
        return

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=city_df, x="timestamp", y="predicted_temp_c", marker="o")
    plt.title(f"Predicted Temperature Over Time — {city}")
    plt.xlabel("Time")
    plt.ylabel("Predicted Temp (°C)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.barplot(x=city_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M"), y=city_df["predicted_humidity_pct"], palette="Blues_d")
    plt.title(f"Predicted Humidity Readings — {city}")
    plt.xlabel("Timestamp")
    plt.ylabel("Predicted Humidity (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def print_summary(df: pd.DataFrame) -> None:
    print("Weather telemetry summary:\n")
    print(df.describe(include="all"))
    print("\nSample rows:\n")
    print(df.head(10).to_string(index=False))


def plot_temperature_over_time(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df, x="timestamp", y="temperature_c", hue="location", marker="o")
    plt.title("Temperature Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature (°C)")
    plt.tight_layout()
    plt.show()


def plot_humidity_distribution(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.histplot(df["humidity_pct"], bins=15, kde=True)
    plt.title("Humidity Distribution")
    plt.xlabel("Humidity (%)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_wind_speed_by_location(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="location", y="wind_kph", estimator="mean", ci=None)
    plt.title("Average Wind Speed by Location")
    plt.xlabel("Location")
    plt.ylabel("Wind Speed (km/h)")
    plt.tight_layout()
    plt.show()


def plot_precipitation_over_time(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df, x="timestamp", y="precipitation_mm", hue="location", marker="o")
    plt.title("Precipitation Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Precipitation (mm)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = load_weather_data()
    print_summary(df)
    plot_temperature_over_time(df)
    plot_humidity_distribution(df)
    plot_wind_speed_by_location(df)
    plot_precipitation_over_time(df)


if __name__ == "__main__":
    main()
