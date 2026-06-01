from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


TELEMETRY_FILE = Path("data/telemetry.csv")


def load_weather_data(path: Path | str = TELEMETRY_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


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
