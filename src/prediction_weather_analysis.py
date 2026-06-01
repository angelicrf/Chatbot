from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROMPT_DATA_FILE = Path("data/prediction_weather_data.csv")
WEATHER_KEYWORDS = [
    "rain", "snow", "sunny", "cloudy", "fog", "storm", "thunder",
    "wind", "humidity", "temperature", "hot", "cold", "forecast",
    "weather", "precipitation", "heat", "windy", "drizzle"
]


def extract_prompt_keywords(prompt: str) -> List[str]:
    normalized = prompt.lower()
    found = [keyword for keyword in WEATHER_KEYWORDS if keyword in normalized]
    return sorted(set(found))


def load_prompt_data(path: Path | str = PROMPT_DATA_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt weather data file not found: {path}")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["keywords"] = df["prompt"].apply(lambda text: ", ".join(extract_prompt_keywords(text)))
    return df


def print_prompt_summary(df: pd.DataFrame) -> None:
    print("Prompt weather analysis summary:\n")
    print(df[["timestamp", "city", "prompt", "keywords", "predicted_condition", "predicted_temp_c"]].head(10).to_string(index=False))
    print("\nCity counts:")
    print(df["city"].value_counts())
    print("\nTop prompt keywords:")
    keyword_counts = df["keywords"].str.split(", ").explode().value_counts()
    print(keyword_counts)


def plot_keyword_frequency(df: pd.DataFrame) -> None:
    keyword_counts = df["keywords"].str.split(", ").explode().value_counts()
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.barplot(x=keyword_counts.values, y=keyword_counts.index, palette="crest")
    plt.title("Prompt Keyword Frequency")
    plt.xlabel("Count")
    plt.ylabel("Keyword")
    plt.tight_layout()
    plt.show()


def plot_forecast_by_city(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 5))
    order = df.groupby("city")["predicted_temp_c"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="city", y="predicted_temp_c", order=order, palette="viridis")
    plt.title("Average Predicted Temperature by City")
    plt.xlabel("City")
    plt.ylabel("Predicted Temperature (°C)")
    plt.tight_layout()
    plt.show()


def plot_condition_distribution(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="predicted_condition", order=df["predicted_condition"].value_counts().index, palette="mako")
    plt.title("Predicted Weather Condition Distribution")
    plt.xlabel("Condition")
    plt.ylabel("Number of Prompts")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_temperature_vs_humidity(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="predicted_temp_c", y="predicted_humidity_pct", hue="city", s=120, palette="tab10")
    plt.title("Predicted Temperature vs Humidity by City")
    plt.xlabel("Predicted Temperature (°C)")
    plt.ylabel("Predicted Humidity (%)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = load_prompt_data()
    print_prompt_summary(df)
    plot_keyword_frequency(df)
    plot_forecast_by_city(df)
    plot_condition_distribution(df)
    plot_temperature_vs_humidity(df)


if __name__ == "__main__":
    main()
