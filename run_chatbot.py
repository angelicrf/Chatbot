from qwen_chatbot import QwenChatbot
import time
from datetime import datetime
from pathlib import Path
import csv
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ANALYSIS_FILE = Path("analysis_results.csv")
PREDICTION_DATA_FILE = Path("prediction_weather_data.csv")


def append_analysis_record(record: dict, path: Path | str = ANALYSIS_FILE) -> None:
    path = Path(path)
    fieldnames = [
        "timestamp",
        "model_name",
        "prompt",
        "response",
        "prompt_length",
        "response_length",
        "response_time_ms",
    ]
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def display_response_data(record: dict) -> None:
    print("\n=== Response Data ===")
    print(f"Timestamp: {record['timestamp']}")
    print(f"Prompt length: {record['prompt_length']}")
    print(f"Response length: {record['response_length']}")
    print(f"Response time (ms): {record['response_time_ms']}")
    print("====================\n")


def load_prediction_data(path: Path | str = PREDICTION_DATA_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def find_city_in_prompt(prompt: str, df: pd.DataFrame) -> Optional[str]:
    prompt_l = prompt.lower()
    cities = df["city"].dropna().unique() if not df.empty else []
    for city in cities:
        if str(city).lower() in prompt_l:
            return city
    return None


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


def interactive_loop():
    chatbot = QwenChatbot()
    prediction_df = load_prediction_data()

    print("Interactive weather chatbot. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        # Enforce weather-only prompts
        if not chatbot.is_weather_prompt(prompt):
            print("Rejected: only weather-related questions are allowed. No response generated.")
            continue

        start = time.time()
        response = chatbot.generate_response(prompt)
        end = time.time()

        print(f"Bot: {response}")

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": "Qwen/Qwen3-0.6B",
            "prompt": prompt,
            "response": response,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "response_time_ms": round((end - start) * 1000, 2),
        }

        display_response_data(record)
        append_analysis_record(record)

        # Try to detect a city in the prompt and visualize predictions
        city = find_city_in_prompt(prompt, prediction_df)
        if city:
            print(f"Showing prediction plots for {city}...")
            visualize_city_predictions(city, prediction_df)
        else:
            print("No city from prediction data detected in prompt; skipping visualization.")


if __name__ == "__main__":
    interactive_loop()
