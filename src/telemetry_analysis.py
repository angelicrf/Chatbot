import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.qwen_chatbot import QwenChatbot

TELEMETRY_FILE = Path("data/telemetry.csv")
ANALYSIS_FILE = Path("data/analysis_results.csv")


def create_telemetry_record(prompt: str, response: str, start_time: float, end_time: float, model_name: str = "Qwen/Qwen3-0.6B") -> Dict[str, object]:
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "model_name": model_name,
        "prompt": prompt,
        "response": response,
        "prompt_length": len(prompt),
        "response_length": len(response),
        "response_time_ms": round((end_time - start_time) * 1000, 2),
    }


def save_telemetry(records: List[Dict[str, object]], path: Path | str = ANALYSIS_FILE) -> None:
    if not records:
        return

    path = Path(path)
    fieldnames = list(records[0].keys())
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)


def load_telemetry(path: Path | str = ANALYSIS_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def plot_response_length_distribution(df: pd.DataFrame) -> None:
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.histplot(df["response_length"], bins=30, kde=False)
    plt.title("Response Length Distribution")
    plt.xlabel("Response Length")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_prompt_vs_response_length(df: pd.DataFrame) -> None:
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x="prompt_length", y="response_length", hue="model_name", alpha=0.7)
    plt.title("Prompt Length vs Response Length")
    plt.xlabel("Prompt Length")
    plt.ylabel("Response Length")
    plt.tight_layout()
    plt.show()


def plot_requests_over_time(df: pd.DataFrame, freq: str = "H") -> None:
    counts = df.set_index("timestamp").resample(freq).size()
    plt.figure(figsize=(10, 5))
    counts.plot(kind="line", marker="o")
    plt.title(f"Requests Per {freq}")
    plt.xlabel("Time")
    plt.ylabel("Number of Requests")
    plt.tight_layout()
    plt.show()


def run_demo(prompts: List[str]) -> None:
    chatbot = QwenChatbot()
    records: List[Dict[str, object]] = []

    for prompt in prompts:
        start_time = time.time()
        response = chatbot.generate_response(prompt)
        end_time = time.time()

        records.append(create_telemetry_record(prompt, response, start_time, end_time))
        print(f"Prompt: {prompt}\nResponse: {response}\n")

    save_telemetry(records)
    print(f"Saved {len(records)} telemetry records to {ANALYSIS_FILE}")

    df = load_telemetry()
    print(df[["timestamp", "prompt_length", "response_length", "response_time_ms"]].describe())

    plot_response_length_distribution(df)
    plot_prompt_vs_response_length(df)
    plot_requests_over_time(df)


if __name__ == "__main__":
    demo_prompts = [
        "What is the weather forecast for New York this afternoon?",
        "Will it rain in London tonight? /no_think",
        "What is the current humidity and wind speed in Tokyo? /think",
    ]
    run_demo(demo_prompts)
