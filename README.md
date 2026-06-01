# Chatbot

## Project Summary

This project is a lightweight Python chatbot built around the `Qwen/Qwen3-0.6B` model from Transformers. It loads a causal language model and tokenizer, maintains a simple conversation history, and generates responses for user prompts. The main goal is to enable rapid experimentation with responses and to capture response data for telemetry analysis.

## Telemetry Use Case

The chatbot is a good foundation for telemetry analysis because it already records user prompts and model responses in a structured history list. You can extend this to collect additional metadata such as:

- prompt text
- response text
- response length
- model latency
- generation tokens
- success / error flags
- timestamps

This telemetry data can then be loaded into `pandas` and visualized with charts and graphs to understand model behavior, usage patterns, and prompt effectiveness.

## Installation and Dependencies

### 1. Run the build script

This repository includes a cross-platform setup helper called `build` for Mac/Linux and `build.bat` for Windows. It creates the `.venv` virtual environment and installs all dependencies from `requirements.txt`.

Windows PowerShell:

```powershell
.\build.bat
```

Mac/Linux:

```bash
./build
```

### 2. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Mac/Linux:

```bash
source .venv/bin/activate
```

### 3. Run the chatbot

```powershell
python run_chatbot.py
```

## How to Capture and Analyze Prompt/Response Data

The actual telemetry capture and analysis workflow is implemented in `telemetry_analysis.py`. It records prompt/response pairs, saves them to `analysis_results.csv`, and creates charts using `pandas`, `matplotlib`, and `seaborn`.

For keyword-based weather prompt visualization, use `prediction_weather_analysis.py` with the `prediction_weather_data.csv` sample dataset. That script extracts weather-related keywords from user prompts and visualizes predictions by city, condition, and prompt keyword frequency.

The chatbot is configured to only answer weather-related questions. If you ask a non-weather question, it will return a reminder to keep the conversation on weather topics.

Example telemetry fields include:

- `timestamp`
- `prompt`
- `response`
- `prompt_length`
- `response_length`
- `response_time_ms`
- `model_name`

To run the analysis script:

```powershell
python telemetry_analysis.py
```

## Displaying Data with Pandas and Charts

Visualization is handled by `weather_visualization.py` using the weather sample data in `telemetry.csv`. This file loads weather telemetry, prints summary statistics, and displays charts for:

- temperature over time
- humidity distribution
- average wind speed by location
- precipitation over time

To generate charts, run:

```powershell
python weather_visualization.py
```

## Recommended Next Steps

- Add logging for every prompt and response inside `qwen_chatbot.py`.
- Persist telemetry to `telemetry.csv` or `telemetry.json`.
- Build charts for model response time, prompt complexity, and user sentiment if available.
- Use `seaborn` to compare different prompt modes (`/think`, `/no_think`) visually.

## Notes

- The current project is centered on the chatbot engine and does not yet include telemetry persistence or charting code.
- Installing `pandas`, `matplotlib`, and `seaborn` is required to visualize telemetry data.
