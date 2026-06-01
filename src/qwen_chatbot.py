from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

    def is_weather_prompt(self, user_input: str) -> bool:
        weather_keywords = [
            "weather", "temperature", "forecast", "rain", "snow", "sunny",
            "cloudy", "wind", "windy", "humidity", "storm", "precipitation",
            "thunder", "lightning", "drizzle", "sleet", "blizzard", "heat",
            "cold", "wind chill", "uv index", "climate", "conditions"
        ]
        normalized = user_input.lower()
        return any(keyword in normalized for keyword in weather_keywords)

    def generate_response(self, user_input):
        if not self.is_weather_prompt(user_input):
            response = (
                "I can only answer weather-related questions. "
                "Please ask about weather conditions, forecasts, temperature, wind, humidity, or precipitation."
            )
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": response})
            return response

        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response
