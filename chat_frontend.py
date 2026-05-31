from qwen_chatbot import QwenChatbot


def run_chat_interface():
    print("Qwen Chatbot Frontend")
    print("Type a message and press Enter to chat. Type 'quit' or 'exit' to stop.")
    print("")

    chatbot = QwenChatbot()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break

        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}")
        print("---")


if __name__ == "__main__":
    run_chat_interface()
