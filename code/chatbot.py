# chatbot.py
import ollama

# Load initial system context
with open("prompts/chatbot_context.md", "r", encoding="utf-8") as f:
    SYSTEM_CONTEXT = f.read()

# Initialize chat memory
chat_history = [{"role": "system", "content": SYSTEM_CONTEXT}]

def query_ollama(user_message):
    chat_history.append({"role": "user", "content": user_message})

    response = ollama.chat(
        model="gemma3",   # or whichever local model you prefer
        messages=chat_history,
        options={"temperature": 0.1}
    )

    assistant_message = response['message']['content']
    chat_history.append({"role": "assistant", "content": assistant_message})

    return assistant_message