import requests
import json
from rich.console import Console
import os

# === Config ===
API_URL = "http://localhost:11434/api/generate"  # Use the generate endpoint for base models
MODEL_NAME = "llama3.2"  # Make sure you're using the correct model name
HISTORY_FILE = "chat_history.json"

console = Console()

# === Load/Save History ===
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# === Chat Core ===
messages = load_history()

def send_message(user_input):
    prompt = " ".join([msg["content"] for msg in messages]) + "\n" + user_input  # Combine previous history into one prompt
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True
    }

    response = requests.post(API_URL, json=payload, stream=True)
    #print(response)

    assistant_reply = ""
    console.print("[bold green]Assistant:[/bold green]", end=" ")

    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            content = data.get("response", "")  # ✅ For /api/generate with llama3.2
            assistant_reply += content
            console.print(content, end="", highlight=False)

    print()  # New line after assistant finishes
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": assistant_reply})
    save_history(messages)

# === Chat Loop ===
def chat_loop():
    console.print("[bold cyan]Welcome back to your local Ollama Chat![/bold cyan]")
    if messages:
        console.print(f"[bold yellow]Resuming conversation with {len(messages)//2} previous messages.[/bold yellow]")

    while True:
        try:
            user_input = input("\n[You]: ")
            if user_input.lower() in {"exit", "quit"}:
                console.print("[bold yellow]Session saved. Goodbye![/bold yellow]")
                break
            send_message(user_input)
        except KeyboardInterrupt:
            print("\n[!] Interrupted. Saving and exiting.")
            save_history(messages)
            break

if __name__ == "__main__":
    chat_loop()
