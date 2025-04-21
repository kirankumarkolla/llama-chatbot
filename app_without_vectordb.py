from flask import Flask, render_template, request, stream_with_context, Response, jsonify
import requests
import json
import os

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"
HISTORY_FILE = "chat_history.json"

# === Persistent History ===
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

messages = load_history()

# === POST message ===
@app.route("/submit", methods=["POST"])
def submit_message():
    user_input = request.form["message"]
    messages.append({"role": "user", "content": user_input})
    save_history(messages)
    return jsonify({"ok": True})

# === GET stream response ===
@app.route("/stream")
def stream_response():
    def generate():
        prompt = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages if m["content"])
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": True
        }

        with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        chunk = data.get("response", "")
                        yield f"data: {chunk}\n\n"
                    except json.JSONDecodeError:
                        continue

        # Finish with a done signal
        yield "event: done\ndata: [DONE]\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route("/new", methods=["POST"])
def new_chat():
    global messages
    messages = []
    save_history(messages)
    return jsonify({"ok": True})


# === UI ===
@app.route("/")
def index():
    return render_template("chat.html", messages=messages)

if __name__ == "__main__":
    app.run(debug=True)
