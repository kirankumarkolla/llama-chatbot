from flask import Flask, render_template, request, stream_with_context, Response, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
import requests, json, os, re

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"
CHROMA_DIR = "chroma_db/"
HISTORY_FILE = "chat_history.json"


def detect_intent(user_input):
    """Basic rule-based intent detection (replace with LLM later if you want)"""
    personal_keywords = ["my", "me", "i", "passport", "aadhar", "birthday", "email"]
    doc_keywords = ["show", "find", "document", "resume", "csv", "certificate"]

    personal = any(word in user_input.lower() for word in personal_keywords)
    doc_related = any(word in user_input.lower() for word in doc_keywords)

    if personal and not doc_related:
        return "memory"
    elif doc_related:
        return "documents"
    return "general"

# === Chat Memory ===
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

messages = load_history()

# === Vector DBs ===
embeddings = OllamaEmbeddings(model="nomic-embed-text")
docs_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
memory_store = Chroma(persist_directory=CHROMA_DIR, collection_name="memory", embedding_function=embeddings)

# === Personal Fact Extractor ===
def extract_fact_from_message(message):
    patterns = [
        r"my ([\w\s]+?) is ([\w\s\d\-\.@]+)",
        r"i use ([\w\s\d\-\.@]+)",
        r"my name is ([\w\s]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None

# === Document Context Search ===
def get_context_from_docs(query):
    results = docs_store.similarity_search(query, k=4)
    context_parts = [
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content.strip()}"
        for doc in results
    ]
    return "\n\n".join(context_parts)

# === Dynamic Context Builder ===
def build_dynamic_prompt(user_input, recent_chat):
    intent = detect_intent(user_input)

    memory_context = ""
    doc_context = ""

    if intent in ["memory", "general"]:
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
        memory_results = db.similarity_search(user_input, k=4)
        memory_context = "\n\n".join([
            f"[memory] {doc.page_content.strip()}" for doc in memory_results
        ])

    if intent in ["documents", "general"]:
        doc_context = get_context_from_docs(user_input)

    chat_context = "\n".join([
        f"{m['role'].capitalize()}: {m['content']}" for m in recent_chat
    ])

    prompt = f"""You are a private, secure personal assistant. You help answer questions using the user's own documents and remembered personal information.

                The user has granted permission to reference personal documents (e.g., Aadhaar, Passport) and personal facts for their use only.

            Conversation so far:
            {chat_context}

            Relevant personal memory:
            {memory_context}

            Relevant documents:
            {doc_context}

            Answer the latest user message:
            Assistant:"""

    return prompt


# === Memory Context Search ===
def get_context_from_memory(query):
    results = memory_store.similarity_search(query, k=4)
    return "\n".join([doc.page_content.strip() for doc in results])

# === Routes ===
@app.route("/")
def index():
    return render_template("chat.html", messages=messages)

@app.route("/submit", methods=["POST"])
def submit_message():
    user_input = request.form["message"]
    messages.append({"role": "user", "content": user_input})
    save_history(messages)

    # 💡 Detect and store personal info securely
    fact = extract_fact_from_message(user_input)
    if fact:
        memory_store.add_texts([fact])
        print(f"🔐 Stored fact in memory DB: {fact}")

    return jsonify({"ok": True})

@app.route("/stream")
def stream_response():
    def generate():
        user_input = messages[-1]["content"]
        recent_chat = messages[-4:] if len(messages) >= 4 else messages

        # Build context dynamically based on intent
        prompt = build_dynamic_prompt(user_input, recent_chat)
#         chat_context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent_chat])

#         doc_context = get_context_from_docs(user_input)
#         memory_context = get_context_from_memory(user_input)

#         prompt = f"""You are a private, secure personal assistant. You help answer questions using the user's own documents and remembered personal information.

# The user has granted permission to reference personal documents (e.g., Aadhaar, Passport) and personal facts for their use only.

# Recent conversation:
# {chat_context}

# Known personal facts:
# {memory_context}

# Relevant document info:
# {doc_context}

# Latest user message:
# {user_input}

# Assistant:"""

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": True
        }

        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        assistant_reply = ""

        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                chunk = data.get("response", "")
                assistant_reply += chunk
                yield f"data: {chunk}\n\n"

        messages.append({"role": "assistant", "content": assistant_reply})
        save_history(messages)

    return Response(stream_with_context(generate()), content_type="text/event-stream")

@app.route("/new", methods=["POST"])
def new_chat():
    global messages
    messages = []
    save_history(messages)
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=True)
