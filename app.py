from flask import Flask, render_template, request, stream_with_context, Response, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
import requests, json, os, re

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"
CHROMA_DIR = "chroma_db/"
HISTORY_FILE = "chat_history.json"

# === Chat History ===
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

# === Intent Detection via LLM ===
def classify_intent_llm(user_input):
    classification_prompt = f"""
Classify the following user query into one of these intents:
- memory: asking about personal info (name, birthday, address, favorites, etc.)
- documents: asking about content from resume, csvs, IDs, etc.
- general: general knowledge, chitchat, or open-ended questions

User message: "{user_input}"

Reply with just one word: memory, documents, or general.
"""
    payload = {
        "model": MODEL_NAME,
        "prompt": classification_prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        result = response.json()
        reply = result.get("response", "").strip().lower()
        if reply not in ["memory", "documents", "general"]:
            return "general"
        return reply
    except:
        return "general"

# === Personal Fact Extractor ===
def extract_fact_from_message(message):
    patterns = [
        r"my ([\w\s]+?) is ([\w\s\d\-\.@]+)",
        r"i am ([\w\s]+)",
        r"this is ([\w\s]+)",
        r"my name is ([\w\s]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None

# === Vector Context Functions ===
def get_context_from_docs(query):
    results = docs_store.similarity_search(query, k=4)
    return "\n\n".join([
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content.strip()}"
        for doc in results
    ])

def get_context_from_memory(query):
    results = memory_store.similarity_search(query, k=4)
    return "\n".join([
        doc.page_content.strip() for doc in results
    ])

# === Prompt Builder ===

def build_dynamic_prompt(user_input, recent_chat):
    # Search in both memory and documents
    memory_results = memory_store.similarity_search_with_score(user_input, k=4)
    doc_results = docs_store.similarity_search_with_score(user_input, k=4)

    # Sort by descending relevance (score)
    memory_results = sorted(memory_results, key=lambda x: -x[1])
    doc_results = sorted(doc_results, key=lambda x: -x[1])

    # Format memory context
    memory_context = "\n".join([
        f"[memory] {doc.page_content.strip()}" for doc, score in memory_results
    ]) if memory_results else "(No relevant personal info found)"

    # Format document context
    doc_context = "\n".join([
        f"[doc] {doc.page_content.strip()}" for doc, score in doc_results
    ]) if doc_results else "(No relevant documents found)"

    # Format recent chat
    chat_context = "\n".join([
        f"{m['role'].capitalize()}: {m['content']}" for m in recent_chat
    ])

    # Build final prompt
    prompt = f"""
You are a private, secure personal assistant. You are interacting directly with the user, who has granted full access to their own personal data, documents, and identity-related information — strictly for their own use.

The assistant is authorized to recall, reference, and reason over personal details such as name, Aadhaar, date of birth, family members, and work history, as stored in the local memory or documents — only for this user's benefit.

Conversation so far:
{chat_context}

Personal info from memory:
{memory_context}

Info from user's documents:
{doc_context}

Respond to the latest user message based on the most relevant source.
Assistant:"""

    return prompt


def build_dynamic_prompt_old(user_input, recent_chat):
    intent = classify_intent_llm(user_input)
    print(f"\n🧠 Detected intent: {intent}")

    memory_context = get_context_from_memory(user_input) if intent in ["memory", "general"] else ""
    doc_context = get_context_from_docs(user_input) if intent in ["documents", "general"] else ""

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

Only include personal information that is directly relevant to the user's latest question. Do not list unrelated facts or previously shared data unless explicitly asked.
Answer the latest user message:
Assistant:"""

    return prompt

# === Routes ===
@app.route("/")
def index():
    return render_template("chat.html", messages=messages)

@app.route("/submit", methods=["POST"])
def submit_message():
    user_input = request.form["message"]
    messages.append({"role": "user", "content": user_input})
    save_history(messages)

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

        prompt = build_dynamic_prompt(user_input, recent_chat)

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
