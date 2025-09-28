from flask import Flask, render_template, request, jsonify
import os
import json

app = Flask(__name__)

# ---- Pages ----
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/folder")
def folder_page():
    return render_template("folder.html")

@app.route("/outfits/create")
def outfits_create():
    return render_template("create.html")

@app.route("/outfits/saved")
def outfits_saved():
    return render_template("saved.html")

@app.route("/shop")
def shop_ai():
    return render_template("shop.html")


# ---- Simple Shopping AI API (OpenAI-backed) ----
# Set OPENAI_API_KEY in your environment:  export OPENAI_API_KEY=sk-...
# pip install openai
@app.route("/api/shop", methods=["POST"])
def api_shop():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "missing_api_key"}), 400

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return jsonify({"error": f"openai_import_error: {e}"}), 500

    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    # Guard: messages should be a list of {"role": "...", "content": "..."}
    if not isinstance(messages, list):
        return jsonify({"error": "bad_request"}), 400

    system_prompt = (
        "You are a shopping stylist assistant. Read the conversation and extract a clean shopping intent. "
        "Return STRICT JSON ONLY with keys:\n"
        "{\n"
        '  "item": string | null,\n'
        '  "color": string | null,\n'
        '  "style": string | null,\n'
        '  "budget": number | null,\n'
        '  "queries": string[]  // 6-10 concise search queries to find relevant products,\n'
        '  "response": string   // a friendly one-paragraph reply for the user\n'
        "}\n"
        "If any field is unknown, use null. Keep queries short (2-6 words each) and specific.\n"
        "Examples of queries: 'black leather jacket men', 'streetwear cargo pants', 'minimalist white sneakers women'.\n"
        "Do NOT include any text outside of the JSON."
    )

    # Build request to the model
    chat_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        # Model choice: widely available & inexpensive
        # You can change to 'gpt-4o' if desired.
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_messages,
            temperature=0.2,
        )
        content = completion.choices[0].message.content
    except Exception as e:
        return jsonify({"error": f"openai_api_error: {e}"}), 500

    # Parse strict JSON from the assistant
    try:
        payload = json.loads(content)
        # Basic shape validation
        if "queries" not in payload or "response" not in payload:
            raise ValueError("missing keys")
    except Exception as e:
        # If model ever responded with non-JSON, fail gracefully
        return jsonify({"error": f"bad_model_output: {e}"}), 502

    return jsonify(payload), 200


if __name__ == "__main__":
    app.run(debug=True)
