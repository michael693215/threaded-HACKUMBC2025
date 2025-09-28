from flask import Flask, render_template, request, jsonify
import os
import json
import base64
import io
from PIL import Image, UnidentifiedImageError

# NOTE:
# - This version removes ALL uses of `signal` inside the request handler.
# - Background removal is executed in a separate PROCESS with a 9.5s timeout,
#   which avoids the “signal only works in main thread” error you saw and keeps
#   total request time under ~10s.
# - Images are downscaled to a sane max size before processing and auto-cropped.

app = Flask(__name__)

# -------------------- Pages --------------------
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


# -------------------- Shop AI (optional) --------------------
@app.route("/api/shop", methods=["POST"])
def api_shop():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "missing_api_key"}), 400

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)  # noqa: F841
    except Exception as e:
        return jsonify({"error": f"openai_import_error: {e}"}), 500

    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
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
        '  "queries": string[],\n'
        '  "response": string\n'
        "}\n"
        "If any field is unknown, use null. Keep queries short (2-6 words each). "
        "Do NOT include any text outside of the JSON."
    )

    chat_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
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

    try:
        payload = json.loads(content)
        if "queries" not in payload or "response" not in payload:
            raise ValueError("missing keys")
    except Exception as e:
        return jsonify({"error": f"bad_model_output: {e}"}), 502

    return jsonify(payload), 200


# -------------------- Background Removal API --------------------
# Implementation notes:
# * We process in a separate **process** using ProcessPoolExecutor to avoid the
#   “signal only works in main thread” crash when Flask runs handlers in threads.
# * We cap the longest dimension to 1024px for speed and consistent thumbnails.
# * We trim transparent borders so the item appears centered/contained on the site.

from concurrent.futures import ProcessPoolExecutor, TimeoutError  # noqa: E402


def _remove_bg_worker(image_bytes: bytes, model_name: str = "isnet-general-use") -> bytes:
    """
    Runs inside a separate process. Returns PNG bytes with transparency.
    """
    from rembg import remove, new_session  # imported inside the worker process
    from PIL import Image
    import io as _io

    # Load image
    img = Image.open(_io.BytesIO(image_bytes)).convert("RGBA")

    # Downscale for speed
    max_side = 1024
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Try primary model, then fallback
    try:
        session = new_session(model_name)
        out = remove(img, session=session)
    except Exception:
        # Fallback to 'u2net'
        session = new_session("u2net")
        out = remove(img, session=session)

    # Convert to PIL Image (remove() may return bytes or Image)
    if isinstance(out, Image.Image):
        result = out.convert("RGBA")
    else:
        result = Image.open(_io.BytesIO(out)).convert("RGBA")

    # Trim transparent borders
    alpha = result.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        result = result.crop(bbox)

    # Encode PNG
    buf = _io.BytesIO()
    result.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


@app.route("/api/remove-bg", methods=["POST"])
def api_remove_bg():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no_file"}), 400

    try:
        raw = f.read()
        # quick load validation before handing off to worker
        Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        return jsonify({"error": "bad_image"}), 400
    except Exception as e:
        return jsonify({"error": f"read_error: {e}"}), 400

    # Run the worker in a separate process with a strict timeout
    try:
        # Using a context manager here avoids issues with Flask's reloader.
        with ProcessPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_remove_bg_worker, raw, "isnet-general-use")
            out_png = fut.result(timeout=9.5)  # keep total under ~10s as requested
    except TimeoutError:
        return jsonify({"error": "timeout"}), 504
    except Exception as e:
        return jsonify({"error": f"processing_failed: {e}"}), 500

    data_url = "data:image/png;base64," + base64.b64encode(out_png).decode("ascii")
    return jsonify({"data_url": data_url}), 200


# -------------------- Main --------------------
if __name__ == "__main__":
    # If you still see double-start issues on macOS with the dev reloader,
    # you can set `use_reloader=False` here.
    app.run(debug=True)
