"""
Emotion Detection Dashboard
A Flask web dashboard for the emotion detection model.
Upload images, use webcam, and view model/dataset statistics.
"""

import os
import io
import json
import base64
import numpy as np
import cv2
from flask import Flask, render_template_string, request, jsonify, Response
from collections import Counter
import threading
import time

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
app = Flask(__name__)

MODEL_JSON_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.json")
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.weights.h5")

emotion_dict = {
    0: "Angry", 1: "Disgust", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised",
    7: "Class_7", 8: "Class_8", 9: "Class_9", 10: "Class_10",
    11: "Class_11", 12: "Class_12", 13: "Class_13", 14: "Class_14",
    15: "Class_15", 16: "Class_16", 17: "Class_17", 18: "Class_18",
}

# Lazy-loaded model
_model = None
_model_lock = threading.Lock()

# Webcam state
_camera = None
_camera_lock = threading.Lock()
_streaming = False

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def get_model():
    """Lazy-load the Keras model so startup stays fast."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if not os.path.exists(MODEL_JSON_PATH) or not os.path.exists(MODEL_WEIGHTS_PATH):
            return None
        try:
            from keras.models import model_from_json
            with open(MODEL_JSON_PATH, "r") as f:
                _model = model_from_json(f.read())
            _model.load_weights(MODEL_WEIGHTS_PATH)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            _model = None
    return _model


def predict_emotions(frame):
    """
    Detect faces and predict emotions for each face in the given frame.
    Returns the annotated frame and a list of {emotion, confidence, bbox} dicts.
    """
    model = get_model()
    results = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (48, 48))
        roi_input = np.expand_dims(np.expand_dims(roi_resized, -1), 0) / 255.0

        if model is not None:
            preds = model.predict(roi_input, verbose=0)[0]
            idx = int(np.argmax(preds))
            emotion = emotion_dict.get(idx, f"Class_{idx}")
            confidence = float(preds[idx]) * 100
        else:
            emotion = "No model"
            confidence = 0.0

        results.append({"emotion": emotion, "confidence": round(confidence, 1),
                         "bbox": [int(x), int(y), int(w), int(h)]})

        # Draw on frame
        color = _emotion_color(emotion)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{emotion} {confidence:.0f}%"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame, results


def _emotion_color(emotion):
    """Return a BGR color for each emotion for nice overlays."""
    palette = {
        "Angry": (0, 0, 255), "Disgust": (0, 140, 255), "Fearful": (255, 0, 255),
        "Happy": (0, 255, 0), "Neutral": (255, 255, 0), "Sad": (255, 165, 0),
        "Surprised": (0, 255, 255),
    }
    return palette.get(emotion, (200, 200, 200))


def get_model_summary_info():
    """Return a dict with model architecture details."""
    model = get_model()
    if model is None:
        return None
    layers = []
    for layer in model.layers:
        cfg = layer.get_config()
        layers.append({
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": str(layer.output_shape) if hasattr(layer, 'output_shape') else "?",
            "params": int(layer.count_params()),
        })
    return {
        "total_params": int(model.count_params()),
        "layers": layers,
        "num_layers": len(model.layers),
    }


def get_dataset_stats(train_dir="train", test_dir="test"):
    """Return class distribution from local train/test folders."""
    stats = {}
    for label, d in [("train", train_dir), ("test", test_dir)]:
        counts = {}
        if os.path.isdir(d):
            for cls_name in sorted(os.listdir(d)):
                cls_path = os.path.join(d, cls_name)
                if os.path.isdir(cls_path):
                    counts[cls_name] = len([
                        f for f in os.listdir(cls_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ])
        stats[label] = counts
    return stats


# ---------------------------------------------------------------------------
# Webcam streaming helpers
# ---------------------------------------------------------------------------
def _gen_frames():
    """Generator that yields JPEG frames with emotion overlay."""
    global _camera, _streaming
    with _camera_lock:
        if _camera is None or not _camera.isOpened():
            _camera = cv2.VideoCapture(0)
        _streaming = True

    while _streaming:
        success, frame = _camera.read()
        if not success:
            break
        frame, _ = predict_emotions(frame)
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    with _camera_lock:
        if _camera is not None:
            _camera.release()
            _camera = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Accept an uploaded image and return emotion predictions."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    annotated, results = predict_emotions(frame)

    # Encode the annotated image back to base64
    _, buf = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buf).decode("utf-8")

    return jsonify({"image": b64, "faces": results})


@app.route("/api/model-info")
def api_model_info():
    info = get_model_summary_info()
    if info is None:
        return jsonify({"status": "no_model", "message": "Model files not found. Train the model first using code.py."})
    return jsonify({"status": "ok", **info})


@app.route("/api/dataset-stats")
def api_dataset_stats():
    stats = get_dataset_stats()
    return jsonify(stats)


@app.route("/api/emotion-map")
def api_emotion_map():
    return jsonify(emotion_dict)


@app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stop-camera", methods=["POST"])
def stop_camera():
    global _streaming
    _streaming = False
    return jsonify({"status": "stopped"})


# ---------------------------------------------------------------------------
# HTML Template  — dark glassmorphism, gradient accents, micro-animations
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Emotion Detection Dashboard</title>
<meta name="description" content="Real-time emotion detection dashboard powered by deep learning — upload images or use your webcam." />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
<style>
/* ── CSS Reset & Variables ─────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b0e17;--surface:rgba(255,255,255,.04);--glass:rgba(255,255,255,.06);
  --border:rgba(255,255,255,.08);--text:#e4e6f0;--text-dim:#8b8fa3;
  --accent:#7c5cfc;--accent2:#00d4ff;--accent3:#ff6ec7;
  --green:#34d399;--red:#f87171;--orange:#fbbf24;
  --radius:16px;--radius-sm:10px;
}
html{font-size:15px;scroll-behavior:smooth}
body{
  font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);
  min-height:100vh;overflow-x:hidden;
}
/* Gradient orbs behind content */
body::before,body::after{
  content:'';position:fixed;border-radius:50%;filter:blur(120px);opacity:.25;z-index:0;pointer-events:none;
}
body::before{width:600px;height:600px;top:-120px;left:-100px;background:radial-gradient(circle,var(--accent),transparent 70%)}
body::after{width:500px;height:500px;bottom:-80px;right:-80px;background:radial-gradient(circle,var(--accent2),transparent 70%)}

/* ── Layout ────────────────────────────────────────────── */
.app{position:relative;z-index:1;max-width:1280px;margin:0 auto;padding:28px 24px 60px}
header{text-align:center;margin-bottom:36px}
header h1{
  font-size:2.4rem;font-weight:800;
  background:linear-gradient(135deg,var(--accent),var(--accent2),var(--accent3));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  letter-spacing:-.5px;
}
header p{color:var(--text-dim);margin-top:6px;font-size:.95rem}

.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}

/* ── Card (glass) ──────────────────────────────────────── */
.card{
  background:var(--glass);border:1px solid var(--border);border-radius:var(--radius);
  padding:26px 28px;backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
  transition:transform .25s ease,box-shadow .25s ease;
}
.card:hover{transform:translateY(-3px);box-shadow:0 12px 40px rgba(124,92,252,.12)}
.card-title{
  font-size:1.05rem;font-weight:700;margin-bottom:14px;display:flex;align-items:center;gap:8px;
}
.card-title .dot{width:8px;height:8px;border-radius:50%;display:inline-block}

/* ── Upload Zone ───────────────────────────────────────── */
.upload-zone{
  border:2px dashed var(--border);border-radius:var(--radius-sm);padding:40px 20px;
  text-align:center;cursor:pointer;transition:.3s;position:relative;overflow:hidden;
}
.upload-zone:hover{border-color:var(--accent);background:rgba(124,92,252,.06)}
.upload-zone.dragover{border-color:var(--accent2);background:rgba(0,212,255,.08)}
.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer}
.upload-zone .icon{font-size:2.4rem;margin-bottom:8px}
.upload-zone p{color:var(--text-dim);font-size:.88rem}

/* ── Buttons ───────────────────────────────────────────── */
.btn{
  display:inline-flex;align-items:center;gap:6px;
  padding:10px 22px;border:none;border-radius:var(--radius-sm);font-weight:600;
  font-size:.88rem;cursor:pointer;transition:.25s;font-family:inherit;
}
.btn-primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff}
.btn-primary:hover{filter:brightness(1.12);transform:scale(1.03)}
.btn-danger{background:var(--red);color:#fff}
.btn-danger:hover{filter:brightness(1.12)}
.btn-outline{background:transparent;border:1px solid var(--border);color:var(--text)}
.btn-outline:hover{border-color:var(--accent);color:var(--accent)}

/* ── Results ───────────────────────────────────────────── */
#result-img{max-width:100%;border-radius:var(--radius-sm);margin-top:14px;display:none}
.face-results{margin-top:14px}
.face-chip{
  display:inline-flex;align-items:center;gap:8px;
  background:var(--surface);border:1px solid var(--border);border-radius:30px;
  padding:6px 16px;margin:4px;font-size:.85rem;
  animation:popIn .35s ease forwards;
}
@keyframes popIn{from{opacity:0;transform:scale(.85)}to{opacity:1;transform:scale(1)}}
.face-chip .conf{color:var(--accent2);font-weight:600}

/* ── Webcam ────────────────────────────────────────────── */
#webcam-container{position:relative;text-align:center}
#webcam-feed{max-width:100%;border-radius:var(--radius-sm);display:none}
#webcam-placeholder{
  padding:60px 20px;border:2px dashed var(--border);border-radius:var(--radius-sm);
  color:var(--text-dim);font-size:.9rem;
}

/* ── Model Info Table ──────────────────────────────────── */
.layer-table{width:100%;border-collapse:collapse;font-size:.82rem;margin-top:10px}
.layer-table th,.layer-table td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border)}
.layer-table th{color:var(--text-dim);font-weight:600;text-transform:uppercase;font-size:.72rem;letter-spacing:.5px}
.layer-table tr:hover td{background:rgba(124,92,252,.06)}
.param-badge{
  background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;
  padding:4px 12px;border-radius:20px;font-size:.82rem;font-weight:700;display:inline-block;
}

/* ── Chart (bar chart via CSS) ─────────────────────────── */
.bar-chart{margin-top:12px}
.bar-row{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.bar-label{width:90px;font-size:.78rem;color:var(--text-dim);text-align:right;flex-shrink:0}
.bar-track{flex:1;height:22px;background:var(--surface);border-radius:6px;overflow:hidden;position:relative}
.bar-fill{
  height:100%;border-radius:6px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  transition:width .6s ease;
}
.bar-val{font-size:.78rem;width:50px;font-weight:600;color:var(--text)}

/* ── Emotion map grid ──────────────────────────────────── */
.emo-grid{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.emo-tag{
  background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:6px 14px;font-size:.8rem;font-weight:500;
  transition:.2s;
}
.emo-tag:hover{border-color:var(--accent);color:var(--accent)}

/* ── Status indicator ──────────────────────────────────── */
.status-badge{
  display:inline-flex;align-items:center;gap:6px;font-size:.82rem;
  padding:5px 14px;border-radius:20px;font-weight:600;
}
.status-badge.ok{background:rgba(52,211,153,.12);color:var(--green)}
.status-badge.warn{background:rgba(248,113,113,.12);color:var(--red)}

/* ── Loading spinner ───────────────────────────────────── */
.spinner{display:none;text-align:center;padding:20px}
.spinner::after{
  content:'';display:inline-block;width:28px;height:28px;
  border:3px solid var(--border);border-top-color:var(--accent);border-radius:50%;
  animation:spin .7s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Footer ────────────────────────────────────────────── */
footer{text-align:center;margin-top:48px;color:var(--text-dim);font-size:.78rem}
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>🧠 Emotion Detection Dashboard</h1>
    <p>Deep learning-powered facial emotion recognition &mdash; upload an image or use your webcam</p>
    <div style="margin-top:12px" id="model-status"></div>
  </header>

  <div class="grid">
    <!-- ────────── Upload Card ────────── -->
    <div class="card" id="upload-card">
      <div class="card-title"><span class="dot" style="background:var(--accent)"></span> Image Upload</div>
      <div class="upload-zone" id="drop-zone">
        <input type="file" accept="image/*" id="file-input" />
        <div class="icon">📁</div>
        <p>Drag &amp; drop an image here, or <strong>click to browse</strong></p>
      </div>
      <div class="spinner" id="upload-spinner"></div>
      <img id="result-img" alt="Prediction result" />
      <div class="face-results" id="face-results"></div>
    </div>

    <!-- ────────── Webcam Card ────────── -->
    <div class="card">
      <div class="card-title"><span class="dot" style="background:var(--accent2)"></span> Live Webcam</div>
      <div id="webcam-container">
        <div id="webcam-placeholder">📷 Click <strong>Start Camera</strong> to begin live emotion detection</div>
        <img id="webcam-feed" alt="Webcam feed" />
      </div>
      <div style="margin-top:14px;display:flex;gap:10px">
        <button class="btn btn-primary" id="btn-start-cam">▶ Start Camera</button>
        <button class="btn btn-danger" id="btn-stop-cam" disabled>⏹ Stop</button>
      </div>
    </div>

    <!-- ────────── Model Info Card ────────── -->
    <div class="card">
      <div class="card-title"><span class="dot" style="background:var(--accent3)"></span> Model Architecture</div>
      <div id="model-info">Loading…</div>
    </div>

    <!-- ────────── Dataset Stats Card ────────── -->
    <div class="card">
      <div class="card-title"><span class="dot" style="background:var(--green)"></span> Dataset Distribution</div>
      <div id="dataset-stats">Loading…</div>
    </div>

    <!-- ────────── Emotion Map Card (full width) ────────── -->
    <div class="card" style="grid-column:1/-1">
      <div class="card-title"><span class="dot" style="background:var(--orange)"></span> Emotion Class Map</div>
      <div id="emotion-map">Loading…</div>
    </div>
  </div>

  <footer>Emotion Detection Dashboard &bull; Built with Flask, Keras &amp; OpenCV</footer>
</div>

<script>
/* ── helpers ─────────────────────────────────── */
const $ = s => document.querySelector(s);

/* ── Image Upload ────────────────────────────── */
const dropZone = $('#drop-zone');
const fileInput = $('#file-input');

['dragover','dragenter'].forEach(e=>dropZone.addEventListener(e,ev=>{ev.preventDefault();dropZone.classList.add('dragover')}));
['dragleave','drop'].forEach(e=>dropZone.addEventListener(e,ev=>{ev.preventDefault();dropZone.classList.remove('dragover')}));
dropZone.addEventListener('drop',e=>handleFile(e.dataTransfer.files[0]));
fileInput.addEventListener('change',()=>handleFile(fileInput.files[0]));

function handleFile(file){
  if(!file) return;
  const form = new FormData();
  form.append('file', file);
  $('#upload-spinner').style.display='block';
  $('#face-results').innerHTML='';
  $('#result-img').style.display='none';
  fetch('/api/predict',{method:'POST',body:form})
    .then(r=>r.json()).then(d=>{
      $('#upload-spinner').style.display='none';
      if(d.error){$('#face-results').innerHTML=`<p style="color:var(--red)">${d.error}</p>`;return}
      $('#result-img').src='data:image/jpeg;base64,'+d.image;
      $('#result-img').style.display='block';
      let html='';
      d.faces.forEach(f=>{
        html+=`<span class="face-chip">${f.emotion} <span class="conf">${f.confidence}%</span></span>`;
      });
      if(d.faces.length===0) html='<p style="color:var(--text-dim)">No faces detected.</p>';
      $('#face-results').innerHTML=html;
    }).catch(()=>{$('#upload-spinner').style.display='none';$('#face-results').innerHTML='<p style="color:var(--red)">Request failed.</p>'});
}

/* ── Webcam ──────────────────────────────────── */
$('#btn-start-cam').addEventListener('click',()=>{
  $('#webcam-feed').src='/video_feed';
  $('#webcam-feed').style.display='block';
  $('#webcam-placeholder').style.display='none';
  $('#btn-start-cam').disabled=true;
  $('#btn-stop-cam').disabled=false;
});
$('#btn-stop-cam').addEventListener('click',()=>{
  fetch('/api/stop-camera',{method:'POST'});
  $('#webcam-feed').style.display='none';
  $('#webcam-placeholder').style.display='block';
  $('#btn-start-cam').disabled=false;
  $('#btn-stop-cam').disabled=true;
});

/* ── Load panels ─────────────────────────────── */
fetch('/api/model-info').then(r=>r.json()).then(d=>{
  if(d.status==='no_model'){
    $('#model-status').innerHTML=`<span class="status-badge warn">⚠ ${d.message}</span>`;
    $('#model-info').innerHTML=`<p style="color:var(--text-dim)">${d.message}</p>`;
    return;
  }
  $('#model-status').innerHTML='<span class="status-badge ok">✅ Model loaded</span>';
  let html=`<p style="margin-bottom:8px">Total parameters: <span class="param-badge">${d.total_params.toLocaleString()}</span></p>`;
  html+='<table class="layer-table"><thead><tr><th>#</th><th>Layer</th><th>Type</th><th>Output Shape</th><th>Params</th></tr></thead><tbody>';
  d.layers.forEach((l,i)=>{
    html+=`<tr><td>${i+1}</td><td>${l.name}</td><td>${l.type}</td><td>${l.output_shape}</td><td>${l.params.toLocaleString()}</td></tr>`;
  });
  html+='</tbody></table>';
  $('#model-info').innerHTML=html;
});

fetch('/api/dataset-stats').then(r=>r.json()).then(d=>{
  let html='';
  ['train','test'].forEach(split=>{
    const counts=d[split];
    const keys=Object.keys(counts);
    if(keys.length===0){html+=`<p style="color:var(--text-dim)">No ${split} data found locally.</p>`;return;}
    const max=Math.max(...Object.values(counts));
    html+=`<h4 style="margin:12px 0 6px;font-size:.88rem;color:var(--text-dim);text-transform:uppercase">${split} Set</h4><div class="bar-chart">`;
    keys.forEach(k=>{
      const pct=Math.round(counts[k]/max*100);
      html+=`<div class="bar-row"><span class="bar-label">${k}</span><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div><span class="bar-val">${counts[k]}</span></div>`;
    });
    html+='</div>';
  });
  if(!html) html='<p style="color:var(--text-dim)">Dataset folders not found locally. Run code.py first.</p>';
  $('#dataset-stats').innerHTML=html;
});

fetch('/api/emotion-map').then(r=>r.json()).then(d=>{
  let html='<div class="emo-grid">';
  Object.entries(d).forEach(([k,v])=>{html+=`<span class="emo-tag"><strong>${k}</strong> → ${v}</span>`});
  html+='</div>';
  $('#emotion-map').innerHTML=html;
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Starting Emotion Detection Dashboard…")
    print("   Open http://localhost:5000 in your browser\n")
    # debug=False because "code.py" shadows Python's built-in 'code' module
    # which Flask's debugger tries to import, causing a crash
    app.run(debug=False, host="0.0.0.0", port=5000)
