#!/usr/bin/env python3
"""Simple local web interface for the TakeMeter classifier.

Run after fine-tuning a model and saving it locally:

    python3 scripts/takemeter_interface.py

By default this looks for one of:
- $TAKEMETER_MODEL_DIR
- takemeter_distilbert_model/
- takemeter-model/
- takemeter-model/checkpoint-*
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_FALLBACK = {0: "analysis", 1: "hot_take", 2: "reaction"}


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TakeMeter</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
      background: #f5f7fa;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      box-sizing: border-box;
    }
    main {
      width: min(880px, 100%);
      background: #ffffff;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      box-shadow: 0 18px 50px rgba(31, 41, 51, 0.08);
      padding: 24px;
      box-sizing: border-box;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 0;
    }
    p {
      margin: 0 0 18px;
      color: #52606d;
      line-height: 1.45;
    }
    textarea {
      width: 100%;
      min-height: 160px;
      resize: vertical;
      box-sizing: border-box;
      border: 1px solid #b8c4d3;
      border-radius: 6px;
      padding: 12px;
      font: inherit;
      line-height: 1.45;
    }
    .row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
      flex-wrap: wrap;
    }
    button {
      border: 1px solid #2f6f9f;
      background: #2f6f9f;
      color: white;
      border-radius: 6px;
      padding: 10px 14px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button.secondary {
      background: #ffffff;
      color: #2f6f9f;
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    #result {
      margin-top: 18px;
      border-top: 1px solid #e4e7eb;
      padding-top: 18px;
      display: none;
    }
    .label {
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      padding: 0 12px;
      border-radius: 6px;
      background: #e6f6ff;
      color: #0b4f71;
      font-weight: 750;
    }
    .confidence {
      margin-left: 10px;
      color: #52606d;
      font-weight: 650;
    }
    .bars {
      margin-top: 14px;
      display: grid;
      gap: 8px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 88px 1fr 56px;
      gap: 10px;
      align-items: center;
      font-size: 14px;
    }
    .track {
      height: 10px;
      background: #edf1f7;
      border-radius: 999px;
      overflow: hidden;
    }
    .fill {
      height: 100%;
      width: 0;
      background: #2f6f9f;
    }
    .error {
      color: #b42318;
      font-weight: 650;
    }
  </style>
</head>
<body>
  <main>
    <h1>TakeMeter</h1>
    <p>Classify an r/nba comment as analysis, hot_take, or reaction.</p>
    <textarea id="text">I love being a fraud and fake number one seed and can't wait to get swept by the wolves with my star player having 50 FTA</textarea>
    <div class="row">
      <button id="classify">Classify</button>
      <button class="secondary" id="analysis">Sample analysis</button>
      <button class="secondary" id="hot">Sample hot_take</button>
      <button class="secondary" id="reaction">Sample reaction</button>
    </div>
    <section id="result">
      <div>
        <span class="label" id="label"></span>
        <span class="confidence" id="confidence"></span>
      </div>
      <div class="bars" id="bars"></div>
    </section>
  </main>
  <script>
    const samples = {
      analysis: "They had no real 2nd level of their offense. The progression tree was basically spacing into a quick three, then a late drive if the first action failed.",
      hot: "This team is fake and will get exposed the second they play a real contender.",
      reaction: "What is this 5 guard lineup lmao?"
    };

    const text = document.getElementById("text");
    const classify = document.getElementById("classify");
    const result = document.getElementById("result");
    const label = document.getElementById("label");
    const confidence = document.getElementById("confidence");
    const bars = document.getElementById("bars");

    document.getElementById("analysis").onclick = () => text.value = samples.analysis;
    document.getElementById("hot").onclick = () => text.value = samples.hot;
    document.getElementById("reaction").onclick = () => text.value = samples.reaction;

    async function run() {
      classify.disabled = true;
      result.style.display = "block";
      label.textContent = "Running";
      confidence.textContent = "";
      bars.innerHTML = "";
      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({text: text.value})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Request failed");
        label.textContent = data.label;
        confidence.textContent = `${Math.round(data.confidence * 100)}% confidence`;
        bars.innerHTML = data.probabilities.map(item => `
          <div class="bar-row">
            <div>${item.label}</div>
            <div class="track"><div class="fill" style="width:${Math.round(item.probability * 100)}%"></div></div>
            <div>${Math.round(item.probability * 100)}%</div>
          </div>
        `).join("");
      } catch (err) {
        label.textContent = "Error";
        confidence.innerHTML = `<span class="error">${err.message}</span>`;
      } finally {
        classify.disabled = false;
      }
    }

    classify.onclick = run;
  </script>
</body>
</html>
"""


def find_model_dir(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([Path("takemeter_distilbert_model"), Path("takemeter-model")])
    candidates.extend(sorted(Path("takemeter-model").glob("checkpoint-*"), reverse=True))

    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate
    raise FileNotFoundError(
        "No fine-tuned model directory found. Set TAKEMETER_MODEL_DIR or run fine-tuning first."
    )


class Predictor:
    def __init__(self, model_dir: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.id_to_label = {
            int(key): value for key, value in getattr(self.model.config, "id2label", LABEL_FALLBACK).items()
        }

    def predict(self, text: str) -> dict:
        if not text.strip():
            raise ValueError("Enter a comment before classifying.")
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).tolist()

        pred_id = max(range(len(probs)), key=lambda index: probs[index])
        probabilities = [
            {"label": self.id_to_label.get(index, str(index)), "probability": float(prob)}
            for index, prob in enumerate(probs)
        ]
        probabilities.sort(key=lambda item: item["probability"], reverse=True)
        return {
            "label": self.id_to_label.get(pred_id, str(pred_id)),
            "confidence": float(probs[pred_id]),
            "probabilities": probabilities,
        }


def make_handler(predictor: Predictor):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            self._send(200, HTML, content_type="text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = predictor.predict(str(payload.get("text", "")))
                self._send(200, json.dumps(result), content_type="application/json")
            except Exception as exc:
                self._send(400, json.dumps({"error": str(exc)}), content_type="application/json")

        def log_message(self, format, *args):
            return

        def _send(self, status: int, body: str, content_type: str):
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    model_dir = find_model_dir(args.model_dir)
    predictor = Predictor(model_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(predictor))
    print(f"Loaded model from {model_dir}")
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
