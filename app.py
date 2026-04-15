import os
import io
import time
import logging
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH  = "bloodGroupDetection(93%_accuracy).pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = 128
CLASS_NAMES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

BLOOD_INFO = {
    "A+":  {"abo":"A",  "rh":"Positive", "donate_to":["A+","AB+"], "receive_from":["A+","A-","O+","O-"], "rarity":"35.7%"},
    "A-":  {"abo":"A",  "rh":"Negative", "donate_to":["A+","A-","AB+","AB-"], "receive_from":["A-","O-"],           "rarity":"6.3%"},
    "AB+": {"abo":"AB", "rh":"Positive", "donate_to":["AB+"], "receive_from":["All types"],        "rarity":"3.4%"},
    "AB-": {"abo":"AB", "rh":"Negative", "donate_to":["AB+","AB-"],          "receive_from":["All Rh- types"],    "rarity":"0.6%"},
    "B+":  {"abo":"B",  "rh":"Positive", "donate_to":["B+","AB+"],           "receive_from":["B+","B-","O+","O-"],"rarity":"8.5%"},
    "B-":  {"abo":"B",  "rh":"Negative", "donate_to":["B+","B-","AB+","AB-"],"receive_from":["B-","O-"],          "rarity":"1.5%"},
    "O+":  {"abo":"O",  "rh":"Positive", "donate_to":["A+","B+","AB+","O+"], "receive_from":["O+","O-"],          "rarity":"37.4%"},
    "O-":  {"abo":"O",  "rh":"Negative", "donate_to":["All types"],          "receive_from":["O-"],               "rarity":"6.6%"},
}

# ── ResUnit: exactly Conv(index 0) + BN(index 1) ───
# Keys in .pth:  res1.0.0.weight  res1.0.1.weight  (0=Conv, 1=BN)


class ResUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self._0 = nn.Conv2d(channels, channels, 3, padding=1)
        self._1 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return torch.relu(self._1(self._0(x)) + x)

# PyTorch uses attribute names as state_dict keys.
# We need attributes literally named "0" and "1" to match res1.0.0 / res1.0.1.
# Trick: override __setattr__ / __getattr__ is messy, so use a tiny Sequential instead.
# BUT Sequential adds ReLU at wrong index. Cleanest fix: subclass Sequential directly.


class ResUnit(nn.Module):
    """
    Stored in .pth as:
        res*.N.0  = Conv2d
        res*.N.1  = BatchNorm2d
    PyTorch Module attribute names become the key prefix.
    We register them under names '0' and '1' using add_module().
    """
    def __init__(self, channels):
        super().__init__()
        self.add_module("0", nn.Conv2d(channels, channels, 3, padding=1))
        self.add_module("1", nn.BatchNorm2d(channels))

    def forward(self, x):
        conv = self._modules["0"]
        bn = self._modules["1"]
        return torch.relu(bn(conv(x)) + x)


class BloodGroupCNN(nn.Module):
    """
    Architecture exactly matching the 58-key state_dict:

      conv1.0  Conv2d(3,64,3)      conv1.1  BN(64)
      conv2.0  Conv2d(64,128,3)    conv2.1  BN(128)
      res1     ModuleList of 2 ResUnit(128)
        res1.0.0  Conv  res1.0.1  BN
        res1.1.0  Conv  res1.1.1  BN
      conv3.0  Conv2d(128,256,3)   conv3.1  BN(256)
      conv4.0  Conv2d(256,512,3)   conv4.1  BN(512)
      res2     ModuleList of 2 ResUnit(512)
        res2.0.0  Conv  res2.0.1  BN
        res2.1.0  Conv  res2.1.1  BN
      classifier.0  AdaptiveAvgPool2d
      classifier.1  Flatten
      classifier.2  Dropout
      classifier.3  Linear(512, 8)
    """
    def __init__(self, num_classes=8):
        super().__init__()

        # conv*.0 = Conv,  conv*.1 = BN  (no extra layers so indices stay 0,1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
        )
        self.res1 = nn.ModuleList([ResUnit(128), ResUnit(128)])

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
        )
        self.res2 = nn.ModuleList([ResUnit(512), ResUnit(512)])

        # indices: 0=AvgPool  1=Flatten  2=Dropout  3=Linear
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        for unit in self.res1:
            x = unit(x)
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.relu(self.conv4(x))
        for unit in self.res2:
            x = unit(x)
        return self.classifier(x)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\n  Model not found: '{MODEL_PATH}'\n"
            f"  Put the .pth file in the same folder as app.py\n"
        )

    model = BloodGroupCNN(num_classes=len(CLASS_NAMES))

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    log.info("Model loaded  |  device=%s", DEVICE)
    return model


preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def prepare_image(file_bytes):
    img    = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    return tensor.to(DEVICE)


def predict(tensor):
    with torch.no_grad():
        logits = MODEL(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    idx        = int(probs.argmax())
    confidence = float(probs[idx]) * 100
    blood_type = CLASS_NAMES[idx]
    info       = BLOOD_INFO[blood_type]

    top3 = [
        {"type": CLASS_NAMES[i], "prob": round(float(probs[i]) * 100, 2)}
        for i in probs.topk(3).indices.tolist()
    ]

    log.info("Predicted: %s  |  confidence: %.1f%%  |  top3: %s", blood_type, confidence, top3)

    return {
        "blood_type":   blood_type,
        "confidence":   round(confidence, 2),
        "abo_group":    info["abo"],
        "rh_factor":    info["rh"],
        "donate_to":    info["donate_to"],
        "receive_from": info["receive_from"],
        "rarity":       info["rarity"],
        "top3":         top3,
    }


app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("blood-group-detection.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE), "classes": CLASS_NAMES})


@app.route("/api/predict", methods=["POST"])
def predict_api():
    if "image" not in request.files:
        return jsonify({"error": "No image field. Use key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    img_bytes = file.read()

    try:
        tensor = prepare_image(img_bytes)
        t0     = time.perf_counter()
        result = predict(tensor)
        result["time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return jsonify(result)

    except Exception as e:
        log.exception("Inference failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    MODEL = load_model()
    log.info("Starting  ->  http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)