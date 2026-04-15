# 🩸 HemaVision — Blood Group Detection

AI-powered blood group classification using deep learning (TensorFlow/Keras).  
Identifies **ABO group** and **Rh factor** from blood sample images.

---

## 📁 Project Structure

```
Blood_group_detection_using_fingerprint/
├── app.py                     # Flask backend (API + static serving)
├── blood_group_detection.html # Frontend UI
├── my_model.keras             # ← Your trained Keras model (place here)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── static/  (optional)
│   ├── css/
│   └── js/
│
├── dataset/  (optional)       # Training images
└── .venv/                     # Virtual environment
```

---

## ⚡ Quick Start

### 1. Create & activate virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place your model

Copy your trained model file into the project root:

```
BGD/my_model.keras
```

> ⚠️ **Model not ready yet?**  
> No problem — the app runs in **mock mode** automatically. It returns random
> blood types so you can test the UI end-to-end.

### 4. Run the server

```bash
python app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000
```

---

## 🔌 API Endpoints

### `GET /health`

Returns backend status and model state.

```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
  "img_size": [224, 224]
}
```

---

### `POST /predict`

Send a blood sample image for classification.

**Request:** `multipart/form-data`  
**Field:** `file` — image file (PNG, JPG, JPEG, BMP, TIFF, WEBP)  
**Max size:** 16 MB

**Response:**
```json
{
  "blood_type":     "A+",
  "abo_group":      "A",
  "rh_factor":      "Positive (+)",
  "confidence":     0.9732,
  "reasoning":      "The model identified ABO group 'A' with positive Rh factor...",
  "mock":           false,
  "processing_ms":  124.5
}
```

---

## 🧠 Model Specification

| Property        | Value                                 |
|-----------------|---------------------------------------|
| Input shape     | `(224, 224, 3)` — RGB image           |
| Normalization   | Divide by 255 (values in `[0, 1]`)    |
| Output          | Softmax over 8 classes                |
| Classes         | A+, A−, B+, B−, AB+, AB−, O+, O−    |
| Model file      | `my_model.keras`                      |
| Load method     | `tf.keras.models.load_model(..., compile=False)` |

---

## 🎨 Frontend Features

- **Dark medical theme** — deep red palette, futuristic aesthetic
- **Drag-and-drop** upload with animated border glow
- **Scanning animation** — red scan line over image during analysis
- **Step-by-step indicator** — shows preprocessing → inference stages
- **Result reveal** — blood type pops in with scale animation
- **Confidence bar** — animated gradient fill with glow effect
- **Floating particles** — red blood cell particles in background
- **Custom cursor** — red dot that expands on hover
- **Backend status pill** — live green/red indicator
- **Error UI** — "Analysis Failed" card with explanation
- **Fully responsive** — works on mobile

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `"Failed to fetch"` error | Make sure `python app.py` is running and visit via `http://127.0.0.1:5000` (not by opening the HTML file directly) |
| `model_loaded: false` | Place `my_model.keras` in the project root |
| Import errors | Run `pip install -r requirements.txt` inside your venv |
| Port in use | Change `port=5000` to another port in `app.py` |
| GPU errors | TensorFlow will fall back to CPU automatically |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `flask` | Web server and routing |
| `tensorflow` | Keras model loading and inference |
| `Pillow` | Image opening and resizing |
| `numpy` | Array operations and normalization |

---

*HemaVision v1.0 — Built with Flask + TensorFlow*