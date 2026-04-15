# 🧬 HemaTrace — Blood Group Detection via Fingerprint

> 🚀 AI-powered system that predicts blood group using fingerprint images
> ⚡ Fast • Non-invasive • Deep Learning-based

---

## 🌟 Overview

**HemaTrace** is an innovative deep learning project that predicts a person's **blood group (ABO + Rh factor)** using fingerprint images.

Traditional blood testing is invasive and time-consuming.
This project explores a **non-invasive alternative** using **Computer Vision + CNNs**.

---

## 🎯 Key Features

✨ Upload fingerprint image and get instant prediction
🧠 Deep Learning model trained on fingerprint patterns
📊 Displays confidence score
🩸 Shows blood compatibility (donate/receive)
🌐 Flask API + Streamlit UI support
⚡ Fast inference (< 3 seconds)

---

## 🖼️ Demo Preview

![Snapshot1]('photos/photo1.png')

---

## 🧠 Tech Stack

| Category         | Technology Used |
| ---------------- | --------------- |
| Language         | Python 🐍       |
| ML Framework     | PyTorch 🔥      |
| Backend          | Flask           |
| Frontend         | HTML, CSS, JS   |
| UI App           | Streamlit       |
| Image Processing | PIL, OpenCV     |

---

## 📁 Project Structure

```
BGD/
│── app.py                  # Flask backend
│── streamlit_app.py        # Streamlit UI
│── templates/
│   └── blood-group-detection.html
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Setup

### 🔹 1. Clone the repository

```bash
git clone https://github.com/niranjan-18-25/Blood-group-detection-by-fingerprint.git
cd Blood-group-detection-by-fingerprint
```

### 🔹 2. Create virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 🔹 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### 🚀 Run Flask API

```bash
python app.py
```

Open: http://127.0.0.1:5000/

---

### 🎨 Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## ⚠️ Important Note

📌 The trained model file (`.pth`) is **not included** due to size limitations.
You need to place it manually in the project directory.

---

## 📊 Model Details

* Input Size: 128×128 RGB images
* Architecture: CNN with Residual Blocks
* Output Classes: 8 blood types
* Accuracy: ~93%

---

## 🚧 Future Improvements

* 🔬 Improve dataset size & diversity
* 📱 Mobile app integration
* ☁️ Deploy on cloud (AWS / Render / HuggingFace)
* 📈 Add Grad-CAM visualization
* 🧠 Improve model accuracy

---

## 🙋‍♂️ Author

**Niranjan**
📍 Hassan, Karnataka
🎓 AI & ML Student

---

## ⭐ Support

If you like this project:

👉 Star this repository
👉 Share with others
👉 Give feedback

---

## 📌 Disclaimer

This project is for **educational and research purposes only**.
Not intended for medical use.

---

💡 *“Where AI meets healthcare innovation.”*
