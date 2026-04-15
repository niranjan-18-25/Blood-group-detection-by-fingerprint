import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ---- MODEL CLASS (same as your Flask model) ----


class ResUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.add_module("0", nn.Conv2d(channels, channels, 3, padding=1))
        self.add_module("1", nn.BatchNorm2d(channels))

    def forward(self, x):
        conv = self._modules["0"]
        bn = self._modules["1"]
        return torch.relu(bn(conv(x)) + x)


class BloodGroupCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

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

# ---- LOAD MODEL ----


model = BloodGroupCNN()

state = torch.load("FingurePrintTOBloodGroupkaggle.pth",    
                   map_location="cpu")
model.load_state_dict(state)   # strict=True default

model.eval()

# ---- LABELS ----
labels = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# ---- PREPROCESS ----
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ---- UI ----
st.title("🧬 Blood Group Detection")

uploaded_file = st.file_uploader("Upload fingerprint",
                                 type=["jpg", "png", "bmp"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img)

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    idx = torch.argmax(probs).item()

    st.success(f"Prediction: {labels[idx]}")
    st.info(f"Confidence: {probs[idx]*100:.2f}%")
