import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import os

st.title("Multi-Modality Clinical AI Reporting System")
st.write("Upload a medical image and select modality to generate an AI-powered radiology report.")

modalities = {
    "X-ray": "data/XRAY",
    "CT": "data/CT",
    "Mammography": "data/MAMMO",
    "Ultrasound": "data/US",
    "MRI": "data/MRI"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_report_dict(modality_folder):
    report_dict = {}
    for fname in os.listdir(modality_folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            txt_path = os.path.splitext(fname)[0]+'.txt'
            txt_path_full = os.path.join(modality_folder, txt_path)
            if os.path.exists(txt_path_full):
                with open(txt_path_full, "r", encoding="utf8") as f:
                    report_dict[fname] = f.read().strip()
    return report_dict

def load_dataset(modality_folder):
    images, labels = [], []
    for fname in os.listdir(modality_folder):
        if fname.lower().endswith(('.jpg','.jpeg','.png')):
            img_path = os.path.join(modality_folder, fname)
            txt_path = os.path.splitext(img_path)[0]+'.txt'
            if os.path.exists(txt_path):
                images.append(img_path)
                with open(txt_path, "r", encoding="utf8") as f:
                    labels.append(f.read().strip())
    label_names = list(set(labels))
    label2idx = {name: i for i, name in enumerate(label_names)}
    idx2label = {i: name for name, i in label2idx.items()}
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    return images, labels, label2idx, idx2label, tf

@st.cache_resource
def train_model(images, labels, label2idx, tf):
    if len(set(labels)) < 2 or not images: return None
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(label2idx))
    model = model.to(device)
    model.train()
    X = []
    y = []
    for img_path, label in zip(images, labels):
        img = tf(Image.open(img_path).convert("RGB"))
        X.append(img.unsqueeze(0))
        y.append(label2idx[label])
    X = torch.cat(X).to(device)
    y = torch.tensor(y).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(3):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model

modality = st.selectbox("Select Modality", list(modalities.keys()))
uploaded = st.file_uploader("Upload Image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
modality_path = modalities[modality]
report_dict = load_report_dict(modality_path)
images, labels, label2idx, idx2label, tf = load_dataset(modality_path)
model = train_model(images, labels, label2idx, tf)

def clinical_report(uploaded_image, report_dict, model, label2idx, idx2label, tf, threshold=0.75):
    if uploaded_image is None: return ""
    imgname = uploaded_image.name
    img = Image.open(uploaded_image).convert("RGB")
    img_tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)
        label = idx2label[idx.item()]
    if imgname in report_dict and conf.item() > threshold:
        return report_dict[imgname]
    elif conf.item() <= threshold:
        suggestions = ", ".join(list(label2idx.keys())[:3])
        return f"Possible findings: {label} (suggestions: {suggestions}). Further expert review recommended."
    return f"AI Model Prediction: {label}"

if uploaded:
    st.image(uploaded, use_column_width=True)
    if model:
        report = clinical_report(uploaded, report_dict, model, label2idx, idx2label, tf)
        st.markdown(f"### Radiology Report:\n{report}")
    else:
        st.warning("No model available (not enough training data for this modality).")
