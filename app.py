import streamlit as st
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

# Determine dtype based on device
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Load Lingshu-7B model and processor
@st.cache_resource
def load_lingshu_model():
    processor = AutoProcessor.from_pretrained("lingshu-medical-mllm/Lingshu-7B")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "lingshu-medical-mllm/Lingshu-7B",
        torch_dtype=dtype,
        device_map="auto"
    )
    return processor, model

# Load MedGemma-27B-IT model and processor
@st.cache_resource
def load_medgemma_model():
    processor = AutoProcessor.from_pretrained("google/medgemma-27b-it")
    model = AutoModelForImageTextToText.from_pretrained(
        "google/medgemma-27b-it",
        torch_dtype=dtype,
        device_map="auto"
    )
    return processor, model

# Function to generate report using a model
def generate_report(model, processor, image, prompt, is_lingshu=False):
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt + " Generate a diagnostic report."}]}
    ]

    if is_lingshu:
        # Lingshu-specific processing
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    else:
        # MedGemma processing
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = generated_ids[0][input_len:]
        return processor.decode(generated_ids, skip_special_tokens=True)

# Streamlit app interface
st.title("AI-Based Doctor Companion: Multi-Modality Medical Image Analysis")

uploaded_file = st.file_uploader("Upload a medical image (X-ray, CT, MRI, etc.)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    prompt = st.text_area("Enter additional context or prompt for analysis:", "Describe the findings in this medical image.")

    if st.button("Generate Reports"):
        with st.spinner("Loading models and generating reports..."):
            # Load models
            lingshu_processor, lingshu_model = load_lingshu_model()
            medgemma_processor, medgemma_model = load_medgemma_model()

            # Generate reports
            lingshu_report = generate_report(lingshu_model, lingshu_processor, image, prompt, is_lingshu=True)
            medgemma_report = generate_report(medgemma_model, medgemma_processor, image, prompt)

        st.subheader("Lingshu-7B Generated Report")
        st.write(lingshu_report)

        st.subheader("MedGemma-27B-IT Generated Report")
        st.write(medgemma_report)
