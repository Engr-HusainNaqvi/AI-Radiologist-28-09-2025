import streamlit as st
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoModelForImageTextToText, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

# Set up quantization config for memory efficiency
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load Lingshu-7B model and processor
@st.cache_resource
def load_lingshu_model():
    processor = AutoProcessor.from_pretrained("lingshu-medical-mllm/Lingshu-7B")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "lingshu-medical-mllm/Lingshu-7B",
        quantization_config=quant_config,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    return processor, model

# Load MedGemma-27B-IT model and processor
@st.cache_resource
def load_medgemma_model():
    processor = AutoProcessor.from_pretrained("google/medgemma-27b-it")
    model = AutoModelForImageTextToText.from_pretrained(
        "google/medgemma-27b-it",
        quantization_config=quant_config,
        device_map="auto"
    )
    return processor, model

# Function to generate report using a model (adapted for each)
def generate_report(model, processor, image, prompt, is_lingshu=False):
    if is_lingshu:
        # Lingshu-specific processing
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt + " Generate a diagnostic report."}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # MedGemma processing
        inputs = processor(prompt + " Generate a diagnostic report:", image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    return processor.decode(outputs[0], skip_special_tokens=True)

# Streamlit app interface
st.title("AI-Based Doctor Companion: Multi-Modality Medical Image Analysis")

uploaded_file = st.file_uploader("Upload a medical image (X-ray, CT, MRI, etc.)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

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
