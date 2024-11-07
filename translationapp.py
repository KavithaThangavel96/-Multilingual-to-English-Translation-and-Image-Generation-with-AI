import streamlit as st
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load models with caching
@st.cache_resource
def load_translation_model():
    model_name = "facebook/mbart-large-50-many-to-one-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="ta_IN")
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_sd_pipeline():
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipeline.to(device)

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Initialize cached resources
translation_tokenizer, translation_model = load_translation_model()
sd_pipeline = load_sd_pipeline()
blip_processor, blip_model = load_blip_model()

# Translation function
def translate_to_english(tamil_text):
    inputs = translation_tokenizer(tamil_text, return_tensors="pt", padding=True)
    translated = translation_model.generate(**inputs)
    english_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
    return english_text

# Image generation function
def generate_image_from_text(english_text):
    with torch.no_grad():
        image = sd_pipeline(english_text).images[0]
    return image

# Creative description generation function
def generate_creative_text_from_image(image):
    inputs = blip_processor(images=image, return_tensors="pt")
    caption = blip_model.generate(**inputs)
    creative_text = blip_processor.decode(caption[0], skip_special_tokens=True)
    return creative_text

# Streamlit UI
st.title("Tamil to English Translation and Image Generation")

# Input for Tamil text
tamil_text = st.text_area("Enter Tamil Text", placeholder="Type Tamil text here...")

# Button to trigger the translation, image generation, and description
if st.button("Generate"):
    if tamil_text:
        # Step 1: Translate Tamil text to English
        english_text = translate_to_english(tamil_text)
        st.subheader("English Translation")
        st.write(english_text)

        # Step 2: Generate image from English text
        with st.spinner("Generating image..."):
            image = generate_image_from_text(english_text)
        st.subheader("Generated Image")
        st.image(image, caption="Generated based on translation", width=400)

        # Step 3: Generate creative description from the generated image
        with st.spinner("Generating creative description..."):
            creative_text = generate_creative_text_from_image(image)
        st.subheader("Creative Description")
        st.write(creative_text)
    else:
        st.warning("Please enter some Tamil text.")
