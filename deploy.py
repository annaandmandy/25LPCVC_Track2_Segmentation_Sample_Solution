import gradio as gr
import torch
from PIL import Image
from transformers import CLIPTokenizer
from compile_and_profile.build_baseline_model import build_baseline_model
import numpy as np

# Load the model
def load_model(conf_file="./configs/xdecoder/focalt_unicl_lang_finetune.yaml"):
    # Example inputs for initialization
    conf_file = conf_file
    dummy_image = torch.zeros((1, 3, 1024, 1024))  # Dummy image input
    dummy_text = (torch.zeros((1, 77), dtype=torch.long), torch.ones((1, 77), dtype=torch.long))  # Dummy text input
    model = build_baseline_model(dummy_image, dummy_text, test_torch_model_local=False, conf_file=conf_file)   
    return model

model = load_model()

# Define the prediction function
def predict(image, text, conf_file_name):
    conf_file = conf_files[conf_file_name]

    # Reload the model with the selected configuration file
    global model
    model = load_model(conf_file)
    # Preprocess image
    image = image.resize((1024, 1024))  # Resize image to 1024x1024
    image_array = np.array(image)  # Convert PIL Image to NumPy array
    image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).float()  # Convert to tensor (1, 3, 1024, 1024)


    # Preprocess text
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    text_input = (tokens["input_ids"], tokens["attention_mask"])

    # Run the model
    with torch.no_grad():
        pred_mask = model(image_tensor, text_input)

    # Convert prediction to image
    pred_mask = pred_mask.squeeze().cpu().numpy() * 255  # Scale to 0-255
    pred_image = Image.fromarray(pred_mask.astype("uint8"))
    return pred_image

# Create the Gradio interface
conf_files = {
    "Baseline Model Config": "./configs/xdecoder/focalt_unicl_lang_finetune.yaml",
    "DYT Config": "./configs/xdecoder/focalt_unicl_lang_dyt.yaml",
    "SwiGLU Config": "./configs/xdecoder/focalt_unicl_lang_SwiGLU.yaml",
    "SwiGLU DYT Config": "./configs/xdecoder/focalt_unicl_lang_swiGLUDYT.yaml",
        "Linear Attention Config": "./configs/xdecoder/focalt_unicl_lang_linearattention.yaml",
}

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),  # Use gr.Image instead of gr.inputs.Image
        gr.Textbox(lines=2, placeholder="Enter text description", label="Text Input"),  # Use gr.Textbox instead of gr.inputs.Textbox
        gr.Dropdown(choices=list(conf_files.keys()), value="Finetune Config", label="Select Configuration File")  # Dropdown for conf_file
    ],
    outputs=gr.Image(type="pil", label="Predicted Mask"),  # Use gr.Image instead of gr.outputs.Image
    title="XDecoder Segmentation",
    description="Upload an image and provide a text description to test the XDecoder model."
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()