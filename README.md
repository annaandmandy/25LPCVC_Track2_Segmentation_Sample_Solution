# Sample Solution - Track 2: Open-Vocabulary Segmentation with Text-Prompt (LPCVC 2025)

## :fire: News
- [2025.02.13] OpenCV Webinar by Professor Lu introducing LPCVC 2025
- [2025.02.01] Sample solution of Track2: X-Decoder released
- [2025.01.10] LPCVC 2025 accepted as CVPR 2025 Workshop
- [2024.12.10] LPCVC 2025 announced at NeurIPS 2024

## 1. Model Training and Evaluation
:point_right: ***Please refer to [[LPCVC2025_BASELINE_INSTRUCTION]](LPCVC2025_BASELINE_INSTRUCTION.md) for the baseline environment setup and training instructions. As well as some hints to obtain a better solution for the competition! 

For more architectures and details, please refer to [[XDecoder]](https://github.com/microsoft/X-Decoder).***


## 2. Model Compilation and Profiling (Qualcomm AI Hub)
For detailed instructions on model compilation, profiling and inference, refer to the [[AI Hub Documentation]](https://app.aihub.qualcomm.com/docs/).

:point_right: ***LPCVC 2025 Track 2, Sample solution compilation, profiling and inference pipeline available in [[compile_profile_inference_aihub]](./compile_and_profile)***. Run `python ./compile_and_profile/compile_profile_inference_aihub.py` to get your torch model inference, converting to onnx format, andn QNN model **compiling + profiling + inference** on AIHub all together.

```python
# Submit compilation job to AIHub
compile_job = hub.submit_compile_job(
    model=model_path,
    name="lpcvc25_track2_sample_solution",
    device=hub.Device(deploy_device),
    options="--target_runtime qnn_context_binary",
)
# IMPORTANT! You must share your compile job to lpcvc organizers thus we can pull and evalaute it.
compile_job.modify_sharing(add_emails=['lowpowervision@gmail.com'])
model = compile_job.get_target_model()

# Profile model if requested
profile_job = hub.submit_profile_job(
    name="lpcvc25_track2_sample_solution",
    model=model, 
    device=hub.Device(deploy_device)
)
```

## 3. Inference and Evaluation

:point_right: ***See [[compile_profile_inference_aihub]](./compile_and_profile) for complete inference and evaluation pipeline.***

#### :warning: Important Note
During evaluation, only the following inference commands will be used. Ensure your submitted model is correctly compiled and produces valid outputs on AIHub:

```python
# Prepare inputs
aihub_inputs = {
    'image_input': [image_input.detach().cpu().numpy()], 
    'text_input': [text_input.detach().cpu().numpy()]
}

# Run inference
inference_job = hub.submit_inference_job(
    name="lpcvc25_track2_sample_solution",
    model=model,
    device=hub.Device(deploy_device),
    inputs=aihub_inputs
)
qnn_outputs = inference_job.download_output_data() # shape=[1024, 1024], numpy.array
```

### Evaluation Details

#### Device
- Snapdragon X Elite CRD

#### Test Dataset
- 1000 images from ~200 categories
- 3-5 annotated masks per image
- Balanced across mask sizes and categories
- 3-5 text descriptions per mask
- Text descriptions include:
  - Keywords
  - Short phrases
  - Detailed sentences (appearance, location, semantics, relationships, etc.)

#### Input Format
- **Image**:
  - RGB format, shape: 3x1024x1024
  - Longest edge resized to 1024, padded to square
- **Text**:
  - Shape: 2x1x77 (tokenized embedding + attention mask)
  - Uses CLIP tokenizer output

```python
# Image preprocessing example
img = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize(1000, max_size=1024),
])
image = transform(img)
image = torch.from_numpy(np.asanyarray(image)).float().permute(2, 0, 1)
images = [image]
image_input = ImageList.from_tensors(images, size_divisibility=1024).tensor

# All the input images have the same input shape 3x1024x1024 with RGB values [0, 255]. The original images are first resized to make the longest edge equals 1024, then padded to square 1024x1024 by 0s.
```

#### Text Processing
- Uses OpenAI CLIP tokenizer
- Format: `[input_ids; attention_mask]`

```python
# Text tokenization
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
tokens = tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
text_input = torch.stack((tokens['input_ids'], tokens['attention_mask']))  # Shape: 2x1x77

# (Text tokenization) QNN library does not support tokenization of text input yet. In order to reduce the influence of different text tokenizer used to the final performance, accuracy and latency, we pre-fixed the text tokenizer and only input the tokenized vector of the input text to the model
```

#### Evaluation Metric
- **mIoU** (mean Intersection over Union)

```python
def compute_IoU(pred_seg, gd_seg):
    I = (pred_seg & gd_seg)
    U = (pred_seg | gd_seg)
    return I.sum() / (U.sum() + 1e-6)

# Compute mIoU across test set
pred = output['grounding_mask']  # Binary mask (threshold prediction.sigmoid() > 0.5)
gt = input['groundings']['masks'].bool()
IoUs = [compute_IoU(p, g) for p, g in zip(pred, gt)]
mIoU = sum(IoUs) / len(IoUs) * 100
```



## Acknowledgements
* The sample solution for LPCVC 2025 Track-2 is built on [XDecoder](https://github.com/microsoft/X-Decoder)

## Contact
LPCVC 2025 Organizers:
- Website: [Homepage](https://lpcv.ai/)
- Community: [Slack](https://aihub.qualcomm.com/community/slack)
- Email: [lowpowervision@gmail.com](mailto:lowpowervision@gmail.com)
