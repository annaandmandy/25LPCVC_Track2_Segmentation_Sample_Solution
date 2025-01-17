# Baseline Solution - Track 2: Open-Vocabulary Segmentation with Text-Prompt (LPCVC 2025)

## :fire: News
- [2025.02.01] Sample solution of Track2: OVSeg is released
- [2025.01.10] LPCVC 2025 is accepted as CVPR 2025 Workshop
- [2024.12.10] LPCVC 2025 is announced on NeurIPS 2024

### 1. Model Training and Evaluation
:point_right: ***\*Please refer to [[XDecoder]]() for more details about model training and evaluation.***
- Architecture: Focal-T / ViT-b
- Training data: COCO
- Evaluation data: RefCOCOg
- Task: Grounding Segmentation
- Finetuned model weights for LPCVC Track2:
  - Training: `sh command.sh`
  - Init weights: [[Google Drive]](https://drive.google.com/file/d/1zTaVW_I4fe6MSBq5GAg284TuQfLcB0Yd/view?usp=drive_link)
- :bulb: Hints:
  - Higher resolution of input image usually increase the segmentation accuracy, but also invovle more computational cost. There is always a trade-off.

### 2. Compiling and Profiling on Qualcomm Chips via AI Hub
:point_right: ***\* Please refer to [[AI Hub]]() documents for more details regarding model compiling, profiling, and inference.***

```python
    # Compile model on a specific device
    compile_job = qai_hub.submit_compile_job(
        model=f"./xdecoder_lpcvc25.onnx",
        name="xdecoder_ovss",
        device=qai_hub.Device("Snapdragon X Elite CRD"),
        options="--truncate_64bit_io --target_runtime qnn_context_binary",
    )
    compiled_model = compile_job.get_target_model().download(f"./xdecoder_lpcvc25.bin")

    image, text_emb, text_attn_mask = prepare_data(img_path, text)
    input_array = (image, text_emb, text_attn_mask)

    # Submit an inference job for the model.
    inference_job = qai_hub.submit_inference_job(
        model=compiled_model,
        device=qai_hub.Device("Snapdragon X Elite CRD"),
        inputs=input_array
    )
    output_array = inference_job.download_output_data()
```

### 3. Inference and Evaluation
- :point_right: ***\* Please check the scripts [[.evaluate_model.py]]() more details of inference the on AIHub***
- **Device**: Snapdragon X Elite CRD
- **Test Details**: During inference and evaluate all submitted solutions on AIHub, we prepare all input data and ground-truth to the same format and size to make it fair to all participants. Specifically,
  - *Input*: 
    - ***Image***: RGB, shape=3x1024x1024 # resize the longest edge to 1024, then padded to 1024x1024 square
    - ***Text_emb***: , shape=1x77 # output of openai-clip tokenizer
    - ***text_attn_mask***: shape=1x77 # output of tokenizer, binary values
  - *Output*: 
    - Mask prediction: binary matrix, shape=1x1024x1024 # used to calculate the IoU with ground-truth mask
- **Evaluation Metric**
  - **mIoU**: IoU of all test samples
    ```python
    def computeIoU(pred_seg, gd_seg):
        I = (pred_seg & gd_seg)
        U = (pred_seg | gd_seg)
        return I, U
    # compute mIoU over all test image-text pairs
    pred = output['grounding_mask'].sigmoid() > 0.5
    gt = input['groundings']['masks'].bool()
    bsi = len(pred)
    I, U = self.computeIoU(pred, gt)
    IoU = I.reshape(bsi,-1).sum(-1)*1.0 / (U.reshape(bsi,-1).sum(-1) + 1e-6)
    self.mIoU += IoU.sum().cpu()
    ```
- **Test Data Format**:
  Every image and a text description will be input to the model after the following preparation operations to make the input format fixed. The corresponding mask of the text description is the ground-truth. 
  - **Image Input**: We have 1000 images from around 200 categories, and each image is annotated with 3~5 masks of objects/stuff with various sizes and classes. We tried our best to make the test dataset balanced across mask sizes, categories, and more. All the input images have the same input shape 3x1024x1024 with RGB values [0, 255]. The original images are first resized to make the longest edge equals 1024, then padded to square 1024x1024 by 0s.
    ```python
    image = utils.read_image(img_path, format='RGB')
    transform = []
    transform.extend([T.ResizeShortestEdge(1024, max_size=1024),])    
    image, _ = T.apply_transform_gens(transform, image)
    pad_image = numpy.zeros((1024, 1024, 3), numpy.uint8)
    pad_image[:image.shape[0], :image.shape[1]] = image
    pad_image = torch.as_tensor(numpy.ascontiguousarray(pad_image.transpose(2, 0, 1))).cuda()
    input_iamge = torch.unsqueeze(pad_image, 0) # shape=1x3x1024x1024
    ```
  - **Text Input**: Each annotated mask is assigned 3~5 text descriptions. The textual descriptions include keywords, short phrases, long sentences describing the appearance, location, spatial relationships, or semantic knowledge of the target objects/stuff. (*Text tokenization*) QNN library does not support tokneization of text input yet. In order to reduce the influence of different text tokenzer used to the final performance, accuracy and latency, we pre-fixed the text tokenzier and only input the tokenized vector of the input text to the model as below:
    ```python
    # prefixed text tokenizer
    from transformers import CLIPTokenizer
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    pretrained_tokenizer = 'openai/clip-vit-base-patch32'
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
    tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})

    # example tokenized text embedding input to the model
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
    text_emb = tokens['input_ids'].cuda() # shape=1x77
    text_attn_mask = tokens['attention_mask'].cuda() # shape=1x77
    input_text = [text_emb, text_attn_mask]
    ```

## Acknowledgement
* The baseline is built on top of [XDecoder]()

## Contact
LPCVC 2025 Organizers: [[Homepage]](lpcv.ai) [[slack]](https://aihub.qualcomm.com/community/slack) [[Email]](lowpowervision@gmail.com)
