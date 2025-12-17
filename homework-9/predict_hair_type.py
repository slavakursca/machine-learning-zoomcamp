import onnxruntime as ort
from PIL import Image
from io import BytesIO
from urllib import request
import numpy as np
from torchvision import transforms

# --- Global Initialization ---
# This runs once when the container starts (Cold Start)
MODEL_PATH = "hair_classifier_empty.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# --- The Handler Function ---
def lambda_handler(event, context):
    # 'event' usually contains the JSON input. 
    # Adjust 'url' key based on how you send your request.
    url = event.get('url') 
    
    if not url:
        return {"error": "No URL provided"}

    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = download_image(url)
    resized_img = prepare_image(img)
    image_tensor = transform_pipeline(resized_img)
    image_np = image_tensor.numpy()
    image_with_batch = np.expand_dims(image_np, axis=0)

    result = session.run([output_name], {input_name: image_with_batch})
    prediction = result[0][0].tolist()

    return {
        "prediction": prediction
    }