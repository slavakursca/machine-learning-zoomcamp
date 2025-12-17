import onnxruntime as ort
from PIL import Image
from io import BytesIO
from urllib import request
import numpy as np
from torchvision import transforms

# --- Model and Data URLs ---
PREFIX = "https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle"
DATA_URL = f"{PREFIX}/hair_classifier_v1.onnx.data"
MODEL_URL = f"{PREFIX}/hair_classifier_v1.onnx"

# --- Download Model (if not already present) ---
def download_model_files():
    # In a real Docker image, these files might be copied directly
    # or downloaded during image build. For a script run locally, we download.
    print(f"Downloading {DATA_URL}...")
    request.urlretrieve(DATA_URL, "hair_classifier_v1.onnx.data")
    print(f"Downloading {MODEL_URL}...")
    request.urlretrieve(MODEL_URL, "hair_classifier_v1.onnx")

# --- Initialize ONNX Runtime Session ---
def initialize_model(onnx_model_path="hair_classifier_v1.onnx"):
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name

# --- Image Preprocessing Functions ---
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# --- Prediction Function ---
def predict(image_url, session, input_name, output_name, target_size=(200, 200)):
    # Define the same transformation pipeline used in the notebook
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = download_image(image_url)
    resized_img = prepare_image(img, target_size)
    image_tensor = transform_pipeline(resized_img)
    image_np = image_tensor.numpy()  # Convert PyTorch tensor to NumPy array
    image_with_batch = np.expand_dims(image_np, axis=0) # Add batch dimension

    result = session.run([output_name], {input_name: image_with_batch})
    return result[0][0].tolist()

# --- Example Usage (when run as a script) ---
if __name__ == "__main__":
    download_model_files()
    session, input_name, output_name = initialize_model()

    test_image_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    prediction_score = predict(test_image_url, session, input_name, output_name)
    print(f"Prediction for {test_image_url}: {prediction_score}")

    # You can add more test cases here
    test_image_url_2 = "https://www.hairfinder.com/tips/womanlonghair.jpg"
    prediction_score_2 = predict(test_image_url_2, session, input_name, output_name)
    print(f"Prediction for {test_image_url_2}: {prediction_score_2}")