import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
import requests

CLASSES = ['bathroom', 'bedroom', 'kitchen', 'living_room']

# ImageNet normalization (MobileNetV2)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Preprocess image for model inference
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed numpy array ready for model input
    """
    # Resize (matches transforms.Resize((224, 224)))
    img = img.resize((224, 224), Image.BILINEAR)

    # Convert to RGB just like torchvision
    img = img.convert("RGB")

    # ToTensor(): HWC → CHW and scale to [0,1]
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))

    # Normalize
    img = (img - MEAN[:, None, None]) / STD[:, None, None]

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


def predict_from_url(image_url: str, model_session) -> dict:
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    input_tensor = preprocess_image(image)

    outputs = model_session.session.run(
        [model_session.output_name],
        {model_session.input_name: input_tensor}
    )

    probs = outputs[0][0]
    class_id = int(np.argmax(probs))

    full_dict = dict(zip(CLASSES, probs.tolist()))
    sorted_probs = sorted(full_dict.items(), key=lambda x: x[1], reverse=True)

    return {
        "class": CLASSES[class_id],
        "confidence": float(probs[class_id]),
        "probabilities": sorted_probs
    }
