import base64
import io

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)


# Function to load the YOLO model with specified task
def load_model(model_path, task='segment'):
    model = YOLO(model_path, task=task)
    return model


# Function to extract the full hairline area
def extract_full_hairline_area(original_image, mask):
    # Ensure the mask is a binary mask (0 or 255)
    binary_mask = mask[0] * 255

    # Resize the mask to match the original image dimensions
    binary_mask = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))

    # Create an alpha channel based on the binary mask
    alpha_channel = binary_mask.astype(original_image.dtype)

    # Split the original image into its color channels
    b_channel, g_channel, r_channel = cv2.split(original_image)

    # Combine the color channels and alpha channel into a BGRA image
    transparent_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    return transparent_image


# Function to calculate final hair density based on class confidences
def calculate_final_hair_density(response_data):
    # Hair density rates based on class names
    hair_density_rates = {
        "1": 96,
        "2": 85,
        "3": 75,
        "4": 65,
        "5": 50,
        "6": 35,
        "7": 20
    }

    final_density = 0.0
    for entry in response_data:
        class_name = entry['name']
        confidence = entry['confidence']
        density_rate = hair_density_rates.get(class_name, 0)
        weighted_density = density_rate * confidence
        final_density += weighted_density

    return final_density


# Load the segmentation model
model_path = './segmentation1.pt'  # Replace with your model path
segmentation_model = load_model(model_path, task='segment')

# Load MobileNetV2 model (local)
mobilenet_model_path = "./ds5_model - 2 june 2024 21_56.tflite"  # Replace with the actual path
interpreter = tf.lite.Interpreter(model_path=mobilenet_model_path)
interpreter.allocate_tensors()

# Get input and output details for MobileNetV2
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert("RGBA")

    # Convert the PIL image to an OpenCV image (numpy array)
    original_image = np.array(image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)

    # Get the model output (segmentation mask)
    segmentation_result = segmentation_model(original_image)
    mask = segmentation_result[0].masks.data.cpu().numpy()

    # Extract the full hairline area using the mask
    transparent_image = extract_full_hairline_area(original_image, mask)

    # Convert to PIL Image
    transparent_image_pil = Image.fromarray(cv2.cvtColor(transparent_image, cv2.COLOR_BGRA2RGBA))

    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    transparent_image_pil.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode the image to base64 string
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Resize the image
    img_try = transparent_image_pil.resize((640, 640))
    new_img = Image.new("RGB", image.size, (255, 255, 255))
    new_img.paste(transparent_image_pil, (0, 0), transparent_image_pil.split()[3])
    new_img = new_img.resize((640, 640))

    # Convert to NumPy array and preprocess
    img_array = np.array(new_img) / 255.0
    img_array = img_array.astype(np.float32)
    img_array = img_array[np.newaxis, ...]

    # Set the input tensor (now with correct data type)
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run the model
    interpreter.invoke()

    # Get the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Prepare the response data in the expected format
    response_data = [{"class": i + 1, "confidence": float(predictions[0][i]), "name": str(i + 1)} for i in
                     range(len(predictions[0]))]

    # Calculate the final hair density
    final_hair_density = calculate_final_hair_density(response_data)

    return jsonify({
        "final_hair_density": final_hair_density,
        "predictions": response_data,
        "masked_image": img_str  # base64 encoded image
    })


if __name__ == '__main__':
    app.run(debug=True)
