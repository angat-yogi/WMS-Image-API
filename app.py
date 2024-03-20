import os
from flask import Flask, request, send_file, jsonify
from rembg import remove
import cv2
import numpy as np
from io import BytesIO
import easyocr

app = Flask(__name__)

# Increase timeout for the Flask app (e.g., 30 seconds)
TIMEOUT_SECONDS = 30

@app.route("/", methods=["GET"])
def index():
    print("Endpoint: /")
    return "Hello User! Now you have successfully called the WMS Image API"

@app.route("/get_final_image", methods=["POST", "GET"])
def process_image():
    try:
        print("request",request)
        print("request header",request.headers)

        if request.method == 'POST':
            if 'file' not in request.files:
                print("No file provided in the request", request.files)
                return jsonify({"error": "No file provided"}), 400
            
            uploaded_file = request.files['file']
            image_data = uploaded_file.read()
            image_array = np.frombuffer(image_data, np.uint8)
            input_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            output_image = process_image_sync(input_image)
            
            # Convert the NumPy array to bytes
            success, encoded_image = cv2.imencode('.png', output_image)
            if not success:
                return jsonify({"error": "Failed to encode image"}), 500

            # Create a BytesIO object to wrap the bytes data
            image_bytes = BytesIO(encoded_image.tobytes())

            # Return the processed image as a file
            return send_file(image_bytes, mimetype='image/png', as_attachment=True, download_name='processed_image.png'), 200
        elif request.method == 'GET':
            return "GET method is not supported for this endpoint", 405
    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500

def process_image_sync(input_image):
    return remove(input_image)


# Initialize the OCR reader
reader = easyocr.Reader(['en'])

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        # Receive image file
        image_file = request.files['image']

        # Convert image to numpy array
        image_data = image_file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Perform color analysis if needed
        dominant_colors = extract_dominant_color(image)
        print("dominant colors",dominant_colors)
        #dominant_colors_serializable = [list(color) for color in dominant_colors]
        color_percentages = calculate_color_percentages(dominant_colors, image)

        # Perform text extraction using easyocr
        extracted_text = reader.readtext(image)
        print("texts",extracted_text)

        # Construct response
        analysis_result = {
            'dominant_color': color_percentages,  # Convert to list for JSON serialization
            'text': extracted_text,
            # Add more analysis results as needed
        }

        return jsonify(analysis_result)

    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500


def extract_dominant_color(image):
    # Convert image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels
    pixels = image_rgb.reshape((-1, 3))

    # Convert to float32
    pixels = np.float32(pixels)

    # Define criteria, number of clusters (K), and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3  # Number of clusters (you can adjust this value)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to uint8 and get the dominant colors
    dominant_colors = [tuple(color) for color in centers.astype(np.uint8)]

    return dominant_colors


def calculate_color_percentages(dominant_colors, image):
    # Convert the image to grayscale for easier analysis
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the total number of pixels in the image
    total_pixels = gray_image.size

    # Calculate the percentage of each dominant color
    color_percentages = {}
    for color in dominant_colors:
        # Convert the color to a numpy array for compatibility with cv2.inRange()
        color_array = np.array(color)

        # Define a small threshold for each color component
        lower_bound = color_array - np.array([10, 10, 10])
        upper_bound = color_array + np.array([10, 10, 10])

        # Ensure that the color components stay within the valid range
        lower_bound = np.clip(lower_bound, 0, 255).astype(np.uint8)
        upper_bound = np.clip(upper_bound, 0, 255).astype(np.uint8)

        # Create a mask to extract pixels within the specified color range
        mask = cv2.inRange(image, lower_bound, upper_bound)

        # Count the number of pixels with the current color
        num_pixels = cv2.countNonZero(mask)

        # Calculate the percentage of pixels with the current color
        percentage = (num_pixels / total_pixels) * 100

        # Store the percentage in the color_percentages dictionary
        color_name = get_color_name(color)
        color_percentages[color_name] = round(percentage, 2)

    return color_percentages

def get_color_name(color):
    # Here you can define a mapping from RGB values to color names
    # For simplicity, let's just return the RGB values as a string
    return f"({color[0]}, {color[1]}, {color[2]})"

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000, host='0.0.0.0', timeout=TIMEOUT_SECONDS)
