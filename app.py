import os
from flask import Flask, request, send_file, jsonify
from rembg import remove
import cv2
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Increase timeout for the Flask app (e.g., 30 seconds)
TIMEOUT_SECONDS = 30

@app.route("/", methods=["GET"])
def index():
    print("Endpoint: /")
    return "Hello User! Now you have successfully called the WMS Image API"

@app.route("/get-finalimage", methods=["POST", "GET"])
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

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000, host='0.0.0.0', timeout=TIMEOUT_SECONDS)
