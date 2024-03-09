from flask import Flask, send_file, jsonify
import requests
from rembg import remove
import cv2
import numpy as np
from io import BytesIO
from urllib.parse import quote
from datetime import datetime
import os

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello User! Now you have successfully called the WMS Image API"

@app.route("/get-finalimage/<path:image_url>")
def process_image(image_url):
    try:
        # Download image from URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        image_data = response.content

        # Convert image data to numpy array
        image_array = np.frombuffer(image_data, np.uint8)

        # Read image using OpenCV
        input_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Apply background removal
        output_image = remove(input_image)

        # Generate filename based on current date and time
        now = datetime.now()
        filename = now.strftime("%Y%m%d_%H%M%S.png")

     # Specify the folder to save the image
        folder = 'formatted_images'
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Save the processed image in the specified folder
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, output_image)

        # Convert output image to bytes
        _, buffer = cv2.imencode('.png', output_image)
        output_data = buffer.tobytes()

        # Return processed image
        return send_file(BytesIO(output_data), mimetype='image/png'), 200
    except Exception as e:
        # Return error message in case of any exception
        error_message = f"Error processing image: {str(e)}"
        return jsonify({"error": error_message}), 500

if __name__ == "__main__":
    app.run(debug=True)
