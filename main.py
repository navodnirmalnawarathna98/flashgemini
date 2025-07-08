import os
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import google.generativeai as genai

# --- Flask App Setup ---
app = Flask(__name__)

# --- Configure Gemini API ---
# IMPORTANT: Store your API key securely, preferably as an environment variable.
# DO NOT hardcode your API key in production code.
# For local development, you can set it like this:
# set GEMINI_API_KEY=YOUR_API_KEY_HERE (Windows CMD)
# $env:GEMINI_API_KEY="YOUR_API_KEY_HERE" (Windows PowerShell)
# export GEMINI_API_KEY="YOUR_API_KEY_HERE" (Linux/macOS)
try:
    genai.configure(api_key="AIzaSyBzKUAFWfH9HAoRdPqLCejxIy-k4M2vjXw")
except KeyError:
    print("Key error.")
    exit(1) # Exit if API key is not found

# Initialize the Gemini Vision Model
try:
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    print("Ensure you have a working internet connection and the API key is valid.")
    exit(1)

# --- API Endpoint ---
@app.route('/read_prescription', methods=['POST'])
def read_prescription():
    # 1. Check if an image file is present in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided in the request."}), 400

    image_file = request.files['image']

    # Check if the file is empty
    if image_file.filename == '':
        return jsonify({"error": "No selected image file."}), 400

    # Check if the file has a valid content type (optional, but good practice)
    if not image_file.content_type.startswith('image/'):
        return jsonify({"error": "Invalid file type. Please upload an image."}), 400

    try:
        # 2. Read the image data from the Flask request and prepare for PIL
        image_data = image_file.read()
        pil_image = Image.open(BytesIO(image_data))

        # 3. Prepare the prompt for Gemini
        # We'll ask for only the medicine names as previously refined
        prompt_parts = [
            pil_image,
            "List only the names of the medicines exactly as they appear in this prescription, one per line. Do not include dosages, frequencies, or any other details. For example: 'Medicine A', 'Medicine B'.",
        ]

        # 4. Send the image and prompt to Google Gemini
        gemini_response = gemini_model.generate_content(prompt_parts)

        # 5. Extract the text response from Gemini
        medicine_names = gemini_response.text.strip()

        # 6. Return the extracted medicine names to the web server API
        return jsonify({"medicine_names": medicine_names}), 200

    except Exception as e:
        # Generic error handling for unexpected issues
        print(f"An error occurred: {e}")
        return jsonify({"error": f"Failed to process prescription: {str(e)}"}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    # It's recommended to run Flask in debug mode only for development.
    # For production, use a production-ready WSGI server like Gunicorn or uWSGI.
    app.run(debug=True, host='0.0.0.0', port=5000)
