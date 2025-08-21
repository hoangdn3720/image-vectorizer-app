import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree

# Configuration
UPLOAD_FOLDER = 'uploads'
VECTOR_FOLDER = 'vectors'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VECTOR_FOLDER'] = VECTOR_FOLDER

# Ensure upload and vector folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def vectorize_image_to_svg(img_path: str, svg_path: str, approx_epsilon: float = 2.0) -> None:
    """
    Convert a raster image to a simplistic vector representation (SVG) by
    detecting contours. This function uses Canny edge detection to find edges
    and then extracts the contours of these edges. The resulting contours
    are approximated and written as SVG path elements.

    Parameters
    ----------
    img_path : str
        The path to the input image file.
    svg_path : str
        The path where the output SVG file will be saved.
    approx_epsilon : float, optional
        Approximation accuracy. Lower values keep more detail but produce
        larger SVG files. Higher values simplify the shapes.
    """
    # Read input image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Unable to read image file: {img_path}")
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to smooth the image and reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Create root SVG element with viewBox equal to image dimensions
    height, width = edges.shape
    svg = Element('svg', xmlns="http://www.w3.org/2000/svg",
                  width=str(width), height=str(height),
                  viewBox=f"0 0 {width} {height}")
    # Loop through each contour and convert to path data
    for contour in contours:
        # Approximate contour to reduce number of points
        approx = cv2.approxPolyDP(contour, epsilon=approx_epsilon, closed=True)
        if len(approx) <= 1:
            continue
        # Construct path string
        commands = []
        first_pt = approx[0][0]
        commands.append(f"M {first_pt[0]} {first_pt[1]}")
        for pt in approx[1:]:
            x, y = pt[0]
            commands.append(f"L {x} {y}")
        commands.append('Z')  # Close the path
        d = " ".join(commands)
        # Add path element to SVG. Set fill to none and stroke to black.
        SubElement(svg, 'path', d=d, fill='none', stroke='black', stroke_width='1')
    # Save SVG to file
    ElementTree(svg).write(svg_path)


@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the upload form or process the uploaded image."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error="No file part in the request"), 400
        file = request.files['file']
        # If no file selected
        if file.filename == '':
            return render_template('index.html', error="No file selected"), 400
        # Validate and save file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            upload_name = f"{unique_id}_{filename}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_name)
            file.save(upload_path)
            # Generate SVG filename and path
            svg_filename = f"{unique_id}.svg"
            svg_path = os.path.join(app.config['VECTOR_FOLDER'], svg_filename)
            # Vectorize the image
            try:
                vectorize_image_to_svg(upload_path, svg_path)
            except Exception as e:
                return render_template('index.html', error=f"Error processing image: {e}"), 500
            # Redirect to result page
            return redirect(url_for('result', svg_filename=svg_filename))
        else:
            return render_template('index.html', error="Unsupported file type"), 400
    # GET request
    return render_template('index.html')


@app.route('/result/<svg_filename>')
def result(svg_filename: str):
    """Display the vectorized SVG in the browser and provide download link."""
    return render_template('result.html', svg_filename=svg_filename)


@app.route('/download/<svg_filename>')
def download(svg_filename: str):
    """Send the SVG file as a downloadable attachment."""
    file_path = os.path.join(app.config['VECTOR_FOLDER'], svg_filename)
    return send_file(file_path, as_attachment=True)


@app.route('/view_svg/<svg_filename>')
def view_svg(svg_filename: str):
    """Serve the SVG file directly so it can be embedded in HTML."""
    file_path = os.path.join(app.config['VECTOR_FOLDER'], svg_filename)
    # Use the SVG mime type so browsers display it correctly
    return send_file(file_path, mimetype='image/svg+xml')


if __name__ == '__main__':
    # Run the Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)
