from flask import Flask, render_template, request, redirect, url_for
import os
import fitz  # PyMuPDF
import joblib
import pandas as pd
import numpy as np
import re

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load ML Model
model = joblib.load("trained_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to extract values from text
def extract_test_values(text):
    patterns = {
        "Hemoglobin (g/dL)": r"Hemoglobin\s*\(g/dL\)?\s*:?\s*([\d\.]+)",
        "WBC Count (thousand cells/uL)": r"WBC Count\s*\(thousand cells/uL\)?\s*:?\s*([\d\.]+)",
        "Platelet Count (thousand/uL)": r"Platelet Count\s*\(thousand/uL\)?\s*:?\s*([\d]+)",
        "Fasting Blood Sugar (mg/dL)": r"Fasting Blood Sugar\s*\(mg/dL\)?\s*:?\s*([\d]+)",
        "HbA1c (%)": r"HbA1c\s*\(%\)?\s*:?\s*([\d\.]+)",
    }
    
    extracted_values = {}
    for test, pattern in patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            extracted_values[test] = float(match.group(1))
    
    return extracted_values

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    return redirect(url_for('predict', file_path=file_path))

@app.route('/predict')
def predict():
    file_path = request.args.get('file_path')
    text = extract_text_from_pdf(file_path)
    test_values = extract_test_values(text)
    
    formatted_input = {feature: test_values.get(feature, 0) for feature in feature_names}
    df_input = pd.DataFrame([formatted_input])
    df_input = df_input[feature_names]
    df_input_scaled = scaler.transform(df_input)
    prediction = model.predict(df_input_scaled)
    
    disease_labels = ["Diabetes", "Typhoid", "Malaria", "Thyroid Disorder", "High Cholesterol", "Liver Disease", "Stomach Infection"]
    results = {disease_labels[i]: "Positive" if prediction[0][i] == 1 else "Negative" for i in range(len(disease_labels))}
    
    return render_template('result.html', results=results)

# HTML Templates
upload_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Upload Blood Report</title>
</head>
<body>
    <h2>Upload Blood Report (PDF)</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".pdf" required>
        <br><br>
        <button type="submit">Upload & Predict</button>
    </form>
</body>
</html>
"""

result_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Results</title>
</head>
<body>
    <h2>Prediction Results</h2>
    <table border="1">
        <tr>
            <th>Disease</th>
            <th>Prediction</th>
        </tr>
        {% for disease, result in results.items() %}
        <tr>
            <td>{{ disease }}</td>
            <td>{{ result }}</td>
        </tr>
        {% endfor %}
    </table>
    <br>
    <a href="/">Upload Another Report</a>
</body>
</html>
"""

# Save HTML Templates
with open("templates/upload.html", "w") as f:
    f.write(upload_html)

with open("templates/result.html", "w") as f:
    f.write(result_html)

if __name__ == '__main__':
    app.run(debug=True)
