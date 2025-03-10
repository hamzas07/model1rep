import fitz  # PyMuPDF
import re
import numpy as np
import pandas as pd
import joblib

def extract_text_from_pdf(pdf_path):
    """Extracts text from PDF and returns raw text."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    
    print("Extracted Raw Text from PDF:\n", text)  # Debugging print
    return text

def extract_test_values(text):
    """Extracts test values using regex, allowing flexible formatting."""
    patterns = {
        "Hemoglobin (g/dL)": r"Hemoglobin\s*\(g/dL\)?\s*:?\s*([\d\.]+)",
        "WBC Count (thousand cells/uL)": r"WBC Count\s*\(thousand cells/uL\)?\s*:?\s*([\d\.]+)",
        "Platelet Count (thousand/uL)": r"Platelet Count\s*\(thousand/uL\)?\s*:?\s*([\d]+)",
        "Fasting Blood Sugar (mg/dL)": r"Fasting Blood Sugar\s*\(mg/dL\)?\s*:?\s*([\d]+)",
        "HbA1c (%)": r"HbA1c\s*\(%\)?\s*:?\s*([\d\.]+)",
        "Total Cholesterol (mg/dL)": r"Total Cholesterol\s*\(mg/dL\)?\s*:?\s*([\d]+)",
        "LDL (mg/dL)": r"LDL Cholesterol\s*\(mg/dL\)?\s*:?\s*([\d]+)",
        "HDL (mg/dL)": r"HDL Cholesterol\s*\(mg/dL\)?\s*:?\s*([\d]+)",
        "Triglycerides (mg/dL)": r"Triglycerides\s*\(mg/dL\)?\s*:?\s*([\d]+)",
        "TSH (mIU/L)": r"TSH\s*\(mIU/L\)?\s*:?\s*([\d\.]+)",
        "T3 (ng/dL)": r"T3\s*\(ng/dL\)?\s*:?\s*([\d\.]+)",
        "T4 (ug/dL)": r"T4\s*\(ug/dL\)?\s*:?\s*([\d\.]+)",
        "ALT (U/L)": r"ALT\s*\(U/L\)?\s*:?\s*([\d]+)",
        "AST (U/L)": r"AST\s*\(U/L\)?\s*:?\s*([\d]+)",
        "ALP (U/L)": r"ALP\s*\(U/L\)?\s*:?\s*([\d]+)",
        "Bilirubin (mg/dL)": r"Bilirubin\s*\(mg/dL\)?\s*:?\s*([\d\.]+)",
        "CRP (mg/L)": r"CRP\s*\(mg/L\)?\s*:?\s*([\d\.]+)",
        "BUN (mg/dL)": r"BUN\s*\(mg/dL\)?\s*:?\s*([\d]+)",
        "Creatinine (mg/dL)": r"Creatinine\s*\(mg/dL\)?\s*:?\s*([\d\.]+)",
        "Sodium (mEq/L)": r"Sodium\s*\(mEq/L\)?\s*:?\s*([\d]+)",
        "Potassium (mEq/L)": r"Potassium\s*\(mEq/L\)?\s*:?\s*([\d\.]+)"
    }
    
    extracted_values = {}
    for test, pattern in patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            extracted_values[test] = float(match.group(1))
        else:
            print(f"❌ No match found for: {test}")  # Debugging print
    
    return extracted_values

def predict_diseases(pdf_path, model_path, scaler_path, feature_names_path):
    """Extracts values, processes input, and makes predictions."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)  # Load stored feature names
    
    text = extract_text_from_pdf(pdf_path)
    test_values = extract_test_values(text)
    
    print("Extracted Test Values from PDF:", test_values)
    print("Feature names expected by model:", feature_names)
    
    formatted_input = {feature: test_values.get(feature, 0) for feature in feature_names}
    df_input = pd.DataFrame([formatted_input])
    df_input = df_input[feature_names]  # Ensure column order matches training
    
    df_input_scaled = scaler.transform(df_input)
    print("Scaled Input Values Sent to Model:", df_input_scaled)
    
    prediction = model.predict(df_input_scaled)
    
    disease_labels = ["Diabetes", "Typhoid", "Malaria", "Thyroid Disorder", "High Cholesterol", "Liver Disease", "Stomach Infection"]
    
    results = {disease_labels[i]: "Positive" if prediction[0][i] == 1 else "Negative" for i in range(len(disease_labels))}
    return results

if __name__ == "__main__":
    pdf_path = "Diabetes_Patient_Report.pdf"  # Change path to user-uploaded file
    model_path = "trained_model.pkl"
    scaler_path = "scaler.pkl"
    feature_names_path = "feature_names.pkl"  # Path to stored feature names
    
    predictions = predict_diseases(pdf_path, model_path, scaler_path, feature_names_path)
    print("Predicted Diseases:")
    for disease, result in predictions.items():
        print(f"{disease}: {result}")
