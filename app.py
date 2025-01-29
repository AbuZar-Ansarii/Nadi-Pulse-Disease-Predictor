import streamlit as st
import pandas as pd
import pickle
import re

# Streamlit UI
st.title("NADI PULSE DISEASE PREDICTOR")

patient_age = st.number_input("Patient Age", min_value=0, max_value=120)
nadi_patient_data = st.file_uploader("Upload patient data text file", type="txt")

# Load models and data safely
try:
    with open("nadi_model.pkl", "rb") as f:
        nadi_model = pickle.load(f)
    with open("admin_key.pkl", "rb") as f:
        admin_keys = pickle.load(f)
    with open("admin_values.pkl", "rb") as f:
        admin_values = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Strings to remove
string_to_remove = "Start nPULSE001"
r_text = "Start"

# Read uploaded file
if nadi_patient_data is not None:
    try:
        nadi_text = nadi_patient_data.getvalue().decode("utf-8").strip()
    except UnicodeDecodeError:
        st.error("Error decoding file. Please upload a valid text file.")
        st.stop()
else:
    nadi_text = ""


# Function to clean text
def clear_text(text):
    return text[:-25] if len(text) > 25 else text  # Ensure safe slicing


def remove_string_from_content(text, pattern):
    return re.sub(re.escape(pattern), "", text).strip()  # Escape regex characters


# Process text data
cleaned_text = clear_text(nadi_text)
cleaned_text = remove_string_from_content(cleaned_text, string_to_remove)
cleaned_text = remove_string_from_content(cleaned_text, r_text)

# Create DataFrame
patient_df = pd.DataFrame({"Patient Age": [patient_age], "nadi_data": [cleaned_text]})


# Function to process Nadi data
def process_nadi_data(column):
    def safe_convert_to_int(row):
        return [int(value) for value in row.split(",") if value.strip().isdigit()]

    processed_data = column.apply(
        lambda x: [safe_convert_to_int(row) for row in x.strip().split("\n") if row.strip()] if isinstance(x,
                                                                                                           str) else [])
    return processed_data


# Apply processing
patient_df["processed_nadi_data"] = process_nadi_data(patient_df["nadi_data"])


# Calculate average safely
def calculate_avg(data):
    total_sum = sum(sum(sublist) for sublist in data)
    total_count = sum(len(sublist) for sublist in data)
    return total_sum / total_count if total_count > 0 else 0  # Avoid division by zero


patient_df['avg_of_nadi_data'] = patient_df['processed_nadi_data'].apply(calculate_avg)

# Prepare input data
input_data = [[patient_df["Patient Age"][0], patient_df['avg_of_nadi_data'][0]]]
pred_df = pd.DataFrame(input_data, columns=['Patient Age', 'avg_of_nadi_data'])


# Function to find disease
def find_disease(pred_output):
    try:
        index = admin_keys.index(pred_output)
        return admin_values[index]
    except ValueError:
        return "Unknown Disease"


# Display result
if st.button("Predict Disease"):
    try:
        pred_output = nadi_model.predict(pred_df)[0]
        st.header(find_disease(pred_output))
    except Exception as e:
        st.error(f"Error during prediction: {e}")
