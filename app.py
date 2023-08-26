from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load your trained model and TF-IDF vectorizer
with open("your_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Load your dataset (with incident types and resolutions)
incidents_df = pd.read_csv('incident1.csv')
resolutions_df = pd.read_csv('resolution.csv')
merged_df= pd.read_csv('merged.csv')

# Create a label encoder for incident types
label_encoder = LabelEncoder()
incidents_df['Incident_Type'] = label_encoder.fit_transform(incidents_df['Incident_Type'])

# TF-IDF Vectorization of incident descriptions
X_tfidf = tfidf_vectorizer.transform(incidents_df['Description'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        description = request.form['description']
        description_tfidf = tfidf_vectorizer.transform([description])

        # Predict the incident type for the new incident description
        predicted_incident_type = model.predict(description_tfidf)
        predicted_incident_type_label = label_encoder.inverse_transform(predicted_incident_type)

# Print the predicted incident type
        print(f'Predicted Incident Type: {predicted_incident_type_label[0]}')

# Find the corresponding resolution based on the predicted incident type
        matching_resolutions = merged_df[merged_df['Incident_Type'] == predicted_incident_type_label[0]]['Resolution']
        if matching_resolutions.empty:
            return render_template('index.html', error='No resolution found for the predicted incident type.')
        predicted_resolution = matching_resolutions.values[0]
        return render_template('index.html', prediction=predicted_resolution)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
