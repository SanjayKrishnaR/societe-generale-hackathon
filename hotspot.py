import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load datasets
incidents_df = pd.read_csv('incident1.csv')
resolutions_df = pd.read_csv('resolution.csv')
incident_history_df = pd.read_csv('incident_history.csv')

# Merge incidents and resolutions based on Incident_Type
merged_df = incidents_df.merge(resolutions_df, on='Incident_Type', how='inner')
file_path="merged.csv"
merged_df.to_csv(file_path, index=False)

# Create a label encoder for incident types
label_encoder = LabelEncoder()
merged_df['Incident_Type'] = label_encoder.fit_transform(merged_df['Incident_Type'])

# Split data into features (incident descriptions) and target (Incident_Type)
X = merged_df['Description']
y = merged_df['Incident_Type']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization of incident descriptions
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the number of features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Let's assume we have a new incident description
new_incident_description = ["Unauthorized access attempt detected on server"]

# Vectorize the new incident description using the same TF-IDF vectorizer
new_description_tfidf = tfidf_vectorizer.transform(new_incident_description)

# Predict the incident type for the new incident description
predicted_incident_type = classifier.predict(new_description_tfidf)
predicted_incident_type_label = label_encoder.inverse_transform(predicted_incident_type)

# Print the predicted incident type
print(f'Predicted Incident Type: {predicted_incident_type_label[0]}')

# Find the corresponding resolution based on the predicted incident type
predicted_resolution = merged_df[merged_df['Incident_Type'] == predicted_incident_type[0]]['Resolution'].values[0]

# Print the recommended resolution
print(f'Recommended Resolution: {predicted_resolution}')


your_model = classifier

# Specify the filename where you want to save the model
filename = "your_model.pkl"

# Open the file in binary write mode and save the model
with open(filename, 'wb') as file:
    pickle.dump(your_model, file)

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)