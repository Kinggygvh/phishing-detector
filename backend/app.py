from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from urllib.parse import urlparse
import socket

# Initialize Flask app
app = Flask(__name__)

# Path to the trained model
model_file = 'phishing_model.pkl'

# Feature extraction for URLs
def extract_features(url):
    features = []
    
    # Extract domain and host
    domain = urlparse(url).netloc
    features.append(get_ip_from_domain(domain))
    
    # Count number of dots in domain
    features.append(domain.count('.'))
    
    # Count the number of characters in the domain
    features.append(len(domain))
    
    # Check if URL uses HTTP or HTTPS
    features.append(1 if url.startswith('https') else 0)
    
    # Check for number of slashes in the URL path
    features.append(url.count('/'))
    
    # Check for suspicious keywords
    suspicious_keywords = ['login', 'secure', 'bank', 'paypal', 'account']
    features.append(int(any(keyword in url for keyword in suspicious_keywords)))
    
    return features

# Get IP address from domain
def get_ip_from_domain(domain):
    try:
        ip_address = socket.gethostbyname(domain)
        return ip_address
    except socket.gaierror:
        return "0.0.0.0"

# Train model if not already trained
def train_model():
    # Example dataset
    data = {
        'url': ['http://example.com', 'https://bank.com', 'http://login.bank.com', 'https://paypal.com'],
        'label': [0, 1, 1, 0]  # 0 = Safe, 1 = Phishing
    }
    
    df = pd.DataFrame(data)
    
    # Extract features and labels
    X = [extract_features(url) for url in df['url']]  # Now it returns a list, not a dict
    y = df['label']
    
    # Convert to numpy arrays
    X = np.array(X)
    
    # Scaling the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Create and train the RandomForest model
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(clf, model_file)
    print("Model trained and saved successfully!")
    
    return clf

# Load trained model
def load_model():
    try:
        model = joblib.load(model_file)
    except:
        model = train_model()  # Train model if not already loaded
    return model

# Predict phishing
@app.route('/predict')
def predict():
    url = request.args.get('url')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Extract features for the input URL
    features = np.array([extract_features(url)])
    
    # Load model
    model = load_model()
    
    # Predict whether the URL is phishing or not
    prediction = model.predict(features)
    
    # Return result
    is_phishing = prediction[0] == 1
    return jsonify({'is_phishing': is_phishing})

if __name__ == '__main__':
    app.run(debug=True)
