import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load the extracted features
df = pd.read_csv('voicepark/backend/dataset/features.csv')

# Separate features and labels
X = df.drop(['file_name', 'label'], axis=1)  # Features (drop filename and label columns)
y = df['label']  # Labels (0 for healthy, 1 for affected)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model (RandomForest in this case)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model to a file for later use
with open('voicepark/backend/parkinsons_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model training complete and saved as parkinsons_model.pkl")