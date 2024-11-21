import librosa
import numpy as np
import os
import pandas as pd

# Function to extract audio features
def extract_features(file_path):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=None)

        # Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        zero_crossings = librosa.feature.zero_crossing_rate(y=audio)

        # Aggregate features (mean and standard deviation)
        feature_vector = np.hstack([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),  # MFCCs
            np.mean(spectral_centroid), np.std(spectral_centroid),  # Spectral Centroid
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),  # Spectral Bandwidth
            np.mean(rolloff), np.std(rolloff),  # Spectral Rolloff
            np.mean(zero_crossings), np.std(zero_crossings)  # Zero Crossing Rate
        ])
        return feature_vector
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to process a dataset

'''
def process_dataset(directory):
    data = []
    for category in ['affected', 'healthy']:
        folder = os.path.join(directory, category)
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        for file_name in os.listdir(folder):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder, file_name)
                features = extract_features(file_path)
                if features is not None:
                    label = 1 if category == 'affected' else 0
                    data.append([file_name, *features, label])
    
    # Convert to DataFrame
    columns = ['file_name'] + [f'feature_{i}' for i in range(len(data[0]) - 2)] + ['label']
    return pd.DataFrame(data, columns=columns)'''
'''
import librosa

def check_wav_files(directory):
    for category in ['affected', 'healthy']:
        folder = os.path.join(directory, category)
        for file_name in os.listdir(folder):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder, file_name)
                try:
                    audio, sr = librosa.load(file_path, sr=None)
                    print(f"Successfully loaded {file_name}")
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")

# Run this check
check_wav_files(r'D:\CLGProjects\ccp\voicepark\backend\dataset')
'''

def process_dataset(directory):
    data = []
    for category in ['affected', 'healthy']:
        folder = os.path.join(directory, category)
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue  # Skip if folder doesn't exist

        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        print(f"Found {len(files)} files in {folder}")  # Debug print
        if not files:
            print(f"No WAV files found in {folder}")
            continue  # Skip if no files in folder

        for file_name in files:
            file_path = os.path.join(folder, file_name)
            print(f"Processing file: {file_path}")  # Debug print
            features = extract_features(file_path)
            if features is not None:
                label = 1 if category == 'affected' else 0
                data.append([file_name, *features, label])

    if not data:
        print("No data was processed. Please ensure the dataset is properly set up.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Convert to DataFrame
    columns = ['file_name'] + [f'feature_{i}' for i in range(len(data[0]) - 2)] + ['label']
    return pd.DataFrame(data, columns=columns)


# Main script
if __name__ == '__main__':
    dataset_directory = 'voicepark/backend/dataset'
    output_csv = 'voicepark/backend/dataset/features.csv'
  # Replace with your dataset path
      # Path to save extracted features

    # Process dataset and save features
    print("Extracting features...")
    feature_data = process_dataset(dataset_directory)
    feature_data.to_csv(output_csv, index=False)
    print(f"Feature extraction completed. Saved to {output_csv}")
