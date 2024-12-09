import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

# Define constants
DATA_DIR = 'data/genres_original'  # Path to the genres folder
N_MFCC = 23                      # Number of MFCC features
MAX_PAD_LENGTH = 130             # Ensure consistent input length (adjust based on dataset)
OUTPUT_DIR = 'data/processed'    # Directory to save processed data

# Function to extract MFCCs
def extract_mfccs(file_path, n_mfcc=N_MFCC, max_pad_length=MAX_PAD_LENGTH):
    """
    Extract MFCCs from a .wav file. Skip corrupted files.
    Args:
        file_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCC coefficients.
        max_pad_length (int): Length to pad/truncate the MFCCs.
    Returns:
        np.ndarray or None: Transposed MFCC feature matrix or None if file is corrupted.
    """
    try:
        # Load the audio file
        signal, sr = librosa.load(file_path, sr=22050)
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        # Pad or truncate to consistent shape
        if mfccs.shape[1] < max_pad_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_pad_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_length]
        return mfccs.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Initialize data arrays
X, y = [], []
genres = sorted(os.listdir(DATA_DIR))  # Alphabetically sorted genre names

# Process each genre folder
for label, genre in enumerate(genres):
    genre_path = os.path.join(DATA_DIR, genre)
    for file_name in os.listdir(genre_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(genre_path, file_name)
            mfccs = extract_mfccs(file_path)
            if mfccs is not None:  # Skip if file couldn't be processed
                X.append(mfccs)
                y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the MFCCs (zero mean, unit variance)
X = np.array([((mfccs - np.mean(mfccs)) / np.std(mfccs)) for mfccs in X])

# Split the data into training, validation, and test sets (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Save preprocessed data
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_val)
np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)

print("Preprocessing complete. Data saved to 'data/processed'.")