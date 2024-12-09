import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping
from models import cnn_lstm  # Assuming the model definition is in cnn_lstm.py

def train_cnn_lstm_model(input_shape, output_shape, X_train_file, y_train_file, X_val_file, y_val_file,
                         epochs=250, batch_size=32, model_save_path='models/cnn_lstm_model.h5', history_save_path='models/cnn_lstm_history.npy'):
    """
    Trains a CNN-LSTM model on the MFCC feature training data and saves the trained model and history.
    """

    # Load preprocessed data
    X_train = np.load(X_train_file)
    y_train = np.load(y_train_file)
    X_val = np.load(X_val_file)
    y_val = np.load(y_val_file)

    # Reshape data to add a channel dimension for CNN
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    # Initialize the model with the input shape
    model = cnn_lstm(input_shape, output_shape)

    # early termination if val loss is increasing to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    # Save the model and training history
    save_model(model, model_save_path)
    np.save(history_save_path, history.history)

    print(f"Training complete. Model saved to '{model_save_path}', and history saved to '{history_save_path}'")


if __name__ == "__main__":
    # Define file paths for preprocessed data
    X_train_file = 'data/processed/X_train.npy'
    y_train_file = 'data/processed/y_train.npy'
    X_val_file = 'data/processed/X_val.npy'
    y_val_file = 'data/processed/y_val.npy'

    # Define the input shape and output shape
    input_shape = (130, 23, 1)  # Hardcoded shape
    output_shape = 10           # Number of genres/classes

    train_cnn_lstm_model(input_shape, output_shape, X_train_file, y_train_file, X_val_file, y_val_file, batch_size=8)