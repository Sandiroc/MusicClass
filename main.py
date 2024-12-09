import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from scripts import train_cnn_lstm

# Define file paths
MODEL_PATH = 'models/cnn_lstm_model.h5'
HISTORY_PATH = 'models/cnn_lstm_history.npy'
OUTPUT_DIR = 'output_figures'
DATA_DIR = 'data/processed'
GENRE_LABELS = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 
                'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to plot training and validation metrics
def plot_metrics(history, save_figures=False):
    """
    Plot accuracy and loss vs epochs.
    """
    epochs_trained = len(history['accuracy'])

    # Create subplots for accuracy and loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy vs Epochs
    ax1.plot(range(epochs_trained), history['accuracy'], label='Train Accuracy', color='blue')
    ax1.plot(range(epochs_trained), history['val_accuracy'], label='Validation Accuracy', color='orange')
    ax1.set_title('Accuracy vs Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss vs Epochs
    ax2.plot(range(epochs_trained), history['loss'], label='Train Loss', color='blue')
    ax2.plot(range(epochs_trained), history['val_loss'], label='Validation Loss', color='orange')
    ax2.set_title('Loss vs Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save or show the figure
    if save_figures:
        plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_loss.png'))
        print("Saved accuracy and loss plots to output_figures/accuracy_loss.png")
    else:
        plt.show()

# Function to generate and save/display confusion matrix
def plot_confusion_matrix(model, save_figures=False):
    """
    Plot confusion matrix for the test set.
    """
    # Load test data
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    # Add channel dimension for CNN input
    X_test = X_test[..., np.newaxis]

    # Predict on the test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Generate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=GENRE_LABELS, yticklabels=GENRE_LABELS, cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Genre')
    plt.ylabel('True Genre')
    plt.xticks(rotation=45)

    # Save or show the figure
    if save_figures:
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
        print("Saved confusion matrix to output_figures/confusion_matrix.png")
    else:
        plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train, plot, and display CNN-LSTM results for music genre classification.")
    parser.add_argument('--retrain', action='store_true', help="Retrain the model from scratch.")
    parser.add_argument('--save', action='store_true', help="Save the plots instead of displaying them.")

    args = parser.parse_args()

    if args.retrain:
        print("Retraining the model...")
        # Load preprocessed data
        X_train_file = os.path.join(DATA_DIR, 'X_train.npy')
        y_train_file = os.path.join(DATA_DIR, 'y_train.npy')
        X_val_file = os.path.join(DATA_DIR, 'X_val.npy')
        y_val_file = os.path.join(DATA_DIR, 'y_val.npy')

        # Define model input and output shapes
        input_shape = (130, 23, 1)
        output_shape = 10

        # Train the model and save it
        history = train_cnn_lstm.train_cnn_lstm_model(input_shape, output_shape, X_train_file, y_train_file, X_val_file, y_val_file,
                                       epochs=250, batch_size=8, model_save_path=MODEL_PATH, history_save_path=HISTORY_PATH)
        print("Model retrained and saved.")
    else:
        print("Loading pre-trained model and history...")
        model = load_model(MODEL_PATH)
        history = np.load(HISTORY_PATH, allow_pickle=True).item()

    # Plot metrics and confusion matrix
    plot_metrics(history, save_figures=args.save)
    plot_confusion_matrix(model, save_figures=args.save)
