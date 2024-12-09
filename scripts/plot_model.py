import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

# File paths for model and history
model_save_path = 'models/cnn_lstm_model.h5'
history_save_path = 'models/cnn_lstm_history.npy'

# File paths for test data
X_test_file = 'data/processed/X_test.npy'
y_test_file = 'data/processed/y_test.npy'

# Load the trained model and history
model = load_model(model_save_path)
history = np.load(history_save_path, allow_pickle=True).item()  # Load the training history

# Plot training and validation accuracy/loss
epochs_trained = len(history['accuracy'])  # Number of epochs trained

# Create a figure with two subplots
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

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

# Load the test data
X_test = np.load(X_test_file)
y_test = np.load(y_test_file)

# Reshape data to add a channel dimension for CNN
X_test = X_test[..., np.newaxis]

# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

test_accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Test Accuracy: {test_accuracy:.2%}")

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Define the genre labels
genre_labels = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 
                'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=genre_labels, yticklabels=genre_labels, cbar=True)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Genre')
plt.ylabel('True Genre')
plt.xticks(rotation=45)
plt.show()
