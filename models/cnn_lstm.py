from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as tf_layers

def cnn_lstm(input_shape, output_shape):
    """
    Defines a deep learning architecture with 3 convolutional layers and two 
    Long-Short-Term-Memory layers. 
    Convolutional layers are for efficient feature extraction. LSTM layers to capture
    temporal relationships within data. Convolutional layers use ReLU activation function,
    followed by MaxPooling layer, Dropout layer, and BatchNormalization layer. The model 
    is also defined with low dropout to avoid overfitting in recurrent layers. \n
    Args:
        input_shape (tuple): Shape of the input data
        output_shape (int): Number of different possible classifications
    
    Returns:
        model: Compiled CNN-LSTM model
    """
    model = Sequential([
        # 32 filters, each filter has size 3x3, use relu activation, variable input tensor shape
        tf_layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        # reduce dimension of data by taking max of each 2x2 block in tensor
        tf_layers.MaxPooling2D(pool_size=(2, 2)),
        # normalize activations across batch, output shape still same
        tf_layers.BatchNormalization(),
        # drop 10% of neurons to prevent overfitting
        tf_layers.Dropout(0.1),

        # incrementally increase filters to extract more complex features
        tf_layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        # keep decreasing dimensions
        tf_layers.MaxPooling2D(pool_size=(2, 2)),
        tf_layers.BatchNormalization(),
        tf_layers.Dropout(0.1),

        tf_layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        tf_layers.MaxPooling2D(pool_size=(2, 2)),
        tf_layers.BatchNormalization(),
        tf_layers.Dropout(0.1),

        # flatten 3D tensor to LSTM-compatible shape
        tf_layers.Reshape((-1, 64)),  # Dynamically flatten spatial dimensions into sequence length

        # LSTM layers
        tf_layers.LSTM(30, return_sequences=True),
        tf_layers.LSTM(30),
        
        # fully connected dense layer, softmax to get probabilities for each classification
        tf_layers.Dense(output_shape, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
