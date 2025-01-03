import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_model(input_dim):
    """
    Creates and returns a simple neural network model.
    
    Parameters:
        input_dim (int): Number of input features.

    Returns:
        model: A compiled Keras Sequential model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Regression task, linear output
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
