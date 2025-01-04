from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore

def create_model(input_dim, output_dim):
    model = Sequential([
        # Input Layer
        Dense(128, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.01)),
        BatchNormalization(),  # Apply Batch Normalization after Dense layer
        Dropout(0.4),  # Dropout layer for regularization
        
        # Hidden Layer 1
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden Layer 2
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output Layer
        Dense(output_dim, activation='softmax')  # No Batch Normalization here
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'mse']
    )
    return model