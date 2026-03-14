import pennylane as qml
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
from tensorflow.keras import metrics

try:
    from utils.config import (
        CLASSICAL_LAYERS, DROPOUT_RATES, N_QUBITS, N_QLAYERS,
        ENCODING_TYPE, LEARNING_RATE, FOCAL_LOSS_GAMMA, FOCAL_LOSS_ALPHA
    )
except ImportError:
    # Fallback values if config not available
    N_QUBITS = 2
    N_QLAYERS = 4
    LEARNING_RATE = 0.001
    FOCAL_LOSS_GAMMA = 1.0
    FOCAL_LOSS_ALPHA = 0.3

# Define quantum device
dev = qml.device("default.qubit", wires=N_QUBITS, seed=42)

# Define quantum node
@qml.qnode(dev)
def qnode(inputs, weights):
    # Encode inputs into qubits
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    # Trainable quantum layers
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    # Return expectation values
    return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]

# Parameter shapes
weight_shapes = {"weights": (N_QLAYERS, N_QUBITS)}

class HybridQNN:
    def __init__(self, input_dim, learning_rate=LEARNING_RATE):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = None
        self.build_model()
        
    def build_model(self):
        """Build the hybrid quantum-classical model"""
        self.model = Sequential([
            InputLayer(input_shape=(self.input_dim,)),
            Dense(256, activation="relu", kernel_initializer="glorot_uniform"),
            Dropout(0.3),
            Dense(128, activation="relu", kernel_initializer="glorot_uniform"),
            Dropout(0.3),
            Dense(N_QUBITS),
            qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=N_QUBITS),
            Dense(1, activation="sigmoid")
        ])
        return self
    
    def compile_model(self, loss=None):
        """Compile the model with specified loss"""
        if loss is None:
            loss = losses.BinaryFocalCrossentropy(
                gamma=FOCAL_LOSS_GAMMA, 
                alpha=FOCAL_LOSS_ALPHA
            )
        
        self.model.compile(
            loss=loss, 
            optimizer=Adam(learning_rate=self.learning_rate), 
            metrics=["accuracy",metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc') ]
        )
        return self
    
    def get_callbacks(self, fold=None, patience=10):
        """Get training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        if fold is not None:
            os.makedirs('models/saved', exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    f'models/saved/fold_{fold}.h5', 
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        return callbacks
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet")
        
    def save(self, path):
        """Save model to path"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")

# Simple function version (if you don't want to use the class)
def build_simple_model(input_dim):
    """Simplified function to build model without callbacks"""
    model = Sequential([
        InputLayer(input_shape=(input_dim,)),
        Dense(256, activation="relu", kernel_initializer="glorot_uniform"),
        Dropout(0.3),
        Dense(128, activation="relu", kernel_initializer="glorot_uniform"),
        Dropout(0.3),
        Dense(N_QUBITS),
        qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=N_QUBITS),
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        loss=losses.BinaryFocalCrossentropy(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA),
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"]
    )
    
    return model

if __name__ == "__main__":
    # Using the class
    model = HybridQuantumModel(input_dim=10)
    model.compile_model()
    model.summary()
    
    # Get callbacks for fold 1
    callbacks = model.get_callbacks(fold=1)
    
    simple_model = build_simple_model(input_dim=10)
    simple_model.summary()