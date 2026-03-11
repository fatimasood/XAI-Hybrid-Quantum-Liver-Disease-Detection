import pennylane as qml
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from utils.config import (
    CLASSICAL_LAYERS, DROPOUT_RATES, N_QUBITS, N_QLAYERS,
    ENCODING_TYPE, LEARNING_RATE, FOCAL_LOSS_GAMMA, FOCAL_LOSS_ALPHA
)

class QuantumLayer:
    """Enhanced quantum layer with multiple encoding strategies"""
    
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_QLAYERS, encoding_type=ENCODING_TYPE):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_type = encoding_type
        # Use a device compatible with the interface
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weight_shapes = {"weights": (n_layers, n_qubits)}
        
    def _create_qnode(self):
        """Create quantum node with specified encoding"""
        @qml.qnode(self.dev, interface='tf')
        def quantum_circuit(inputs, weights):
            # inputs is now a single sample (1D) thanks to vectorized_map
            circuit_inputs = tf.cast(inputs, tf.float32)

            # Data encoding based on type
            if self.encoding_type == 'angle':
                qml.AngleEmbedding(circuit_inputs, wires=range(self.n_qubits), rotation='Y')
            elif self.encoding_type == 'amplitude':
                n_states = 2 ** self.n_qubits
                # Simple padding logic for amplitude
                padded = tf.pad(circuit_inputs, [[0, max(0, n_states - tf.shape(circuit_inputs)[0])]])
                qml.AmplitudeEmbedding(features=padded[:n_states], wires=range(self.n_qubits), normalize=True)
            else:
                binary_inputs = tf.cast(circuit_inputs > 0.5, tf.int32)
                qml.BasisEmbedding(binary_inputs, wires=range(self.n_qubits))
            
            # Variational layers
            for layer in range(self.n_layers):
                qml.BasicEntanglerLayers(weights[layer:layer+1], wires=range(self.n_qubits))
                for wire in range(self.n_qubits):
                    qml.RY(weights[layer, wire], wires=wire)
                    qml.RZ(weights[layer, wire] * 0.5, wires=wire)
            
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]
        
        return quantum_circuit
    
    def get_layer(self):
        """Get the stable custom Keras layer"""
        return self._create_custom_keras_layer()
    
    def _create_custom_keras_layer(self):
        qnode_fn = self._create_qnode()
        n_qubits = self.n_qubits
        weight_shape = self.weight_shapes["weights"]

        class QuantumKerasLayer(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.q_weights = self.add_weight(
                    name='quantum_weights',
                    shape=weight_shape,
                    initializer='random_normal',
                    trainable=True
                )

            @tf.autograph.experimental.do_not_convert
            def call(self, inputs, training=None):
                # Ensure inputs are float64 for high-precision quantum simulation if needed
                # but PennyLane usually handles the cast. Let's ensure batch processing.
                inputs = tf.cast(inputs, tf.float32)

                # vectorized_map is the "Golden Way" to process batches in QML
                # It avoids Python loops that cause 'InaccessibleTensorError'
                def loop_fn(sample):
                    return qnode_fn(sample, self.q_weights)

                results = tf.vectorized_map(loop_fn, inputs)
                
                # Convert back to float32 for classical layers
                output = tf.cast(results, tf.float32)
                # Reshape to ensure (batch, n_qubits)
                return tf.reshape(output, [-1, n_qubits])

            def compute_output_shape(self, input_shape):
                return (input_shape[0], n_qubits)
        
        return QuantumKerasLayer()

class HybridQNN:
    """Hybrid Quantum-Classical Neural Network"""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.quantum_layer = QuantumLayer()
        self.model = None
        
    def build_model(self):
        """Build the hybrid model architecture"""
        model = Sequential([
            InputLayer(input_shape=(self.input_dim,)),
            
            # Classical feature extraction
            Dense(CLASSICAL_LAYERS[0], activation="relu", kernel_initializer="he_normal"),
            Dropout(DROPOUT_RATES[0]),
            Dense(CLASSICAL_LAYERS[1], activation="relu", kernel_initializer="he_normal"),
            Dropout(DROPOUT_RATES[1]),
            Dense(CLASSICAL_LAYERS[2], activation="relu", kernel_initializer="he_normal"),
            
            # Bottleneck: Compress to match number of qubits
            Dense(self.quantum_layer.n_qubits, activation='tanh', name='pre_quantum'),
            
            # Quantum layer
            self.quantum_layer.get_layer(),
            
            # Output processing
            Dense(CLASSICAL_LAYERS[2] // 2, activation="relu"),
            Dropout(DROPOUT_RATES[2]),
            Dense(1, activation="sigmoid", name='output')
        ])
        
        self.model = model
        return self
    
    def compile_model(self):
        """Compile the model with focal loss"""
        self.model.compile(
            loss=losses.BinaryFocalCrossentropy(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA),
            optimizer=Adam(learning_rate=LEARNING_RATE),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        return self
    
    def get_callbacks(self, fold=None):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        if fold is not None:
            import os
            os.makedirs('models/saved', exist_ok=True)
            callbacks.append(
                ModelCheckpoint(f'models/saved/fold_{fold}.h5', save_best_only=True)
            )
        return callbacks
    
    def summary(self):
        if self.model:
            self.model.summary()
        
    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)