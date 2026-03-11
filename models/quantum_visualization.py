import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.config import PLOTS_DIR

class QuantumVisualizer:
    """Visualization tools for quantum circuits"""
    
    def __init__(self, quantum_layer):
        self.quantum_layer = quantum_layer
        self.dev = quantum_layer.dev
        
    def draw_circuit(self, sample_inputs=None, sample_weights=None):
        """Draw the quantum circuit"""
        
        if sample_inputs is None:
            sample_inputs = np.random.randn(self.quantum_layer.n_qubits)
        if sample_weights is None:
            sample_weights = np.random.randn(self.quantum_layer.n_layers, 
                                            self.quantum_layer.n_qubits)
        
        # Draw circuit
        fig, ax = qml.draw_mpl(self.quantum_layer.qnode)(sample_inputs, sample_weights)
        plt.title(f'Quantum Circuit - {self.quantum_layer.n_qubits} Qubits, '
                 f'{self.quantum_layer.n_layers} Layers', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'quantum_circuit.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def visualize_quantum_state(self, sample_inputs=None, sample_weights=None):
        """Visualize the quantum state after encoding"""
        
        if sample_inputs is None:
            sample_inputs = np.random.randn(self.quantum_layer.n_qubits)
        if sample_weights is None:
            sample_weights = np.random.randn(self.quantum_layer.n_layers, 
                                            self.quantum_layer.n_qubits)
        
        # Get quantum state
        @qml.qnode(self.dev)
        def state_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.quantum_layer.n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(self.quantum_layer.n_qubits))
            return qml.state()
        
        state = state_circuit(sample_inputs, sample_weights)
        
        # Plot Bloch spheres for each qubit
        fig, axes = plt.subplots(1, self.quantum_layer.n_qubits, figsize=(15, 4))
        
        for i in range(self.quantum_layer.n_qubits):
            # Calculate Bloch sphere coordinates from density matrix
            bloch_vector = [
                np.real(state @ qml.PauliX(i) @ state.conj()),
                np.real(state @ qml.PauliY(i) @ state.conj()),
                np.real(state @ qml.PauliZ(i) @ state.conj())
            ]
            
            # Create Bloch sphere
            plot_bloch_vector(bloch_vector, ax=axes[i])
            axes[i].set_title(f'Qubit {i}')
        
        plt.suptitle('Quantum State Visualization (Bloch Spheres)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'bloch_spheres.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def visualize_measurement_probabilities(self, sample_inputs=None, sample_weights=None):
        """Visualize measurement outcome probabilities"""
        
        if sample_inputs is None:
            sample_inputs = np.random.randn(self.quantum_layer.n_qubits)
        if sample_weights is None:
            sample_weights = np.random.randn(self.quantum_layer.n_layers, 
                                            self.quantum_layer.n_qubits)
        
        @qml.qnode(self.dev)
        def prob_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.quantum_layer.n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(self.quantum_layer.n_qubits))
            return qml.probs(wires=range(self.quantum_layer.n_qubits))
        
        probs = prob_circuit(sample_inputs, sample_weights)
        
        # Plot probabilities
        fig, ax = plt.subplots(figsize=(12, 6))
        
        n_states = 2 ** self.quantum_layer.n_qubits
        states = [format(i, f'0{self.quantum_layer.n_qubits}b') for i in range(n_states)]
        
        colors = plt.cm.viridis(probs / max(probs))
        bars = ax.bar(range(n_states), probs, color=colors)
        ax.set_xticks(range(n_states))
        ax.set_xticklabels(states, rotation=45)
        ax.set_xlabel('Basis State')
        ax.set_ylabel('Probability')
        ax.set_title('Measurement Outcome Probabilities', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'measurement_probs.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def plot_bloch_vector(bloch_vector, ax=None):
    """Helper function to plot Bloch sphere"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
    
    # Draw axes
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='b', arrow_length_ratio=0.1)
    
    # Draw state vector
    ax.quiver(0, 0, 0, bloch_vector[0], bloch_vector[1], bloch_vector[2], 
              color='k', linewidth=3, arrow_length_ratio=0.2)
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return ax