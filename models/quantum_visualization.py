import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.config import PLOTS_DIR
from models.hybrid_qnn import qnode, N_QUBITS, N_QLAYERS


class QuantumVisualizer:
    """Visualization tool for the quantum circuit"""

    def __init__(self):
        pass

    def draw_circuit(self, sample_inputs=None, sample_weights=None):
        """Draw and save the quantum circuit"""

        #  inputs
        if sample_inputs is None:
            sample_inputs = np.random.randn(N_QUBITS)

        #  trainable weights
        if sample_weights is None:
            sample_weights = np.random.randn(
                N_QLAYERS,
                N_QUBITS
            )

        # Draw PennyLane circuit
        fig, ax = qml.draw_mpl(qnode)(
            sample_inputs,
            sample_weights
        )

        ax.set_title(
            f"Variational Quantum Circuit Employed in the Hybrid QNN",
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout()

        save_path = os.path.join(
            PLOTS_DIR,
            "quantum_circuit.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        print(f"Quantum circuit saved to: {save_path}")

        return fig