import os
import random
import numpy as np
import tensorflow as tf

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Indian Liver Patient Dataset (ILPD).csv")
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
XAI_DIR = os.path.join(BASE_DIR, "xai", "results")

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(XAI_DIR, exist_ok=True)

# Model hyperparameters
CLASSICAL_LAYERS = [256, 128, 64]
DROPOUT_RATES = [0.3, 0.3, 0.2]
N_QUBITS = 4  
N_QLAYERS = 4
ENCODING_TYPE = 'angle'

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25

# Random seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Feature names
FEATURE_NAMES = ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G']
CLASS_NAMES = ['No Disease', 'Liver Disease']