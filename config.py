# Configuration file for Solar Panel Damage Detection System
# Edit these values to customize the system behavior

# ====================
# Model Configuration
# ====================

MODEL_CONFIG = {
    # Image processing
    'IMAGE_SIZE': (224, 224),  # Input image size (height, width)
    'NORMALIZE': True,          # Normalize pixel values to 0-1
    
    # Training
    'BATCH_SIZE': 32,           # Batch size for training (reduce if out of memory)
    'EPOCHS': 50,               # Number of training epochs
    'LEARNING_RATE': 0.001,     # Initial learning rate
    'VALIDATION_SPLIT': 0.2,    # Fraction of data for validation
    
    # Model architecture
    'ARCHITECTURE': 'mobilenet', # Options: 'mobilenet', 'efficientnet'
    'DROPOUT_RATE': 0.5,         # Dropout rate for regularization
    'DENSE_UNITS': [256, 128],   # Hidden layer sizes
    
    # Model paths
    'MODEL_PATH': 'models/solar_panel_model.h5',
    'CHECKPOINT_PATH': 'models/checkpoints/',
    
    # Classes
    'CLASSES': [
        'Clean',
        'Dusty/Soiling',
        'Bird-drop',
        'Physical-Damage',
        'Electrical-Damage',
        'Snow-Covered'
    ]
}

# ====================
# Impact Analysis
# ====================

IMPACT_CONFIG = {
    # Energy loss estimates (min%, max%) for each class
    'ENERGY_LOSS': {
        'Clean': (0, 0),
        'Dusty/Soiling': (5, 15),
        'Bird-drop': (10, 20),
        'Physical-Damage': (20, 40),
        'Electrical-Damage': (25, 50),
        'Snow-Covered': (70, 100)
    },
    
    # Maintenance priority levels
    'PRIORITY_LEVELS': {
        'Clean': 'Low',
        'Dusty/Soiling': 'Low',
        'Bird-drop': 'Medium',
        'Physical-Damage': 'High',
        'Electrical-Damage': 'High',
        'Snow-Covered': 'Medium'
    },
    
    # Default panel specifications
    'DEFAULT_PANEL_CAPACITY_KW': 0.3,  # 300W panel
    'DAILY_SUN_HOURS': 5,               # Average sun hours per day
    
    # Environmental factors
    'CO2_PER_KWH': 0.5,                 # kg CO2 per kWh (varies by region)
    'TREES_PER_KG_CO2': 21,             # kg CO2 absorbed per tree per year
    
    # Financial
    'DEFAULT_ELECTRICITY_RATE': 0.12,   # USD per kWh
}

# ====================
# Data Augmentation
# ====================

AUGMENTATION_CONFIG = {
    'rotation_range': 30,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': True,
    'fill_mode': 'nearest'
}

# ====================
# Kaggle Datasets
# ====================

DATASET_CONFIG = {
    'datasets': {
        'solar_clean_faulty': 'pythonafroz/solar-panel-images-clean-and-faulty-images',
        'pv_defect': 'alicjalenarczyk/pv-panel-defect-dataset'
    },
    'download_dir': 'datasets',
    'processed_dir': 'data'
}

# ====================
# Training Callbacks
# ====================

CALLBACK_CONFIG = {
    # Early stopping
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 10,
        'restore_best_weights': True
    },
    
    # Learning rate reduction
    'reduce_lr': {
        'monitor': 'val_loss',
        'factor': 0.5,
        'patience': 5,
        'min_lr': 1e-7
    },
    
    # Model checkpoint
    'checkpoint': {
        'monitor': 'val_accuracy',
        'save_best_only': True
    }
}

# ====================
# Streamlit UI
# ====================

UI_CONFIG = {
    'page_title': 'Solar Panel Damage Detection',
    'page_icon': '☀️',
    'layout': 'wide',
    'theme': {
        'primaryColor': '#FF6B35',
        'backgroundColor': '#FFFFFF',
        'secondaryBackgroundColor': '#F0F2F6',
        'textColor': '#262730'
    }
}

# ====================
# Grad-CAM Visualization
# ====================

GRADCAM_CONFIG = {
    'alpha': 0.4,               # Overlay transparency
    'colormap': 'jet',          # Color map for heatmap
    'target_layer': None        # Auto-detect last conv layer if None
}

# ====================
# Inference
# ====================

INFERENCE_CONFIG = {
    'confidence_threshold': 0.6,  # Minimum confidence for high-priority actions
    'batch_inference': False,     # Enable batch processing
    'save_results': True,         # Save prediction results to file
    'results_dir': 'results/'
}

# ====================
# Regional Settings
# ====================

REGIONAL_CONFIG = {
    # Adjust these based on your location
    'timezone': 'UTC',
    'currency': 'USD',
    'co2_factor': 0.5,  # Varies by electricity grid mix
    'avg_sun_hours': 5   # Varies by latitude/season
}

# ====================
# Advanced Options
# ====================

ADVANCED_CONFIG = {
    # Fine-tuning
    'fine_tune_layers': 100,    # Number of layers to keep frozen
    'fine_tune_epochs': 20,
    'fine_tune_lr': 1e-5,
    
    # Ensemble
    'use_ensemble': False,       # Use multiple models for prediction
    'ensemble_models': [],       # List of model paths
    
    # Edge deployment
    'quantize': False,           # Quantize model for edge devices
    'target_size_mb': 10,        # Target model size
    
    # Logging
    'verbose': 1,                # Verbosity level (0=silent, 1=progress, 2=detailed)
    'log_file': 'logs/app.log'
}

# ====================
# Export Configuration
# ====================

def get_config():
    """Get complete configuration dictionary"""
    return {
        'model': MODEL_CONFIG,
        'impact': IMPACT_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'dataset': DATASET_CONFIG,
        'callbacks': CALLBACK_CONFIG,
        'ui': UI_CONFIG,
        'gradcam': GRADCAM_CONFIG,
        'inference': INFERENCE_CONFIG,
        'regional': REGIONAL_CONFIG,
        'advanced': ADVANCED_CONFIG
    }

# ====================
# Usage Example
# ====================
"""
# In your main script:
from config import get_config

config = get_config()
IMAGE_SIZE = config['model']['IMAGE_SIZE']
CLASSES = config['model']['CLASSES']
"""
