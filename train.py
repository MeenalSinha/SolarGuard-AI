"""
Solar Panel Model Training Module
Handles dataset management, model training, and evaluation
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import required packages
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available. Install with: pip install tensorflow")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("⚠️  Matplotlib/Seaborn not available")

from config import get_config

# Load configuration
CONFIG = get_config()
MODEL_CONFIG = CONFIG['model']
DATASET_CONFIG = CONFIG['dataset']
CALLBACK_CONFIG = CONFIG['callbacks']
AUGMENTATION_CONFIG = CONFIG['augmentation']


class KaggleDatasetLoader:
    """Handler for downloading and processing Kaggle datasets"""
    
    def __init__(self):
        self.datasets = DATASET_CONFIG['datasets']
        self.download_dir = DATASET_CONFIG['download_dir']
        self.processed_dir = DATASET_CONFIG['processed_dir']
        
    def setup_kaggle_api(self):
        """Setup Kaggle API credentials"""
        kaggle_json_path = Path.home() / '.kaggle' / 'kaggle.json'
        
        if not kaggle_json_path.exists():
            print("⚠️  Kaggle API credentials not found!")
            print("Please follow these steps:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Click 'Create New API Token'")
            print("3. Save kaggle.json to ~/.kaggle/")
            print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
        return True
    
    def download_datasets(self):
        """Download both Kaggle datasets"""
        if not self.setup_kaggle_api():
            return False
        
        try:
            import kaggle
            output_path = Path(self.download_dir)
            output_path.mkdir(exist_ok=True)
            
            for name, dataset_id in self.datasets.items():
                print(f"\n📥 Downloading {name}...")
                dataset_path = output_path / name
                dataset_path.mkdir(exist_ok=True)
                
                kaggle.api.dataset_download_files(
                    dataset_id,
                    path=str(dataset_path),
                    unzip=True
                )
                print(f"✅ Downloaded {name}")
            
            return True
        except Exception as e:
            print(f"❌ Error downloading datasets: {e}")
            return False
    
    def organize_dataset(self, train_split=0.8):
        """Organize downloaded datasets into training structure"""
        output_path = Path(self.processed_dir)
        train_path = output_path / 'train'
        val_path = output_path / 'val'
        
        # Create directory structure
        for split_path in [train_path, val_path]:
            for class_name in MODEL_CONFIG['CLASSES']:
                (split_path / class_name).mkdir(parents=True, exist_ok=True)
        
        print("✅ Dataset structure created")
        print("\nℹ️  Please manually organize your downloaded images into:")
        print(f"  - {train_path}/<class_name>/")
        print(f"  - {val_path}/<class_name>/")
        print("\nOr implement automatic organization based on your dataset structure.")
        
        return str(output_path)
    
    def get_dataset_statistics(self):
        """Get statistics about the organized dataset"""
        stats = {
            'train': {},
            'val': {}
        }
        
        data_path = Path(self.processed_dir)
        
        for split in ['train', 'val']:
            split_path = data_path / split
            if not split_path.exists():
                continue
                
            for class_name in MODEL_CONFIG['CLASSES']:
                class_path = split_path / class_name
                if class_path.exists():
                    count = len(list(class_path.glob('*.jpg'))) + \
                           len(list(class_path.glob('*.jpeg'))) + \
                           len(list(class_path.glob('*.png')))
                    stats[split][class_name] = count
        
        return stats


class SolarPanelTrainer:
    """Neural network model training for solar panel damage classification"""
    
    def __init__(self, num_classes=None):
        self.num_classes = num_classes or len(MODEL_CONFIG['CLASSES'])
        self.model = None
        self.history = None
        self.image_size = MODEL_CONFIG['IMAGE_SIZE']
        self.batch_size = MODEL_CONFIG['BATCH_SIZE']
        
    def build_model(self, architecture='mobilenet'):
        """Build the CNN model with transfer learning"""
        
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required for training. Install with: pip install tensorflow")
        
        print(f"🏗️  Building model with {architecture} architecture...")
        
        if architecture == 'mobilenet':
            base_model = MobileNetV2(
                input_shape=(*self.image_size, 3),
                include_top=False,
                weights='imagenet'
            )
        elif architecture == 'efficientnet':
            base_model = EfficientNetB0(
                input_shape=(*self.image_size, 3),
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Choose 'mobilenet' or 'efficientnet'")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build classification head
        dense_units = MODEL_CONFIG['DENSE_UNITS']
        dropout_rate = MODEL_CONFIG['DROPOUT_RATE']
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(dense_units[0], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate * 0.6),
            layers.Dense(dense_units[1], activation='relu'),
            layers.Dropout(dropout_rate * 0.4),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=MODEL_CONFIG['LEARNING_RATE']),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
        )
        
        self.model = model
        print(f"✅ Model built successfully!")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return model
    
    def get_data_generators(self, train_data_dir, val_data_dir):
        """Create data generators for training and validation"""
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            **AUGMENTATION_CONFIG
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train(self, train_data_dir, val_data_dir, epochs=None):
        """Train the model on the dataset"""
        if self.model is None:
            print("⚠️  Model not built. Building with default architecture...")
            self.build_model()
        
        epochs = epochs or MODEL_CONFIG['EPOCHS']
        
        print(f"\n🚀 Starting training for {epochs} epochs...")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Image size: {self.image_size}")
        
        # Get data generators
        train_generator, val_generator = self.get_data_generators(train_data_dir, val_data_dir)
        
        print(f"\n📊 Dataset information:")
        print(f"   Training samples: {train_generator.samples}")
        print(f"   Validation samples: {val_generator.samples}")
        print(f"   Classes: {train_generator.class_indices}")
        
        # Setup callbacks
        checkpoint_dir = Path(MODEL_CONFIG['CHECKPOINT_PATH'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor=CALLBACK_CONFIG['early_stopping']['monitor'],
                patience=CALLBACK_CONFIG['early_stopping']['patience'],
                restore_best_weights=CALLBACK_CONFIG['early_stopping']['restore_best_weights'],
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor=CALLBACK_CONFIG['reduce_lr']['monitor'],
                factor=CALLBACK_CONFIG['reduce_lr']['factor'],
                patience=CALLBACK_CONFIG['reduce_lr']['patience'],
                min_lr=CALLBACK_CONFIG['reduce_lr']['min_lr'],
                verbose=1
            ),
            ModelCheckpoint(
                MODEL_CONFIG['MODEL_PATH'],
                monitor=CALLBACK_CONFIG['checkpoint']['monitor'],
                save_best_only=CALLBACK_CONFIG['checkpoint']['save_best_only'],
                verbose=1
            )
        ]
        
        # Train
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        return self.history
    
    def fine_tune(self, train_data_dir, val_data_dir, epochs=20, unfreeze_layers=100):
        """Fine-tune the model by unfreezing some base layers"""
        if self.model is None:
            raise ValueError("Model must be trained before fine-tuning")
        
        print(f"\n🔧 Fine-tuning model (unfreezing last {unfreeze_layers} layers)...")
        
        # Unfreeze the base model
        self.model.layers[0].trainable = True
        
        # Freeze the first N layers
        for layer in self.model.layers[0].layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        fine_tune_lr = CONFIG['advanced']['fine_tune_lr']
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
        )
        
        print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
        # Continue training
        return self.train(train_data_dir, val_data_dir, epochs=epochs)
    
    def evaluate(self, test_data_dir):
        """Evaluate model on test set"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        print("\n📊 Evaluating model on test set...")
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        results = self.model.evaluate(test_generator, verbose=1)
        
        print("\n✅ Evaluation Results:")
        for name, value in zip(self.model.metrics_names, results):
            print(f"   {name}: {value:.4f}")
        
        return results
    
    def save_model(self, path=None):
        """Save the trained model"""
        path = path or MODEL_CONFIG['MODEL_PATH']
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(path)
        print(f"💾 Model saved to {path}")
        
        # Save training history
        if self.history:
            history_path = str(path).replace('.h5', '_history.json')
            history_dict = {k: [float(v) for v in values] for k, values in self.history.history.items()}
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
            print(f"💾 Training history saved to {history_path}")
    
    def load_model(self, path=None):
        """Load a trained model"""
        path = path or MODEL_CONFIG['MODEL_PATH']
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found at {path}")
        
        self.model = keras.models.load_model(path)
        print(f"✅ Model loaded from {path}")
        
        return self.model
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if not self.history or not PLOT_AVAILABLE:
            print("⚠️  Training history not available or matplotlib not installed")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training plots saved to {save_path}")
        else:
            plt.show()
        
        return fig


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Solar Panel Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download datasets
  python train.py --download-data
  
  # Train model with MobileNetV2
  python train.py --train --architecture mobilenet --epochs 50
  
  # Train model with EfficientNetB0
  python train.py --train --architecture efficientnet --epochs 50
  
  # Fine-tune existing model
  python train.py --fine-tune --epochs 20
  
  # Evaluate model
  python train.py --evaluate
        """
    )
    
    parser.add_argument('--download-data', action='store_true', 
                       help='Download Kaggle datasets')
    parser.add_argument('--train', action='store_true', 
                       help='Train model from scratch')
    parser.add_argument('--fine-tune', action='store_true', 
                       help='Fine-tune existing model')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Evaluate model on test set')
    parser.add_argument('--architecture', choices=['mobilenet', 'efficientnet'], 
                       default='mobilenet', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, 
                       help='Batch size (overrides config)')
    parser.add_argument('--train-dir', default='data/train', 
                       help='Training data directory')
    parser.add_argument('--val-dir', default='data/val', 
                       help='Validation data directory')
    parser.add_argument('--test-dir', default='data/test', 
                       help='Test data directory')
    
    args = parser.parse_args()
    
    if args.download_data:
        print("=" * 60)
        print("📥 DOWNLOADING KAGGLE DATASETS")
        print("=" * 60)
        loader = KaggleDatasetLoader()
        if loader.download_datasets():
            print("\n✅ Download complete!")
            loader.organize_dataset()
            
            # Show statistics
            stats = loader.get_dataset_statistics()
            if any(stats['train'].values()) or any(stats['val'].values()):
                print("\n📊 Dataset Statistics:")
                for split in ['train', 'val']:
                    if stats[split]:
                        print(f"\n{split.upper()}:")
                        for class_name, count in stats[split].items():
                            print(f"  {class_name}: {count} images")
        else:
            print("\n❌ Download failed. Check Kaggle API setup.")
            return
    
    elif args.train:
        print("=" * 60)
        print("🚀 TRAINING SOLAR PANEL DETECTION MODEL")
        print("=" * 60)
        
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available. Install with: pip install tensorflow")
            return
        
        # Check if data directories exist
        if not Path(args.train_dir).exists():
            print(f"❌ Training directory not found: {args.train_dir}")
            print("   Please download and organize datasets first with: python train.py --download-data")
            return
        
        trainer = SolarPanelTrainer()
        
        # Override batch size if specified
        if args.batch_size:
            trainer.batch_size = args.batch_size
        
        # Build and train
        trainer.build_model(architecture=args.architecture)
        trainer.train(args.train_dir, args.val_dir, epochs=args.epochs)
        
        # Save model
        trainer.save_model()
        
        # Plot training history
        trainer.plot_training_history(save_path='training_history.png')
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Model saved to: {MODEL_CONFIG['MODEL_PATH']}")
        print("Training plots saved to: training_history.png")
    
    elif args.fine_tune:
        print("=" * 60)
        print("🔧 FINE-TUNING MODEL")
        print("=" * 60)
        
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available. Install with: pip install tensorflow")
            return
        
        trainer = SolarPanelTrainer()
        trainer.load_model()
        trainer.fine_tune(args.train_dir, args.val_dir, epochs=args.epochs)
        trainer.save_model()
        
        print("\n✅ Fine-tuning complete!")
    
    elif args.evaluate:
        print("=" * 60)
        print("📊 EVALUATING MODEL")
        print("=" * 60)
        
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available. Install with: pip install tensorflow")
            return
        
        if not Path(args.test_dir).exists():
            print(f"⚠️  Test directory not found: {args.test_dir}")
            print("   Using validation directory instead")
            args.test_dir = args.val_dir
        
        trainer = SolarPanelTrainer()
        trainer.load_model()
        trainer.evaluate(args.test_dir)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
