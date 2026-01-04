"""
Generate Evaluation Artifacts
Creates confusion matrix and sample Grad-CAM visualizations for documentation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available")

from config import get_config

CONFIG = get_config()
MODEL_CONFIG = CONFIG['model']


def generate_confusion_matrix(model_path, test_data_dir, output_path='artifacts/confusion_matrix.png'):
    """Generate and save confusion matrix"""
    
    if not TF_AVAILABLE:
        print("❌ TensorFlow required for evaluation")
        return False
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import confusion_matrix, classification_report
    
    print("📊 Generating confusion matrix...")
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Prepare test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=MODEL_CONFIG['IMAGE_SIZE'],
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get predictions
    print("🔮 Making predictions...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        xticklabels=MODEL_CONFIG['CLASSES'],
        yticklabels=MODEL_CONFIG['CLASSES'],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Solar Panel Damage Detection', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix saved to {output_path}")
    
    # Save classification report
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=MODEL_CONFIG['CLASSES'],
        output_dict=True
    )
    
    report_path = str(output_path).replace('.png', '_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Classification report saved to {report_path}")
    
    # Print summary
    print("\n📋 Classification Report Summary:")
    print(classification_report(y_true, y_pred, target_names=MODEL_CONFIG['CLASSES']))
    
    return True


def generate_sample_gradcams(model_path, test_data_dir, num_samples=6, output_dir='artifacts/gradcam_samples'):
    """Generate sample Grad-CAM visualizations for each class"""
    
    if not TF_AVAILABLE:
        print("❌ TensorFlow required for Grad-CAM generation")
        return False
    
    try:
        import cv2
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        print("❌ OpenCV required for Grad-CAM generation")
        return False
    
    print("🎨 Generating sample Grad-CAMs...")
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Prepare test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=MODEL_CONFIG['IMAGE_SIZE'],
        batch_size=1,
        class_mode='categorical',
        shuffle=True
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Find last conv layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer = layer.name
            break
    
    if not last_conv_layer:
        print("❌ No convolutional layer found in model")
        return False
    
    print(f"📍 Using layer: {last_conv_layer}")
    
    # Generate samples for each class
    samples_per_class = {}
    processed = 0
    
    for img_batch, label_batch in test_generator:
        if processed >= num_samples * len(MODEL_CONFIG['CLASSES']):
            break
        
        # Get true label
        true_label_idx = np.argmax(label_batch[0])
        true_label = MODEL_CONFIG['CLASSES'][true_label_idx]
        
        # Skip if we have enough samples for this class
        if samples_per_class.get(true_label, 0) >= num_samples:
            continue
        
        # Make prediction
        predictions = model.predict(img_batch, verbose=0)
        pred_label_idx = np.argmax(predictions[0])
        pred_label = MODEL_CONFIG['CLASSES'][pred_label_idx]
        
        # Generate Grad-CAM
        try:
            grad_model = keras.models.Model(
                [model.inputs],
                [model.get_layer(last_conv_layer).output, model.output]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, preds = grad_model(img_batch)
                loss = preds[:, pred_label_idx]
            
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()
            
            # Convert image back to 0-255
            img = (img_batch[0] * 255).astype(np.uint8)
            
            # Resize heatmap and apply colormap
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_colored = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
            
            # Overlay
            overlaid = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img)
            axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(overlaid)
            axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            confidence = predictions[0][pred_label_idx] * 100
            fig.suptitle(
                f'True: {true_label} | Predicted: {pred_label} ({confidence:.1f}%)',
                fontsize=14,
                fontweight='bold'
            )
            
            plt.tight_layout()
            
            # Save
            sample_num = samples_per_class.get(true_label, 0) + 1
            filename = f"{true_label.lower().replace('/', '_')}_{sample_num}.png"
            plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            samples_per_class[true_label] = sample_num
            processed += 1
            
            print(f"✅ Saved: {filename}")
            
        except Exception as e:
            print(f"⚠️ Error generating Grad-CAM: {e}")
            continue
    
    print(f"\n✅ Generated {processed} Grad-CAM samples in {output_dir}")
    return True


def main():
    """Main artifact generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation artifacts")
    parser.add_argument('--model', default=MODEL_CONFIG['MODEL_PATH'], 
                       help='Path to trained model')
    parser.add_argument('--test-dir', default='data/val', 
                       help='Test data directory')
    parser.add_argument('--output-dir', default='artifacts', 
                       help='Output directory for artifacts')
    parser.add_argument('--confusion-matrix', action='store_true',
                       help='Generate confusion matrix')
    parser.add_argument('--gradcam-samples', action='store_true',
                       help='Generate Grad-CAM samples')
    parser.add_argument('--all', action='store_true',
                       help='Generate all artifacts')
    parser.add_argument('--num-samples', type=int, default=2,
                       help='Number of Grad-CAM samples per class')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("Please train the model first: python train.py --train")
        return
    
    if not Path(args.test_dir).exists():
        print(f"❌ Test directory not found: {args.test_dir}")
        return
    
    print("=" * 60)
    print("🎨 ARTIFACT GENERATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Test data: {args.test_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    if args.confusion_matrix or args.all:
        print("\n📊 Generating Confusion Matrix...")
        cm_path = Path(args.output_dir) / 'confusion_matrix.png'
        generate_confusion_matrix(args.model, args.test_dir, str(cm_path))
    
    if args.gradcam_samples or args.all:
        print("\n🎨 Generating Grad-CAM Samples...")
        gradcam_dir = Path(args.output_dir) / 'gradcam_samples'
        generate_sample_gradcams(args.model, args.test_dir, args.num_samples, str(gradcam_dir))
    
    if not (args.confusion_matrix or args.gradcam_samples or args.all):
        parser.print_help()
        print("\n💡 Example usage:")
        print("  python generate_artifacts.py --all")
        print("  python generate_artifacts.py --confusion-matrix")
        print("  python generate_artifacts.py --gradcam-samples --num-samples 3")
    
    print("\n" + "=" * 60)
    print("✅ ARTIFACT GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
