import os
import shutil
import yaml
from pathlib import Path
import zipfile
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = r'C:\Users\hasit\OneDrive\Desktop\model\dataset\Dataset.v2i.yolov8'
EPOCHS = 150
MODEL_SIZE = 'm'
PATIENCE = 20
CONFIDENCE_THRESHOLD = 0.25

# ============================================================================
# OBJECTIVE 2: VIDEO ENHANCEMENT & FEATURE EXTRACTION
# Preprocessing algorithms to improve quality and extract disease indicators.
# ============================================================================


def extract_dataset(dataset_path):
    """Extract dataset if it's a zip file"""
    if dataset_path.endswith('.zip'):
        print(f" Extracting {dataset_path}...")
        extract_dir = dataset_path.replace('.zip', '')

        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        print(f" Extracted to: {extract_dir}")
        return extract_dir
    return dataset_path


def find_data_yaml(root_dir):
    """Find data.yaml file in directory"""
    yaml_files = list(Path(root_dir).rglob('data.yaml'))

    if not yaml_files:
        raise FileNotFoundError(f"data.yaml not found in {root_dir}")

    return str(yaml_files[0])


def update_yaml_paths(yaml_path):
    """Update data.yaml with absolute paths"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_root = Path(yaml_path).parent

    config['train'] = str(dataset_root / 'train' / 'images')
    config['val'] = str(dataset_root / 'valid' / 'images')
    config['test'] = str(dataset_root / 'test' / 'images')

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config

# ============================================================================
# OBJECTIVE 3: DISEASE SEGMENTATION
# Isolating diseased regions from healthy tissue using pixel-level masks.
# ============================================================================


def visualize_predictions(model, image_path, output_path, class_names):
    """
    Visualize predictions with SEGMENTATION MASKS (disease regions)
    NOT just bounding boxes - shows exact disease area
    """

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        save=False,
        verbose=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    result = results[0]
    img_with_masks = img_rgb.copy()

    if result.masks is not None:
        # Draw segmentation masks (disease regions)
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()

        # Create color map for different diseases
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
        ]

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            # Get disease class and confidence
            cls_id = int(box[5])
            conf = float(box[4])
            disease_name = class_names[cls_id]

            # Resize mask to image size
            mask_resized = cv2.resize(
                mask, (img_rgb.shape[1], img_rgb.shape[0]))
            mask_bool = mask_resized > 0.5

            # Apply colored overlay on disease region
            color = colors[cls_id % len(colors)]
            overlay = img_with_masks.copy()
            overlay[mask_bool] = color

            # Blend with original image (semi-transparent)
            img_with_masks = cv2.addWeighted(
                img_with_masks, 0.6, overlay, 0.4, 0)

            # Draw contour around disease region
            contours, _ = cv2.findContours(
                mask_bool.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(img_with_masks, contours, -1, color, 3)

            # Add label
            x1, y1 = int(box[0]), int(box[1])
            label = f'{disease_name} {conf:.2f}'

            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                img_with_masks,
                (x1, y1 - text_h - 10),
                (x1 + text_w, y1),
                color,
                -1
            )

            # Text
            cv2.putText(
                img_with_masks,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    axes[1].imshow(img_with_masks)
    axes[1].set_title('Disease Detection (Segmented Regions)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return len(result.masks) if result.masks is not None else 0


def create_training_callback():
    """Custom callback to monitor training and enable early stopping"""
    class TrainingMonitor:
        def __init__(self):
            self.best_fitness = 0
            self.patience_counter = 0
            self.epoch_metrics = []

        def on_train_epoch_end(self, trainer):
            """Called at the end of each epoch"""
            metrics = trainer.metrics
            epoch = trainer.epoch

            # Get current fitness (mAP)
            current_fitness = metrics.get('metrics/mAP50(B)', 0)

            # Store metrics
            self.epoch_metrics.append({
                'epoch': epoch,
                'fitness': current_fitness,
                'loss': metrics.get('train/box_loss', 0),
            })

            # Check for improvement
            if current_fitness > self.best_fitness:
                self.best_fitness = current_fitness
                self.patience_counter = 0
                print(
                    f" Epoch {epoch}: New best mAP50 = {current_fitness:.4f}")
            else:
                self.patience_counter += 1
                print(
                    f" Epoch {epoch}: No improvement ({self.patience_counter}/{PATIENCE})")

            # Early stopping
            if self.patience_counter >= PATIENCE:
                print(f"\n EARLY STOPPING triggered at epoch {epoch}")
                print(f"   Best mAP50: {self.best_fitness:.4f}")
                print(f"   No improvement for {PATIENCE} epochs")
                return True  # Stop training

            return False

    return TrainingMonitor()

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

# ============================================================================
# OBJECTIVE 4: DEEP LEARNING CLASSIFICATION
# Training and optimizing the CNN model for multi-disease accuracy.
# ============================================================================


def main():
    print("=" * 70)
    print(" STRAWBERRY DISEASE DETECTION - VS CODE TRAINING")
    print("=" * 70)

    # Check GPU
    print(f"\n Hardware Check:")
    if not torch.cuda.is_available():
        raise RuntimeError(
            " CUDA not available. Please install PyTorch with CUDA support to use your RTX3050.")

    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ========================================================================
    #  PREPARE DATASET
    # ========================================================================
    print(f"\n{'='*70}")
    print("STEP 1: DATASET PREPARATION")
    print("=" * 70)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    dataset_dir = extract_dataset(DATASET_PATH)

    yaml_path = find_data_yaml(dataset_dir)
    print(f" Found config: {yaml_path}")

    config = update_yaml_paths(yaml_path)

    print(f"\n Dataset Info:")
    print(f"   Classes: {config['nc']}")
    print(f"   Names: {config['names']}")

    for split in ['train', 'val', 'test']:
        path = Path(config[split])
        if path.exists():
            count = len(list(path.glob('*.jpg'))) + \
                len(list(path.glob('*.png')))
            print(f"   {split.capitalize()}: {count} images")

    print(f"\n{'='*70}")
    print("STEP 2: MODEL INITIALIZATION")
    print("=" * 70)

    model_name = f'yolov8{MODEL_SIZE}-seg.pt'
    print(f"\n Loading {model_name}...")
    model = YOLO(model_name)

    print(f" Model loaded successfully")
    print(f"   Task: Instance Segmentation (pixel-level disease detection)")
    print(f"   Mode: Shows disease REGIONS, not just boxes")

    print(f"\n{'='*70}")
    print("STEP 3: MODEL TRAINING")
    print("=" * 70)

    print(f"\n⚙️  Training Configuration:")
    print(f"   Max Epochs: {EPOCHS}")
    print(f"   Early Stopping Patience: {PATIENCE} epochs")
    print(f"   Batch Size: {16 if MODEL_SIZE == 'n' else 12}")
    print(f"   Image Size: 640x640")
    print(f"   Optimizer: AdamW")

    print(f"\n🚀 Starting training...")
    print(f"   This may take 30-120 minutes depending on your hardware")
    print(f"   Training will stop automatically if no improvement\n")

    # Train with early stopping
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=640,
        batch=16 if MODEL_SIZE == 'n' else (12 if MODEL_SIZE == 's' else 8),
        name='strawberry_disease_model',
        task='segment',  # SEGMENTATION - shows disease regions!

        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,

        # Augmentation (handles real-world field conditions)
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,

        # Early stopping
        patience=PATIENCE,  # Built-in early stopping

        # Performance
        cos_lr=True,
        cache=True,
        device=0,
        workers=4,

        # Visualization
        plots=True,
        save=True,
        save_period=10,
        val=True,
    )

    print(f"\n Training completed!")

    # ========================================================================
    #  VALIDATION
    # ========================================================================
    print(f"\n{'='*70}")
    print("STEP 4: MODEL VALIDATION")
    print("=" * 70)

    best_model_path = 'runs/segment/strawberry_disease_model/weights/best.pt'
    model = YOLO(best_model_path)

    print(f"\n Evaluating best model...")
    metrics = model.val(data=yaml_path, imgsz=640, plots=True)

    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"Box mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"Box mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print(
        f"Mask mAP@0.5:      {metrics.seg.map50:.4f} {'🎯' if metrics.seg.map50 >= 0.90 else '⚠️'}")
    print(f"Mask mAP@0.5:0.95: {metrics.seg.map:.4f}")
    print(f"Precision:         {metrics.box.mp:.4f}")
    print(f"Recall:            {metrics.box.mr:.4f}")

    if metrics.box.mp > 0 and metrics.box.mr > 0:
        f1 = 2 * (metrics.box.mp * metrics.box.mr) / \
            (metrics.box.mp + metrics.box.mr)
        print(f"F1 Score:          {f1:.4f}")
    print("=" * 70)

    # ========================================================================
    #  VISUALIZE PREDICTIONS
    # ========================================================================
    print(f"\n{'='*70}")
    print("STEP 5: VISUALIZATION - DISEASE REGION DETECTION")
    print("=" * 70)

    print(f"\n Creating sample predictions with segmentation masks...")
    print(f"   (Shows exact disease regions, not just bounding boxes)")

    test_images = list(Path(config['test']).glob('*.jpg'))[:5]

    vis_dir = 'visualization_results'
    os.makedirs(vis_dir, exist_ok=True)

    for i, img_path in enumerate(test_images):
        output_path = f'{vis_dir}/prediction_{i+1}.png'
        num_detections = visualize_predictions(
            model,
            str(img_path),
            output_path,
            config['names']
        )
        print(
            f"    Saved: {output_path} ({num_detections} diseases detected)")

    print(f"\n Visualizations saved to: {vis_dir}/")

    print(f"\n{'='*70}")
    print("STEP 6: EXPORT FOR MOBILE DEPLOYMENT")
    print("=" * 70)

    print(f"\n Exporting to TensorFlow Lite...")

    tflite_path = model.export(
        format='tflite',
        int8=True,
        imgsz=640,
    )

    # Copy to root
    shutil.copy(tflite_path, 'model.tflite')

    # Create labels.txt
    with open('labels.txt', 'w') as f:
        f.write('\n'.join(config['names']))

    # Get sizes
    tflite_size = os.path.getsize('model.tflite') / (1024 * 1024)
    pt_size = os.path.getsize(best_model_path) / (1024 * 1024)

    print(f"✅ Export complete!")
    print(f"   TFLite model: model.tflite ({tflite_size:.2f} MB)")
    print(f"   PyTorch model: {best_model_path} ({pt_size:.2f} MB)")
    print(f"   Labels: labels.txt")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print(" TRAINING PIPELINE COMPLETE!")
    print("=" * 70)

    print(f"\n Summary:")
    print(f"   Model: YOLOv8{MODEL_SIZE}-seg")
    print(f"   Task: Instance Segmentation")
    print(f"   Detection Method: Pixel-level disease region masks")
    print(f"   Classes: {config['names']}")
    print(f"   Mask mAP@0.5: {metrics.seg.map50:.2%}")
    print(f"   Model Size: {tflite_size:.2f} MB")

    print(f"\n Generated Files:")
    print(f"   1. model.tflite - For Flutter app")
    print(f"   2. labels.txt - Disease class names")
    print(f"   3. {best_model_path} - Best PyTorch weights")
    print(f"   4. runs/segment/strawberry_disease_model/ - Training logs")
    print(f"   5. {vis_dir}/ - Sample predictions with masks")

    print(f"\n How the Model Works:")
    print(f"   ✓ Takes strawberry leaf image as input")
    print(f"   ✓ Identifies disease regions pixel-by-pixel")
    print(f"   ✓ Draws colored masks over diseased areas")
    print(f"   ✓ Returns disease name + confidence score")
    print(f"   ✗ NOT just bounding boxes - shows exact infection area!")

    print(f"\n Flutter Integration:")
    print(f"   1. Copy model.tflite → assets/model/")
    print(f"   2. Copy labels.txt → assets/model/")
    print(f"   3. Your AIService will show disease regions with masks")

    if metrics.seg.map50 >= 0.90:
        print(f"\n EXCELLENT! Model achieved target accuracy!")
        print(f"   Ready for deployment in mobile app")
    else:
        print(f"\n Tips to improve (current: {metrics.seg.map50:.2%}):")
        print(f"   - Increase EPOCHS to 200")
        print(f"   - Try MODEL_SIZE = 's' or 'm'")
        print(f"   - Check dataset quality and balance")

    print(f"\n{'='*70}")
    print(
        f" Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user")
    except Exception as e:
        print(f"\n\n Error: {e}")
        import traceback
        traceback.print_exc()
