"""
🔍 DISEASE DETECTION DEMONSTRATION
Shows how YOLOv8 segmentation detects disease REGIONS (not just boxes)

Run this after training to see how the model works
"""

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
# Exported ONNX model
MODEL_PATH = 'runs/segment/strawberry_disease_model/weights/best.onnx'
# Update this
TEST_IMAGE = r'C:\Users\hasit\OneDrive\Desktop\model\datasets\leaf-spots-Neopestalotiopsis.jpg'
CONFIDENCE = 0.25


def visualize_disease_detection(model_path, image_path):
    """
    Demonstrates how the model detects diseases:
    1. Bounding Box: Where the disease is located
    2. Segmentation Mask: Exact disease region (pixel-level)
    """

    # Load model
    print(f"📂 Loading model: {model_path}")
    model = YOLO(model_path)

    # Load image
    print(f"📷 Loading image: {image_path}")
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Predict
    print(f"🔍 Detecting diseases...")
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=CONFIDENCE,
        save=False,
        verbose=True
    )

    result = results[0]

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # 1. Original Image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('1. Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # 2. Bounding Boxes Only
    img_boxes = img_rgb.copy()
    if result.boxes is not None:
        boxes = result.boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            conf = box[4]
            cls_id = int(box[5])

            # Draw rectangle
            cv2.rectangle(img_boxes, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Add label
            label = f'{model.names[cls_id]} {conf:.2f}'
            cv2.putText(img_boxes, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    axes[0, 1].imshow(img_boxes)
    axes[0, 1].set_title('2. Bounding Boxes (Where disease is)',
                         fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # 3. Segmentation Masks Only
    mask_overlay = np.zeros_like(img_rgb)
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()

        colors = [
            (255, 0, 0),      # Red - Anthracnose
            (0, 255, 0),      # Green - Healthy
            (0, 0, 255),      # Blue - Leaf Spot
            (255, 255, 0),    # Yellow - Powdery Mildew
            (255, 0, 255),    # Magenta - Cyclamen Mite
        ]

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            cls_id = int(box[5])

            # Resize mask to image size
            mask_resized = cv2.resize(
                mask, (img_rgb.shape[1], img_rgb.shape[0]))
            mask_bool = mask_resized > 0.5

            # Color the mask
            color = colors[cls_id % len(colors)]
            mask_overlay[mask_bool] = color

    axes[1, 0].imshow(mask_overlay)
    axes[1, 0].set_title('3. Segmentation Masks (Exact disease area)',
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # 4. Combined: Original + Masks
    img_combined = img_rgb.copy()
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255)
        ]

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            cls_id = int(box[5])
            conf = float(box[4])
            disease_name = model.names[cls_id]

            # Resize mask
            mask_resized = cv2.resize(
                mask, (img_rgb.shape[1], img_rgb.shape[0]))
            mask_bool = mask_resized > 0.5

            # Semi-transparent overlay
            color = colors[cls_id % len(colors)]
            overlay = img_combined.copy()
            overlay[mask_bool] = color
            img_combined = cv2.addWeighted(img_combined, 0.6, overlay, 0.4, 0)

            # Draw contour
            contours, _ = cv2.findContours(
                mask_bool.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(img_combined, contours, -1, color, 3)

            # Add label
            x1, y1 = int(box[0]), int(box[1])
            label = f'{disease_name} {conf:.2f}'

            # Text background
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(img_combined, (x1, y1-text_h-15),
                          (x1+text_w, y1), color, -1)

            # Text
            cv2.putText(img_combined, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    axes[1, 1].imshow(img_combined)
    axes[1, 1].set_title('4. Final Result (Image + Disease Regions)',
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    output_path = 'disease_detection_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_path}")
    plt.show()

    # Print detection details
    print(f"\n{'='*60}")
    print("DETECTION DETAILS")
    print("=" * 60)

    if result.boxes is not None:
        boxes = result.boxes.data.cpu().numpy()
        for i, box in enumerate(boxes):
            cls_id = int(box[5])
            conf = float(box[4])
            disease_name = model.names[cls_id]

            print(f"\nDisease {i+1}:")
            print(f"  Name: {disease_name}")
            print(f"  Confidence: {conf:.2%}")
            print(
                f"  Location: ({int(box[0])}, {int(box[1])}) to ({int(box[2])}, {int(box[3])})")

            if result.masks is not None:
                mask = result.masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(
                    mask, (img_rgb.shape[1], img_rgb.shape[0]))
                mask_area = (mask_resized > 0.5).sum()
                total_area = img_rgb.shape[0] * img_rgb.shape[1]
                print(
                    f"  Affected Area: {mask_area / total_area * 100:.2f}% of image")
    else:
        print("No diseases detected!")

    print("=" * 60)

    return results


def compare_box_vs_segmentation(model_path, image_path):
    """
    Shows the difference between:
    - Bounding Box Detection (rectangular box around disease)
    - Segmentation Detection (exact disease shape/region)
    """

    model = YOLO(model_path)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model.predict(source=image_path, imgsz=640,
                            conf=CONFIDENCE, save=False)
    result = results[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original', fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # Box detection only
    img_box = img_rgb.copy()
    if result.boxes is not None:
        for box in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img_box, (x1, y1), (x2, y2), (255, 0, 0), 4)

    axes[1].imshow(img_box)
    axes[1].set_title('Box Detection\n(Rectangular area)',
                      fontsize=16, fontweight='bold')
    axes[1].axis('off')

    # Segmentation
    img_seg = img_rgb.copy()
    if result.masks is not None:
        for i, mask in enumerate(result.masks.data.cpu().numpy()):
            mask_resized = cv2.resize(
                mask, (img_rgb.shape[1], img_rgb.shape[0]))
            mask_bool = mask_resized > 0.5

            overlay = img_seg.copy()
            overlay[mask_bool] = (255, 0, 0)
            img_seg = cv2.addWeighted(img_seg, 0.5, overlay, 0.5, 0)

            contours, _ = cv2.findContours(
                mask_bool.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(img_seg, contours, -1, (255, 255, 0), 3)

    axes[2].imshow(img_seg)
    axes[2].set_title('Segmentation Detection\n(Exact disease shape)',
                      fontsize=16, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('box_vs_segmentation.png', dpi=150)
    print(f"✅ Comparison saved: box_vs_segmentation.png")
    plt.show()


if __name__ == '__main__':
    print("="*60)
    print("🍓 DISEASE DETECTION DEMONSTRATION")
    print("="*60)

    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"\n❌ Model not found: {MODEL_PATH}")
        print("Please train the model first using train_model.py")
        exit(1)

    # Check if test image exists
    if not Path(TEST_IMAGE).exists():
        # Try to find a test image
        test_images = list(Path('dataset/test/images').glob('*.jpg'))
        if test_images:
            TEST_IMAGE = str(test_images[0])
            print(f"\n📷 Using test image: {TEST_IMAGE}")
        else:
            print(f"\n❌ Test image not found: {TEST_IMAGE}")
            print("Please update TEST_IMAGE in the script")
            exit(1)

    print(f"\n🔍 HOW THE MODEL WORKS:")
    print(f"   1. Takes a strawberry leaf image")
    print(f"   2. Identifies diseased regions pixel-by-pixel")
    print(f"   3. Creates a mask showing EXACT disease area")
    print(f"   4. NOT just a box - shows the actual infected region!")
    print()

    # Run demonstration
    visualize_disease_detection(MODEL_PATH, TEST_IMAGE)

    print(f"\n📊 Comparison: Box vs Segmentation")
    compare_box_vs_segmentation(MODEL_PATH, TEST_IMAGE)

    print(f"\n✅ Demonstration complete!")
    print(f"   Check 'disease_detection_demo.png' to see how it works")
