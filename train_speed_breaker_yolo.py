
import argparse
from pathlib import Path

from ultralytics import YOLO


def train(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 150,
    img_size: int = 640,
    batch_size: int = 16,
    run_name: str = "speed_breaker_model",
):
    """
    Fine-tune YOLOv8 on the speed breaker dataset.

    base_model options (accuracy vs. speed tradeoff):
        yolov8n.pt  - nano,  fastest, best for Raspberry Pi / embedded
        yolov8s.pt  - small, good balance for Jetson Nano / mid hardware
        yolov8m.pt  - medium, most accurate, needs a real GPU for real-time
    """
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml}. See the module docstring "
            f"for the required dataset folder structure."
        )

    model = YOLO(base_model)  # loads COCO-pretrained weights

    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=run_name,
        patience=25,          # early stop if val mAP plateaus for 25 epochs
        # --- augmentation: critical for real-road robustness ---
        hsv_h=0.015,           # slight hue jitter (paint fade/lighting)
        hsv_s=0.6,             # saturation jitter (weather/glare)
        hsv_v=0.4,             # brightness jitter (day/dusk/night)
        degrees=5.0,           # small rotation (camera tilt/road slope)
        translate=0.1,
        scale=0.4,             # scale jitter -> robustness to distance
        fliplr=0.5,            # horizontal flip is valid for road scenes
        flipud=0.0,            # NEVER flip vertically -- road orientation matters
        mosaic=1.0,            # combines 4 images -> better small-object learning
        mixup=0.1,
        copy_paste=0.1,
        # --- optimizer ---
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        # --- misc ---
        val=True,
        plots=True,
        save=True,
        device="mps",               # GPU 0; set to "cpu" if no GPU available
    )

    # Run validation explicitly and print key accuracy metrics
    metrics = model.val()
    print("\n=== Validation results ===")
    print(f"mAP50:     {metrics.box.map50:.3f}  (detection accuracy at IoU 0.5)")
    print(f"mAP50-95:  {metrics.box.map:.3f}  (stricter, averaged IoU)")
    print(f"Precision: {metrics.box.p[0]:.3f}")
    print(f"Recall:    {metrics.box.r[0]:.3f}")
    print(
        "\nRule of thumb: mAP50 > 0.85 and Recall > 0.85 is a solid, "
        "field-usable model. Recall matters most here -- a missed bump "
        "(false negative) is worse for driver safety than an extra false "
        "alarm."
    )
    print(f"\nBest weights saved to: runs/detect/{run_name}/weights/best.pt")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train speed breaker YOLOv8 detector")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--base-model", default="yolov8n.pt",
                         help="yolov8n.pt (fast/embedded) | yolov8s.pt | yolov8m.pt (accurate)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--name", default="speed_breaker_model")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        run_name=args.name,
    )
