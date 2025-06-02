from ultralytics import YOLO
import torch
import cv2

def main():
    # Print device info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load trained model
    model = YOLO("runs/train/exp/weights/best.pt")  # adjust path if needed

    # Choose input source: image, folder, or video
    source = "mangueira2.png"  # replace with your image, folder, or video file path

    # Run inference
    results = model.predict(source=source, device=device, save=True)

    # Print summary for the first result
    for r in results:
        print(f"\nDetected {len(r.boxes)} objects")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"Class ID: {cls_id}, Confidence: {conf:.2f}")

if __name__ == "__main__":
    main()
