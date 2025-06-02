from ultralytics import YOLO
import torch
import os

def main():
    # Debug CUDA
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))

    # Optional: Avoid memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Load model
    model = YOLO('yolov8n.pt')  # smallest model

    # Train with extremely light config
    model.train(
        data='data.yaml',
        epochs=50,
        imgsz=256,        # small image size
        batch=2,          # small batch
        device='cpu',    # force GPU
        workers=0,        # lower RAM/thread usage
        project='runs/train',
        name='exp_hose',
        exist_ok=True
    )

if __name__ == '__main__':
    main()
