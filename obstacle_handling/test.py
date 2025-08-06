from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt') # yolov3-v7
print()  # Prints a blank line
for k, v in model.names.items():
    print(f"{k}: {v}")