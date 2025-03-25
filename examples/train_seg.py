from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
#model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

#output_dir = "/workspace/codes/seg_train_v0"
output_dir = "/workspace/codes/seg_train_layout_v0"
# Train the model
#results = model.train(data="coco8-seg.yaml", project=output_dir, epochs=5, imgsz=640)
results = model.train(data="../layout/layout.yaml", project=output_dir, epochs=5, imgsz=640, batch=4)
