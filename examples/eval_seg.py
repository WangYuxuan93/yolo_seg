from ultralytics import YOLO

#model = YOLO("yolo11n-seg.pt")  # load an official model
#model = YOLO("seg_train_v0/train/weights/best.pt")  # load a custom model
model = YOLO("seg_train_layout_v0/train4/weights/best.pt")  # load a custom model


# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps  # a list contains map50-95(B) of each category
metrics.seg.map  # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps  # a list contains map50-95(M) of each category
