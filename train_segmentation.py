from ultralytics import YOLO
import argparse

def train(model, data_file, output_folder, device="0", epochs=10, warmup_epochs=1, image_size=768, batch_size=16, lr0=0.01, lrf=0.01):
    # Train the model
    #results = model.train(data="coco8-seg.yaml", project=output_dir, epochs=5, imgsz=640)
    results = model.train(data=data_file, project=output_folder, device=device,
                          cos_lr=True, lr0=lr0, lrf=lrf, warmup_epochs=warmup_epochs,
                          epochs=epochs, imgsz=image_size, batch=batch_size, plots=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Segmentation')
    parser.add_argument('--data_file', default='../layout/layout.yaml', type=str, help='path to data yaml file')
    parser.add_argument('--output_folder', default='outputs', type=str, help='output folder')
    parser.add_argument('--epochs', default=10, type=int, help='Training epoch number')
    parser.add_argument('--warmup_epochs', default=1, type=int, help='warmup epoch number')
    parser.add_argument('--batch_size', default=16, type=int, help='Training batch size')
    parser.add_argument('--image_size', default=768, type=int, help='Image size')
    parser.add_argument('--lr0', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lrf', default=0.01, type=float, help='final learning rate fraction')
    parser.add_argument('--device', default="0", type=str, help='training device (e.g., 0,1,2,3)')
    parser.add_argument('--model', default="yolo11x-seg.yaml", type=str, help='model type ()')
    args = parser.parse_args()

    print ("Training model: {}".format(args.model))
    # Load a model
    model = YOLO(args.model)  # build a new model from YAML
    #model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
    #model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    train(model, args.data_file, args.output_folder, device=args.device, epochs=args.epochs, warmup_epochs=args.warmup_epochs, 
          image_size=args.image_size, batch_size=args.batch_size, lr0=args.lr0, lrf=args.lrf)
