from ultralytics import YOLO
import argparse
import logging

def setup_logging(log_file):
    # 设置日志配置
    logging.basicConfig(
        filename=log_file,  # 输出日志到文件
        level=logging.INFO,  # 设置日志级别为INFO
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
        filemode='w'  # 覆盖写模式
    )

def train(model, data_file, output_folder, device="0", epochs=10, warmup_epochs=1, image_size=768, batch_size=16, lr0=0.01, lrf=0.01, box=7.5, cls=0.5, dropout=0.2, 
          multi_scale=False, mosaic=0.3, overlap_mask=True, mask_ratio=2, cos_lr=True, weight_decay=0.0001, log_file="train_log.txt"):
    # Log the training parameters
    logging.info("Training parameters:")
    logging.info(f"data_file: {data_file}")
    logging.info(f"output_folder: {output_folder}")
    logging.info(f"device: {device}")
    logging.info(f"epochs: {epochs}")
    logging.info(f"warmup_epochs: {warmup_epochs}")
    logging.info(f"image_size: {image_size}")
    logging.info(f"batch_size: {batch_size}")
    logging.info(f"lr0: {lr0}")
    logging.info(f"lrf: {lrf}")
    logging.info(f"box: {box}")
    logging.info(f"cls: {cls}")
    logging.info(f"dropout: {dropout}")
    logging.info(f"multi_scale: {multi_scale}")
    logging.info(f"mosaic: {mosaic}")
    logging.info(f"overlap_mask: {overlap_mask}")
    logging.info(f"mask_ratio: {mask_ratio}")
    logging.info(f"cos_lr: {cos_lr}")
    logging.info(f"weight_decay: {weight_decay}")

    # Train the model
    results = model.train(
        data=data_file, 
        project=output_folder, 
        device=device,
        cos_lr=cos_lr, 
        lr0=lr0, 
        lrf=lrf, 
        warmup_epochs=warmup_epochs,
        epochs=epochs, 
        imgsz=image_size, 
        batch=batch_size, 
        plots=True, 
        box=box, 
        cls=cls, 
        dropout=dropout,  # Added dropout
        multi_scale=multi_scale,  # Added multi-scale
        mosaic=mosaic,  # Added mosaic
        overlap_mask=overlap_mask,  # Added overlap_mask
        mask_ratio=mask_ratio,  # Added mask_ratio
        weight_decay=weight_decay  # Added weight_decay
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Segmentation')
    parser.add_argument('--data_file', default='../layout/layout.yaml', type=str, help='path to data yaml file')
    parser.add_argument('--output_folder', default='outputs', type=str, help='output folder')
    parser.add_argument('--epochs', default=100, type=int, help='Training epoch number')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='warmup epoch number')
    parser.add_argument('--batch_size', default=16, type=int, help='Training batch size')
    parser.add_argument('--image_size', default=640, type=int, help='Image size')
    parser.add_argument('--lr0', default=0.005, type=float, help='initial learning rate')
    parser.add_argument('--lrf', default=0.1, type=float, help='final learning rate fraction')
    parser.add_argument('--device', default="0", type=str, help='training device (e.g., 0,1,2,3)')
    parser.add_argument('--model', default="yolo11x-seg.yaml", type=str, help='model type ()')
    # Add parameters
    parser.add_argument('--box', default=7.5, type=float, help='Weight of the box loss component')
    parser.add_argument('--cls', default=0.5, type=float, help='Weight of the classification loss component')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout rate for regularization')
    parser.add_argument('--multi_scale', default=True, type=bool, help='Enable multi-scale training')
    parser.add_argument('--mosaic', default=0.3, type=float, help='Mosaic data augmentation factor (0.0 - 1.0)')
    parser.add_argument('--overlap_mask', default=True, type=bool, help='Whether to merge object masks into a single mask')
    parser.add_argument('--mask_ratio', default=2, type=int, help='Downsample ratio for segmentation masks')
    parser.add_argument('--cos_lr', default=True, type=bool, help='Enable cosine learning rate scheduler')
    parser.add_argument('--weight_decay', default=0.001, type=float, help='L2 regularization (weight decay)')
    parser.add_argument('--log_file', default='train_log.txt', type=str, help='Log file to save parameters')

    args = parser.parse_args()

    # Set up logging to output the parameters to a log file
    setup_logging(args.log_file)

    print ("Training model: {}".format(args.model))
    # Load a model
    model = YOLO(args.model)  # build a new model from YAML
    #model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
    #model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    train(
        model, 
        args.data_file, 
        args.output_folder, 
        device=args.device, 
        epochs=args.epochs, 
        warmup_epochs=args.warmup_epochs, 
        image_size=args.image_size, 
        batch_size=args.batch_size, 
        lr0=args.lr0, 
        lrf=args.lrf, 
        box=args.box,  # Pass box argument
        cls=args.cls,  # Pass cls argument
        dropout=args.dropout,  # Pass dropout argument
        multi_scale=args.multi_scale,  # Pass multi_scale argument
        mosaic=args.mosaic,  # Pass mosaic argument
        overlap_mask=args.overlap_mask,  # Pass overlap_mask argument
        mask_ratio=args.mask_ratio,  # Pass mask_ratio argument
        cos_lr=args.cos_lr,  # Pass cos_lr argument
        weight_decay=args.weight_decay,  # Pass weight_decay argument
        log_file=args.log_file  # Pass log_file argument
    )
