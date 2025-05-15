if __name__ == "__main__":
    
    from ultralytics import YOLO
    import torch

    torch.cuda.empty_cache()
    
    # Load a base model 
    model = YOLO('yolov8s.pt')

    # Start training
    model.train(
        data='dataset/data.yaml',
        epochs= 50,
        imgsz=640,        
        batch=16,         
        workers=12,         
        device=0          
    )
