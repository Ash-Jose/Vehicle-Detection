if __name__ == "__main__":
    from ultralytics import YOLO
    import os

    # Load your trained model
    model = YOLO(r'C:\Users\Ashlin\Documents\College\Mini Project\vehicle_detection\runs\detect\train\weights\best.pt')

    # Validate the model on the validation dataset
    results = model.val(data=r'C:\Users\Ashlin\Documents\College\Mini Project\vehicle_detection\dataset\data.yaml', imgsz=640, batch=16)

    # Print the validation results save directory
    print(f"Validation results saved to: {results.save_dir}")

    # Define the custom output directory for predictions
    predict_save_dir = r'C:\Users\Ashlin\Documents\College\Mini Project\vehicle_detection\runs\detect\predict'

    # Ensure the directory exists
    os.makedirs(predict_save_dir, exist_ok=True)

    # Test the model on the test images and specify the custom directory for saving predictions
    results = model.predict(source=r'C:\Users\Ashlin\Documents\College\Mini Project\vehicle_detection\dataset\valid\images', imgsz=640, conf=0.25, save=True, save_dir=predict_save_dir)
