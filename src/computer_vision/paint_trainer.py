from ultralytics import YOLO

def main():
    #load model/existing weights
    model = YOLO("yolo11n-pose.pt")

    #run model
    results = model.train(
        data="/Users/USERNAME/Desktop/SGA/data/paint_data/data.yaml",
        
        epochs=150,             
        patience=25,            
        imgsz=640, 
        device="mps",
        batch=8,                
        fliplr=0.0,             
        scale=0.15,             
        translate=0.15,         
        degrees=5.0,            
        mosaic=0.0,             
        box=6.0,                
        pose=25.0               
    )

if __name__ == "__main__":
    main()