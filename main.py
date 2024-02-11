from ultralytics import YOLO

#loading model
model = YOLO("yolov8n.yaml")


#use model
results = model.train(data="config.yaml", epochs=25)   #train the model