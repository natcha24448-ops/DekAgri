from ultralytics import YOLO
model = YOLO(r"C:\Users\ASUS\Downloads\best (3).pt")
print(model.names)
