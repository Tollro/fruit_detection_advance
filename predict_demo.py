import ultralytics

model = ultralytics.YOLO("yolov11n.pt")
results = model.predict(source="test_images", save=True)