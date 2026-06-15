import ultralytics

model = ultralytics.YOLO("yolov11n.pt")

model.train(
    data="data.yaml",
    epochs=100
)