import cv2
import tensorflow as tf


class ObjectDetector:
    def __init__(self, model_name="models/ssd_mobilenet_v2"):
        print("TensorFlow version:", tf.__version__)
        print(f"🔍 Loading TensorFlow model: {model_name}")
        self.model = tf.saved_model.load(model_name + "/saved_model")

    def detect(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(img)[tf.newaxis, ...]
        detections = self.model(input_tensor)

        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)
        scores = detections['detection_scores'][0].numpy()

        results = []
        for i in range(len(scores)):
            if scores[i] > 0.5:
                results.append({
                    "box": boxes[i],
                    "class_id": classes[i],
                    "score": scores[i]
                })
        return results
