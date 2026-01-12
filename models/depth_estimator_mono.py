import cv2
import torch


class MonoDepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        """
        model_type: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Loading MiDaS ({model_type}) on {self.device}...")

        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def estimate(self, frame_bgr):
        """
        Повертає відносну глибину (relative depth).
        MiDaS зазвичай повертає інверсну глибину (disparity-like).
        """
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        return output
