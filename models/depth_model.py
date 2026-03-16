import torch
import cv2

class DepthEstimator:

    def __init__(self):

        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        self.transform = transforms.small_transform

    def predict(self, frame):

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():

            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        return depth