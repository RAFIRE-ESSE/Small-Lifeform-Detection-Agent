import cv2,os
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import random, pandas, math
import shutil, numpy

from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from sahi.predict import get_prediction

class customiser:
    def __new__(self, model, data_n):
        self.data = data_n
        
        #plt.imshow(Image.fromarray(self.data))
        #plt.show()
        data = self.image_extracter(self, self.data).to(device)            
        raw_output = model(data).float().view(-1).sigmoid().cpu().detach().numpy()

        return (raw_output)

class dataset_extracter:
    def __init__(self):
       	self.yolo = AutoDetectionModel.from_pretrained(
                                        model_type="ultralytics",
                                        model_path='l-154-m-2.pt',
                                        confidence_threshold=0.5,
                                        device="cuda:0")
        self.batch_size, self.out_batch, self.count = 1, 10, 0
        
        os.makedirs('stable_drive_i', exist_ok=True)
        os.makedirs('stable_drive_v', exist_ok=True)
        os.makedirs('vid_output', exist_ok=True)
        os.makedirs('results', exist_ok=True)

        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        #width = 1920
        #height = 1080
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        while True:
            ret, frame = cap.read()
            self.FrameCapture_i(frame)
            cv2.imshow("l-Feed", cv2.imread(f'demo_data/{self.count}/prediction_visual.png'))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break1
            self.count += 1

    def FrameCapture_i(self, data):
        cv2.imwrite('data.jpg', data)

        result = get_sliced_prediction('data.jpg', self.yolo,
                                       slice_height=512,
                                       slice_width=512,
                                       overlap_height_ratio=0.2,
                                       overlap_width_ratio=0.2)
                    
        result.export_visuals(export_dir=f"demo_data/{self.count}")

dataset_extracter()
