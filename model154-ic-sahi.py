import cv2,os
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import random, pandas, math
import shutil, numpy

from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from sahi.predict import get_prediction


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

seed = 999
print("Random Seed: ", seed)
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class Swin_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = torchvision.models.efficientnet_b0()
        self.model = create_feature_extractor(effnet, ['flatten'])
        self.nn_fracture = torch.nn.Sequential(
            torch.nn.Linear(1280, 1)
        )
    def forward(self, x):
        x = self.model(x)['flatten']
        x = self.nn_fracture(x)
        return x    

def xywh_to_xyxy(xywh):
    x1 = xywh[0]
    y1 = xywh[1]
    x2 = xywh[0] + xywh[2]
    y2 = xywh[1] + xywh[3]
    return numpy.array([int(x1), int(y1), int(x2), int(y2)])


class dataset_extracter:
    def __init__(self):
       	self.yolo = AutoDetectionModel.from_pretrained(
                                        model_type="ultralytics",
                                        model_path='l-154-m-2.pt',
                                        confidence_threshold=0.5,
                                        device="cuda:0")
        self.batch_size, self.out_batch, self.count = 1, 10, 0
        self.SWIN_NET = nn.DataParallel(Swin_Model().float()).to(device)
        self.SWIN_NET.load_state_dict(torch.load(f"lizard-B0-N.pth", weights_only=False, map_location=torch.device('cpu')))
        
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
            cv2.imshow("l-Feed", self.FrameCapture_i(frame))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break1
            self.count += 1

    def Swin_Model_customiser(self, data_n):
        self.data = data_n
        data = self.image_extracter(self.data).to(device)  
        raw_output = self.SWIN_NET(data).float().view(-1).sigmoid().cpu().detach().numpy()
        return raw_output

    def image_extracter(self, image_array):
    	self.shape, color=(256, 256), cv2.COLOR_BGR2RGB
    	processed_image=[]
    	transformer = transforms.Compose([
                          transforms.Resize(self.shape),
                          transforms.RandomRotation(15),
                          transforms.RandomHorizontalFlip(),
                          transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                          transforms.ToTensor()
                          ])
    	kernel = numpy.array([[0, -3, -3], 
                              [3,  0, -3], 
                              [3,  3,  0]])

    	image = transformer(Image.fromarray(image_array).convert('RGB'))
    	return torch.Tensor(numpy.array(image).reshape((1, 3, self.shape[0], self.shape[1]))).float().to(device)

    def FrameCapture_i(self, data):
        cv2.imwrite('data.jpg', data)

        result = get_sliced_prediction('data.jpg', self.yolo,
				    slice_height=1024,
				    slice_width=1024,
				    overlap_height_ratio=0.2,
				    overlap_width_ratio=0.2)
        self.data = data

        result_ = result.to_coco_predictions()
        if result_ != []:
            data_z = self.data
            for i in range(len(result_)):
                x_min, y_min, x_max, y_max = xywh_to_xyxy(result_[i]['bbox'])
                acc = result_[i]['score']
                if acc > 0.3:
                    swin_acc = self.Swin_Model_customiser(self.data[y_min : y_max, x_min : x_max])
                    for i in range(len(swin_acc)):
                        print(swin_acc[i], acc)
                        if swin_acc[i] > 0.5:
                            data_z = cv2.rectangle(cv2.cvtColor(data_z, cv2.COLOR_RGB2BGR), (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
                            cv2.putText(data_z, f'OB: {str(acc)[:4]} CNN: {str(swin_acc[i])[:4]}', 
                                        (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                        (255, 0, 0), 2, cv2.LINE_AA)
            	
            return cv2.cvtColor(data_z, cv2.COLOR_RGB2BGR)
        else:
            return self.data
            	
dataset_extracter()
