from moviepy.editor import VideoFileClip
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
    def __init__(self, vid_input):
        self.batch_size, self.out_batch = 1, 10
        self.yolo = AutoDetectionModel.from_pretrained(
                                        model_type="ultralytics",
                                        model_path='l-154-m-2.pt',
                                        confidence_threshold=0.3,
                                        device="cpu",  # or 'cuda:0'
                                    )
        self.SWIN_NET = nn.DataParallel(Swin_Model().float()).to(device)
        self.SWIN_NET.load_state_dict(torch.load(f"lizard-B0-N.pth", weights_only=False, map_location=torch.device('cpu')))

        os.makedirs('results_cnn', exist_ok=True)
        os.makedirs('results_ob', exist_ok=True)
        os.makedirs('stable_drive_i', exist_ok=True)
        os.makedirs('stable_drive_v', exist_ok=True)
        os.makedirs('vid_output', exist_ok=True)
        
        for self.title in os.listdir(vid_input): 
            print(f'FILE : {self.title}')
            if vid_input == 'stable_drive_v':
                self.FrameCapture_v(f'{vid_input}/{self.title}')
    def Swin_Model_customiser(self, data_n):
        self.data = data_n
        data = self.image_extracter(self.data).to(device)  
        raw_output = self.SWIN_NET(data).float().view(-1).sigmoid().cpu().detach().numpy()
        return raw_output

    def image_extracter(self, image_array):
    	self.shape, color=(300, 300), cv2.COLOR_BGR2RGB
    	processed_image=[]
    	transformer = transforms.Compose([
                          transforms.Resize(self.shape),
                          transforms.ToTensor()
                          ])
    	kernel = numpy.array([[0, -3, -3], 
                              [3,  0, -3], 
                              [3,  3,  0]])

    	image = transformer(Image.fromarray(image_array).convert('RGB'))
    	return torch.Tensor(numpy.array(image).reshape((1, 3, self.shape[0], self.shape[1]))).float().to(device)

    def FrameCapture_v(self, path):
        video, self.count = VideoFileClip(path), 0
        self.fps, data_a = 1, {}

        for i_count, self.data in enumerate(video.iter_frames(fps = self.fps)):
            cv2.imwrite('data.jpg', self.data)

            result = get_sliced_prediction('data.jpg', self.yolo,
				    slice_height=256,
				    slice_width=256,
				    overlap_height_ratio=0.3,
				    overlap_width_ratio=0.3,
				    postprocess_type="GREEDYNMM", 
				    postprocess_match_metric="IOS",
				    postprocess_match_threshold=0.5,)

            result_ = result.to_coco_predictions()
            if result_ != []:
            	data_z = self.data
            	for i in range(len(result_)):
            		x_min, y_min, x_max, y_max = xywh_to_xyxy(result_[i]['bbox'])
            		acc = result_[i]['score']
            		if acc > 0.1:
            			swin_acc = self.Swin_Model_customiser(self.data[y_min : y_max, x_min : x_max])
            			for i in range(len(swin_acc)):
            				print(swin_acc[i], acc)
            				if swin_acc[i] > 0.02:
            					data_z = cv2.rectangle(cv2.cvtColor(data_z, cv2.COLOR_RGB2BGR), (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
	            				cv2.putText(data_z, f'OB: {str(acc)[:4]} CNN: {str(swin_acc[i])[:4]}', 
	            							(x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 1, 
	            							(255, 0, 0), 2, cv2.LINE_AA)

            	cv2.imwrite(f'results_ob/image{i_count}.jpg', data_z)
            else:
            	cv2.imwrite(f'results_ob/image{i_count}.jpg', self.data)

dataset_extracter('stable_drive_v')
