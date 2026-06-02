from moviepy.editor import VideoFileClip
import cv2,os
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import random, pandas, math
import shutil, numpy
from imutils.video import VideoStream

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

class customiser:
    def __new__(self, model, data_n):
        self.data = data_n
        
        #plt.imshow(Image.fromarray(self.data))
        #plt.show()
        data = self.image_extracter(self, self.data).to(device)            
        raw_output = model(data).float().view(-1).sigmoid().cpu().detach().numpy()

        return (raw_output)
        
    def image_extracter(self, image_array):
        self.shape,color=(70, 70),cv2.COLOR_BGR2RGB
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
        for i in image_array:
	        image = transformer(Image.fromarray(cv2.filter2D(i, -1, kernel, borderType=cv2.BORDER_REPLICATE)).convert('RGB'))
	        #transformer(Image.fromarray(i))
	        #transformer(Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2HSV)))
	        #transformer(Image.fromarray(cv2.Canny(cv2.GaussianBlur(i, (5,5), 0, 0), 100, 200)).convert('RGB'))
	        #transformer(Image.fromarray(cv2.filter2D(i, -1, kernel, borderType=cv2.BORDER_REPLICATE)).convert('RGB'))
	        processed_image.append(image)
            
        return torch.Tensor(numpy.array(processed_image).reshape((len(processed_image), 3, self.shape[0], self.shape[1]))).float().to(device)

class dataset_extracter:
    def __init__(self):
       	self.yolo = YOLO('l-154-m.pt')
        self.batch_size, self.out_batch = 1, 10
        
        os.makedirs('stable_drive_i', exist_ok=True)
        os.makedirs('stable_drive_v', exist_ok=True)
        os.makedirs('vid_output', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        #self.SWIN_NET = nn.DataParallel(Swin_Model().float()).to(device)
        #self.SWIN_NET.load_state_dict(torch.load(f"lizard-B0-N.pth", weights_only=False, map_location=torch.device('cpu')))
        #self.SWIN_NET = self.SWIN_NET.eval()

        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        #width = 1920
        #height = 1080
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        while True:
            ret, frame = cap.read()
            self.FrameCapture_i(frame)
            

    def FrameCapture_i(self, data):
        results = self.yolo(data, verbose=False)
        for result in results:
            res = result.boxes.xyxy.cpu().detach().numpy()
            acc = result.boxes.conf.cpu().detach().numpy()

            if len(res) > 0:
            	data_c, z_i = [], []
            	for z,l in zip(res, acc):
            		if l > 0.0:
            			data_c.append(data[int(z[1]):int(z[1])+int(z[3]), int(z[0]):int(z[0])+int(z[2])])
            			z_i.append(z)
            			data = cv2.rectangle(cv2.cvtColor(data, cv2.COLOR_RGB2BGR), (round(z[0]), round(z[1])), (round(z[2]), round(z[3])), (255, 0, 0), 3)
            	#swin_acc = customiser(self.SWIN_NET, data_c)
            	#for i in range(len(swin_acc)):
            		#if swin_acc[i] > 0.1:
            			#print(swin_acc[i])
            			#data = cv2.rectangle(cv2.cvtColor(data, cv2.COLOR_RGB2BGR), (round(z_i[i][0]), round(z_i[i][1])), (round(z_i[i][2]), round(z_i[i][3])), (255, 0, 0), 3)
            
            cv2.imshow("l-Feed", data)
            #cv2.imwrite(f'vid_output/{self.title}.jpg', data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

dataset_extracter()

