from moviepy.editor import VideoFileClip
import cv2,os
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import random, pandas, math
import shutil, numpy
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

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

class dataset_extracter:
    def __init__(self, vid_input):
        
        self.batch_size, self.out_batch = 1, 10
        
        os.makedirs('results_cnn', exist_ok=True)
        os.makedirs('results_ob', exist_ok=True)
        os.makedirs('stable_drive_i', exist_ok=True)
        os.makedirs('stable_drive_v', exist_ok=True)
        os.makedirs('vid_output', exist_ok=True)
        
        for self.title in os.listdir(vid_input): 
            print(f'FILE : {self.title}')
            if vid_input == 'stable_drive_v':
                self.FrameCapture_v(f'{vid_input}/{self.title}')

    def Swin_Model_customiser(self, model, data_n):
        self.data = data_n
        
        #plt.imshow(Image.fromarray(self.data))
        #plt.show()
        data = self.image_extracter(self.data).to(device)            
        raw_output = model(data).float().view(-1).sigmoid().cpu().detach().numpy()

        return (raw_output)
        
    def image_extracter(self, image_array):
        self.shape,color=(500, 500),cv2.COLOR_BGR2RGB
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
            image = transformer(Image.fromarray(cv2.filter2D(i, -1, kernel)).convert('RGB'))
            #transformer(Image.fromarray(i).convert('RGB'))
            #transformer(Image.fromarray(cv2.Canny(cv2.GaussianBlur(i, (5,5), 0, 0), 100, 200)).convert('RGB'))
            #transformer(Image.fromarray(cv2.filter2D(i, -1, kernel, borderType=cv2.BORDER_REPLICATE)).convert('RGB'))
            processed_image.append(image)
            
        return torch.Tensor(numpy.array(processed_image).reshape((len(processed_image), 3, self.shape[0], self.shape[1]))).float().to(device)

    def FrameCapture_v(self, path):
        self.yolo = YOLO('l-154-m.pt')
        
        self.SWIN_NET = nn.DataParallel(Swin_Model().float()).to(device)
        self.SWIN_NET.load_state_dict(torch.load(f"lizard-B0-N.pth", weights_only=False, map_location=torch.device('cpu')))
        video, self.count = VideoFileClip(path), 0
        self.fps, data_a = 1, {}

        for i_count, self.data in enumerate(video.iter_frames(fps = self.fps)):
            data_a[self.count] = self.data

            if (self.count % self.batch_size) == 0:
                data = list(data_a.values())
                #print(data)
                results = self.yolo(data, verbose=False)
                for result, data_i in zip(results, range(len(data))):
                    res, cod_, cod_1 = result.boxes.xyxy.cpu().detach().numpy(), "", []
                    acc, acc_add, acc_len = result.boxes.conf.cpu().detach().numpy(), 0, 0

                    if len(res) > 0:
                        data_c, z_i = [], []
                        for z,l in zip(res, acc):
                            if l > 0.0:
                                data_c.append(data[data_i][int(z[1]):int(z[1])+int(z[3]), int(z[0]):int(z[0])+int(z[2])])
                                z_i.append(z)
                                #data = cv2.rectangle(cv2.cvtColor(data[data_i], cv2.COLOR_RGB2BGR), (round(z[0]), round(z[1])), (round(z[2]), round(z[3])), (255, 0, 0), 3)
                        swin_acc = self.Swin_Model_customiser(self.SWIN_NET, data_c)
                        for i in range(len(swin_acc)):
                            if swin_acc[i] > 0.1:
                                print(swin_acc[i])
                                acc_add += l
                                acc_len += 1
                                cod_1.append(z)
                                for c in z:
                                    cod_ += f'{round(c)} '
                                cod_ += f'\n'
                            
                    if cod_ != '':
                            for z in cod_1:
                                data[data_i] = cv2.rectangle(cv2.cvtColor(data[data_i], cv2.COLOR_RGB2BGR), (round(z[0]), round(z[1])), (round(z[2]), round(z[3])), (255, 0, 0), 3)
                            cv2.imwrite(f'results_cnn/image{i_count}.jpg', data[data_i])
                            result.save(filename = f'results_ob/image{self.count}.jpg')
                    else:
                        cv2.imwrite(f'results_cnn/image{i_count}.jpg', data[data_i])
                        result.save(filename = f'results_ob/image{self.count}.jpg')

                data_a.clear()
            self.count += 1
        height, width, channels = numpy.array(self.data)[0].shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video=cv2.VideoWriter(f'vid_output/{self.title}', fourcc, self.fps, (width,height))

        for i in range(self.count):
            video.write(cv2.imread(f"results_cnn/image{i}.jpg"))
        video.release()
        #shutil.rmtree('results_cnn')

dataset_extracter('stable_drive_v')
