import warnings, time
warnings.simplefilter(action='ignore')

from moviepy.editor import VideoFileClip
import cv2, os, numpy
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import random, pandas, math
import shutil

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

class data_preprocessor:
    def __init__(self):
        print('DATA PREPROCESSING')
        self.files = os.listdir('dataset/images')

        os.makedirs('processed_dataset/images/', exist_ok=True)
        os.makedirs('processed_dataset/labels/', exist_ok=True)
        os.makedirs('processed_dataset/images_m/', exist_ok=True)

        os.makedirs('processed_dataset/train/images', exist_ok=True)
        os.makedirs('processed_dataset/train/labels', exist_ok=True)
        os.makedirs('processed_dataset/val/images', exist_ok=True)
        os.makedirs('processed_dataset/val/labels', exist_ok=True)

        self.data_compiler()
        self.data_spliter()
        
    def data_compiler(self):
        for i in self.files:
            suport_dict = numpy.load(f'dataset/images_m/{i}/suport_file.npy',allow_pickle='TRUE').item()

            for j in suport_dict.values():
                try:
                    image = cv2.imread(f'dataset/images/{i}/{j}')
                    image_1 = cv2.imread(f'dataset/images_m/{i}/{j}')

                    x_1, x_3, x_2, x_4 = [int(i) for i in open(f'dataset/lables/{i}/{j.split(".")[0]}.txt', 'r').read().split(' ')[:4]]
                    height, width, channels = image.shape
                    dwidth = (1.0 / width)
                    dheight = (1.0 / height)
                    x = (((x_1 + x_2) / 2.0) * dwidth)
                    y = (((x_3 + x_4) / 2.0) * dheight)
                    wid = ((x_2 - x_1) * dwidth)
                    hig = ((x_4 - x_3) * dheight)
                    if x < 0: x = 0
                    if y < 0: y = 0
                    if wid < 0: wid = 0
                    if hig < 0: hig = 0

                    #print(f'0 {x} {y} {wid} {hig}')
                    cv2.imwrite(f'processed_dataset/images_m/{i}_{j}', image_1)
                    cv2.imwrite(f'processed_dataset/images/{i}_{j}', image)
                    open(f'processed_dataset/labels/{i}_{j.split(".")[0]}.txt', 'w').write(f'0 {x} {y} {wid} {hig}')

                except:
                    pass
                
    def data_spliter(self):
        X_train, X_test, y_train, y_test = train_test_split(os.listdir('processed_dataset/images/'), 
                                                            os.listdir('processed_dataset/labels/'), 
                                                            test_size=0.1, 
                                                            random_state=100)
        for i in X_train:
        	cv2.imwrite(f'processed_dataset/train/images/{i}', cv2.cvtColor(cv2.imread(f'processed_dataset/images/{i}'), cv2.COLOR_BGR2RGB))
        	open(f'processed_dataset/train/labels/{i.split(".")[0]}.txt', 'w').write(open(f'processed_dataset/labels/{i.split(".")[0]}.txt', 'r').read())

        for i in X_test:
            cv2.imwrite(f'processed_dataset/val/images/{i}', cv2.cvtColor(cv2.imread(f'processed_dataset/images/{i}'), cv2.COLOR_BGR2RGB))
            open(f'processed_dataset/val/labels/{i.split(".")[0]}.txt', 'w').write(open(f'processed_dataset/labels/{i.split(".")[0]}.txt', 'r').read())
        
        shutil.rmtree('processed_dataset/images/')
        shutil.rmtree('processed_dataset/labels/')

class Swin_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = torchvision.models.swin_v2_b()
        self.model = create_feature_extractor(effnet, ['flatten'])
        self.nn_fracture = torch.nn.Sequential(
            torch.nn.Linear(1024, 1)
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
        raw_output = model(data).float().view(-1).sigmoid().cpu().detach().numpy()[0]

        return (raw_output)
        
    def image_extracter(self, image_array):
        self.shape,color=(256, 256),cv2.COLOR_BGR2RGB
        processed_image=[]
        transformer = transforms.Compose([
                          transforms.Resize(self.shape),
                          transforms.RandomRotation(15),
                          transforms.RandomHorizontalFlip(),
                          transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                          transforms.ToTensor()
                          ])
        image = transformer(Image.fromarray(image_array))
        processed_image.append(image)
            
        return torch.Tensor(numpy.array(processed_image).reshape((len(processed_image), 3, self.shape[0], self.shape[1]))).float().to(device)

class dataset_extracter:
    def __init__(self, vid_input):
        self.yolo = YOLO('yolo_n.pt')
        self.batch_size, self.out_batch = 30, 10

        #self.SWIN_NET = Swin_Model().float()
        #self.SWIN_NET = nn.DataParallel(self.SWIN_NET).to(device)
        #self.SWIN_NET.load_state_dict(torch.load(f"swin_v2.pth", weights_only=False, map_location=torch.device('cpu')))

        os.makedirs('dataset', exist_ok=True)
        os.makedirs('dataset/lables', exist_ok=True)
        os.makedirs('dataset/images', exist_ok=True)
        os.makedirs('dataset/images_m', exist_ok=True)
        
        for self.title in os.listdir(vid_input): #len(vid_input) os.listdir(vid_input)
            print(f'FILE : {self.title}')

            #self.title = self.title.split('.')[0]
            if os.path.exists(f'dataset/images/{self.title}') != True:
                os.mkdir(f'dataset/images/{self.title}')
                os.mkdir(f'dataset/lables/{self.title}')
                os.mkdir(f'dataset/images_m/{self.title}')
            
            self.FrameCapture(f'{vid_input}/{self.title}')
            print(f'TOTAL NUMBER EXTRACTED DATA FOR DIRECTORY {self.title} : {len(os.listdir(f"dataset/images/{self.title}"))}')
            
    def FrameCapture(self, path):
        video, self.count = VideoFileClip(path), 0
        self.fps, self.acc, data_a = video.fps, {}, {}
        print(self.fps)
        for i, data in enumerate(video.iter_frames(fps = self.fps)): #self.fps
            data_a[self.count] = data 
            self.count += 1
            if (self.count % self.batch_size) == 0:
            	print(self.count)
            	self.data_colecter(list(data_a.values()), list(data_a.keys()))
            	data_a.clear()

        frame_count = self.count
        acc_rate = len(self.acc)-(self.out_batch)
        if acc_rate < 0:
            acc_rate = 0
        numpy.save(f'dataset/images_m/{self.title}/suport_file.npy', {i : self.acc[i] for i in sorted(self.acc.keys())[acc_rate : len(self.acc)]}) 

    def data_colecter(self, data, n_array):
        results = self.yolo(data, verbose=False)

        for result, n, data_i in zip(results, n_array, range(len(data))):
        	res, cod_, cod_1 = result.boxes.xyxy.cpu().detach().numpy(), "", []
        	acc, acc_add, acc_len = result.boxes.conf.cpu().detach().numpy(), 0, 0

        	if len(res) > 0:
        		for z,l in zip(res, acc):
        			if l >= 0.80:
        				#swin_acc = customiser(self.SWIN_NET, data[data_i][int(z[1]):int(z[1])+int(z[3]), int(z[0]):int(z[0])+int(z[2])])
        				#if swin_acc >= 0.80:
	        				acc_add += l
	        				acc_len += 1
	        				cod_1.append(z)
	        				for c in z:
	        					cod_ += f'{round(c)} '
	        				cod_ += f'\n'

	        if cod_ != '':
	        	self.acc[(acc_add/acc_len)] = f'image{n}.jpg'
	        	open(f'dataset/lables/{self.title}/image{n}.txt', 'w').write(cod_)
	        	cv2.imwrite(f'dataset/images/{self.title}/image{n}.jpg', data[data_i])

	        	for z in cod_1:
	        		data[data_i] = cv2.rectangle(cv2.cvtColor(data[data_i], cv2.COLOR_RGB2BGR), (round(z[0]), round(z[1])), (round(z[2]), round(z[3])), (255, 0, 0), 3)
	        	cv2.imwrite(f'dataset/images_m/{self.title}/image{n}.jpg', data[data_i])

	        	#result.save(filename = f'dataset/images_m/{self.title}/image{n}.jpg')

class Yolo_Finetuner:
    def __init__(self):
    	print('YOLO FINETUNING...')
    	CLASSES=['snakes']

    	yaml_content = f"""
        train: /home/devil/Downloads/yolo-reinforced/processed_dataset/train
        val: /home/devil/Downloads/yolo-reinforced/processed_dataset/val
        
        nc: {len(CLASSES)}
        names: {CLASSES}
        """  	
    	with open("dataset.yaml", "w") as f:
    		f.write(yaml_content)
    	self.model_finetuner()
            
    def model_finetuner(self):
    	yolo = YOLO("satablity_asserts/yolo_n.pt", verbose=False)

    	yolo.train(data='dataset.yaml', epochs=10)
    	valid_results = yolo.val()

    	shutil.copy2('runs/detect/train/weights/best.pt', 'yolo_n.pt')
    	return (valid_results)

while True:

	if os.path.exists(f'dataset') == True:
		shutil.rmtree('dataset')
	dataset_extracter('stable_drive')

	if os.path.exists(f'processed_dataset') == True:
		shutil.rmtree('processed_dataset')
	data_preprocessor()

	if os.path.exists(f'runs') == True:
		shutil.rmtree('runs')
	Yolo_Finetuner()
