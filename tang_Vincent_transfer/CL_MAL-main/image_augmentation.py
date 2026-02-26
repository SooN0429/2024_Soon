import os
import PIL.Image as Image
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import warnings
from torchvision import transforms
from torchvision.transforms import functional as TF

root_path = 'D:\\Tang\\create_datasets\\feature_extract\\dataset\\'
fruits = ['apple', 'banana', 'carambola', 'guava', 'muskmelon', 'peach', 'pear' ,'tomato']
connects = ['_to_\\', 'to_']
files = ['image\\', 'test_image\\']
names = ['source\\', 'target\\']
Train_Test = ['train\\', 'test\\']
Neg_Pos = ['negative\\', 'positive\\']
attribute = 'gray_'
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3)
])

for file in files:
	for source_fruit in fruits:
		for target_fruit in fruits:
			for name in names:
				for TT in Train_Test:
					if name == 'source\\' and TT == 'test\\':
						continue
					for NP in Neg_Pos:
						file_path = root_path+file+source_fruit+connects[0]+connects[1]+target_fruit+'\\sn_200_sp_200\\'+name+TT+NP
						new_img_path = root_path+(attribute+file)+source_fruit+connects[0]+connects[1]+target_fruit+'\\sn_200_sp_200\\'+name+TT+NP
						# print(file_path)
						FFile = os.listdir(file_path)
						FFlight = len(FFile)
						for i in range(FFlight):
							if NP == 'negative\\':
								new_name = 'negative'+'_'+str(i)+'.png'
							else:
								if name == 'source\\':
									new_name = source_fruit+'_'+str(i)+'.png'
								if name == 'target\\':
									new_name = target_fruit+'_'+str(i)+'.png'
							# print(FFile[i])
							image_path = file_path+new_name
							img_pil = Image.open(image_path, mode='r')
							new_img = transform(img_pil)
							if not os.path.exists(new_img_path):
								os.makedirs(new_img_path)
							new_img.save(new_img_path+new_name, 'png')
