#------------------IMPORT THE LIBRARY----------------------
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import os
import random
import sys
from PIL import Image, ImageOps
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import glob
import cv2
from torch.amp import GradScaler, autocast
import timm
import time
import asyncio
import struct
from bleak import BleakClient, BleakError
import datetime
import keyboard
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import envelope, hilbert
from scipy import interpolate

#-------------------DATA PRE-PROCESSING---------------------------
def process_data(data, is_train):   
    a = np.squeeze(np.array(data))
    # if is_train:
    #     a = a*np.random.uniform(0.95, 1.05)
    sos = signal.butter(2, 80, 'high', fs=900, output='sos')
    f = signal.sosfilt(sos, a[:])[100:-100]
    f = f / (max(f)-min(f))
    fr, t, z = signal.stft(f, 900, nperseg=60)
    z = np.abs(z)
    i1 = Image.fromarray((z*255).astype(np.uint8), mode='L')
    i1 = np.array(i1.resize((512, 256)))
 
    a = (a - min(a))/(max(a - min(a)))
    i2 = np.tile(a[100:-100], (30, 1))
    i2 = Image.fromarray((i2*255).astype(np.uint8), mode='L')
    i2 = np.array(i2.resize((512, 256)))
 
    image = np.array([i1, i1, i2])
    return image

def transform_data(data, is_train):
    data = process_data(data, is_train)
    return data
    
class d(Dataset):
    def __init__(self, root_dir, csv_file, is_train):
        self.root = root_dir
        self.csv  = csv_file
        self.is_train = is_train
    def __getitem__(self, index):
        s = self.csv.iloc[index, 0]
        s = s.replace('\\', '/')
        file     = glob.glob(str('/kaggle/input/data-bach-nick-robot/' + s))[0]
        raw_x    = pd.read_csv(file, header=None).values
        # print(file)
        x_tensor = transform_data(raw_x, self.is_train)
        label = int(self.csv.iloc[index, 1])
        output = torch.nn.functional.one_hot(torch.tensor(label), num_classes=20)
        return x_tensor, output
 
    
    def __len__(self):
        return len(self.csv)

import torch.nn as nn

#---------------------DEEP LEARNING MODEL---------------------------- 
class VNet(nn.Module):
    def __init__(self, model):
        super(VNet, self).__init__()
        self.model = timm.create_model(model, pretrained=True)
        self.fc0 = nn.Linear(self.model.classifier.out_features, 1024)
        self.fc1 = nn.Linear(1024, 256)   
        self.fc2 = nn.Linear(256, 20)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, image):
        x = image 
        x = self.model(image)
        x = self.fc0(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


#----------------------LOAD TRANIED WEIGHT-------------------------------
def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint['state_dict'])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_nm = "efficientnet_b0"
model    = VNet(model_nm).to(DEVICE)

checkpoint = torch.load('./voice_model/F0-epoch24.pth', map_location=torch.device('cpu'))
load_checkpoint(checkpoint, model)
model.eval()

#-------------------BLE DATA ACQUIRING------------------------
DEVICE_ADDRESS = "C4:FB:85:EA:84:60"
UART_SERVICE_UUID = "00001524-1212-efde-1523-785feabcd123"
UART_CHAR_UUID = "00001526-1212-efde-1523-785feabcd123"
col = ['start', 'finish', 
       'banana', 'apple', 'bread', 'orange',
       'one', 'two', 'three']

data = []
async def notification_handler(sender, value):
    global data
    for i in range(0, len(value), 2):
        v = value[i] | (value[i + 1] << 8)
        data.append(v)

async def main():
    global data
    print('Connecting')
    async with BleakClient(DEVICE_ADDRESS) as client:

        print(f"Connected to {DEVICE_ADDRESS}")
        await client.start_notify(UART_CHAR_UUID, notification_handler)
        while 1:
            user_input = input(f"press ENTER to start...")
            print("Hearing...")
            data = []
            print(datetime.datetime.now())
            await asyncio.sleep(3)                  # sound signal is collected and stored in "data" within 3s
            print(datetime.datetime.now())
            
            if (len(data) >= 3000):
                raw_x = np.squeeze(np.array(data[-3000:]))
                x_tensor = torch.as_tensor(np.array([transform_data(raw_x, 0)])).to(DEVICE).to(torch.float32)
                start_time = time.time()
                out = model(x_tensor)               # detect the sound signal, output is a number corresponding to the class
                print(col[torch.argmax(torch.as_tensor(out)).item()]) # convert the output number to label
                end_time = time.time()
                running_time = end_time - start_time
                print(f'running time: {running_time}s')
                print('------------------------------')
            else:
                print('Loss connection, please reconnect!')
                break
        await client.stop_notify(UART_CHAR_UUID)

async def predict_once(): 
    global data
    print('Connecting')
    res = None

    try: 
        client = BleakClient(DEVICE_ADDRESS)
        async with client:

            print(f"Connected to {DEVICE_ADDRESS}")
            await client.start_notify(UART_CHAR_UUID, notification_handler)

            print("Hearing...")
            data = []
            await asyncio.sleep(3.5)
            
            if (len(data) >= 3000):
                raw_x = np.squeeze(np.array(data[-3000:]))
                x_tensor = torch.as_tensor(np.array([transform_data(raw_x, 0)])).to(DEVICE).to(torch.float32)

                out = model(x_tensor)
                res = col[torch.argmax(torch.as_tensor(out)).item()] # convert the output number to label
                print(f'Detected {res}') 
            else:
                print('Loss connection, please reconnect!')

            await client.stop_notify(UART_CHAR_UUID)
    
    except BleakError as e:
        return f"Bluetooth error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"
    return res

# asyncio.run(main())
