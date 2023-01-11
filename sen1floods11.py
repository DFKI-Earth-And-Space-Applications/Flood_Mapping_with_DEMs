import sys

import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.models as models
import torch.nn as nn
import random
from PIL import Image
from time import time
import csv
import os
import numpy as np
from scipy.ndimage import laplace
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import math
import pyproj
import richdem as rd
import json
import colorsys

class InMemoryDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])

    def __len__(self):
        return len(self.data_list)

def get_channel_id(channel_name: str):
    channel_dict = {
        "VV":0,
        "VH":1,
        "DEM":2,
        "coastal":3,
        "blue":4,
        "green":5,
        "red":6,
        "redEdge1":7,
        "redEdge2":8,
        "RedEdge3":9,
        "NIR":10,
        "Narrow NIR":11,
        "Cirrus":12,
        "Water Vapor":13,
        "SWIR-1":14,
        "SWIR-2":15,
        "NDWI":16,
        "MNDWI":17,
        "AWEI":18,
        "AWEISH":19,
    }
    return channel_dict[channel_name]

def get_hsv_names(channel_indeces):
    channel_dict = {
        0: "VV",
        1: "VH",
        2: "DEM",
        3: "coastal",
        4: "blue",
        5: "green",
        6: "red",
        7: "redEdge1",
        8: "redEdge2",
        9: "RedEdge3",
        10: "NIR",
        11: "Narrow NIR",
        12: "Cirrus",
        13: "Water Vapor",
        14: "SWIR-1",
        15: "SWIR-2",
        16: "NDWI",
        17: "MNDWI",
        18: "AWEI",
        19: "AWEISH",
    }
    return f'{channel_dict[channel_indeces[0]]}, {channel_dict[channel_indeces[1]]}, {channel_dict[channel_indeces[2]]}'

def processAndAugment(data):
    (x, y, coords, depressed_dem, flow, depressed_flow) = data
    im, label, x_y_coord = x.copy(), y.copy(), coords

    # compute 1st & 2nd order derivative of DEM
    # train_data got created from train_data.append((arr_x, arr_y, coords, depressed_dem, flow, depressed_flow))

    # args.channels: [VV, VH, DEM, Gnorm, Gy, Gx, Laplacian, flow_metric, s2channels, ndwi, mndwi, awei, aweish]
    # convert to PIL for easier transforms
    if args.channels[0]:
        vv_channel = Image.fromarray(im[0])
    if args.channels[1]:
        vh_channel = Image.fromarray(im[1])
    if not args.depression:
        processed_dem = im[2]
    elif args.depression:
        processed_dem = depressed_dem
    # compute 1st & 2nd order derivative of DEM
    # train_data got created from train_data.append((arr_x, arr_y, coords))
    if args.gradient == 'degree':
        Gy, Gx, Gnorm, Laplacian = calc_derivates_deg(processed_dem)
    elif args.gradient == 'meter':
        Gy, Gx, Gnorm, Laplacian = calc_derivates_meter(processed_dem, x_y_coord)
    else:
        sys.exit('WRONG GRADIENT')
    if args.channels[2]:
        dem_channel = Image.fromarray(processed_dem)
    if args.channels[3]:
        gnorm_channel = Image.fromarray(Gnorm)
    if args.channels[4]:
        gy_channel = Image.fromarray(Gy)
    if args.channels[5]:
        gx_channel = Image.fromarray(Gx)
    if args.channels[6]:
        laplacian_channel = Image.fromarray(Laplacian)
    if args.channels[7]:
        if not args.depression:
            flow_metric_channel = flow
        if args.depression:
            flow_metric_channel = depressed_flow
        flow_metric_channel = Image.fromarray(flow_metric_channel)
    if args.channels[8]:
        coastal_channel = Image.fromarray(im[3])
    if args.channels[9]:
        blue_channel = Image.fromarray(im[4])
    if args.channels[10]:
        green_channel = Image.fromarray(im[5])
    if args.channels[11]:
        red_channel = Image.fromarray(im[6])
    if args.channels[12]:
        rededge1_channel = Image.fromarray(im[7])
    if args.channels[13]:
        rededge2_channel = Image.fromarray(im[8])
    if args.channels[14]:
        rededge3_channel = Image.fromarray(im[9])
    if args.channels[15]:
        nir_channel = Image.fromarray(im[10])
    if args.channels[16]:
        narrownir_channel = Image.fromarray(im[11])
    if args.channels[17]:
        watervapor_channel = Image.fromarray(im[12])
    if args.channels[18]:
        cirrus_channel = Image.fromarray(im[13])
    if args.channels[19]:
        swir1_channel = Image.fromarray(im[14])
    if args.channels[20]:
        swir2_channel = Image.fromarray(im[15])
    if args.channels[21]:
        ndwi_channel = Image.fromarray(im[16])
    if args.channels[22]:
        mndwi_channel = Image.fromarray(im[17])
    if args.channels[23]:
        awei_channel = Image.fromarray(im[18])
    if args.channels[24]:
        aweish_channel = Image.fromarray(im[19])

    if args.channels[25]:
        hue_channel = Image.fromarray(im[20])
    if args.channels[26]:
        saturation_channel = Image.fromarray(im[21])
    if args.channels[27]:
        value_channel = Image.fromarray(im[22])

    label = Image.fromarray(label.squeeze())

    # Get params for random transforms
    i, j, h, w = transforms.RandomCrop.get_params(Image.fromarray(im[0]), (256, 256))

    if args.channels[0]:
        vv_channel = F.crop(vv_channel, i, j, h, w)
    if args.channels[1]:
        vh_channel = F.crop(vh_channel, i, j, h, w)
    if args.channels[2]:
        dem_channel = F.crop(dem_channel, i, j, h, w)
    if args.channels[3]:
        gnorm_channel = F.crop(gnorm_channel, i, j, h, w)
    if args.channels[4]:
        gy_channel = F.crop(gy_channel, i, j, h, w)
    if args.channels[5]:
        gx_channel = F.crop(gx_channel, i, j, h, w)
    if args.channels[6]:
        laplacian_channel = F.crop(laplacian_channel, i, j, h, w)
    if args.channels[7]:
        flow_metric_channel = F.crop(flow_metric_channel, i, j, h, w)
    if args.channels[8]:
        coastal_channel = F.crop(coastal_channel, i, j, h, w)
    if args.channels[9]:
        blue_channel = F.crop(blue_channel, i, j, h, w)
    if args.channels[10]:
        green_channel = F.crop(green_channel, i, j, h, w)
    if args.channels[11]:
        red_channel = F.crop(red_channel, i, j, h, w)
    if args.channels[12]:
        rededge1_channel = F.crop(rededge1_channel, i, j, h, w)
    if args.channels[13]:
        rededge2_channel = F.crop(rededge2_channel, i, j, h, w)
    if args.channels[14]:
        rededge3_channel = F.crop(rededge3_channel, i, j, h, w)
    if args.channels[15]:
        nir_channel = F.crop(nir_channel, i, j, h, w)
    if args.channels[16]:
        narrownir_channel = F.crop(narrownir_channel, i, j, h, w)
    if args.channels[17]:
        watervapor_channel = F.crop(watervapor_channel, i, j, h, w)
    if args.channels[18]:
        cirrus_channel = F.crop(cirrus_channel, i, j, h, w)
    if args.channels[19]:
        swir1_channel = F.crop(swir1_channel, i, j, h, w)
    if args.channels[20]:
        swir2_channel = F.crop(swir2_channel, i, j, h, w)
    if args.channels[21]:
        ndwi_channel = F.crop(ndwi_channel, i, j, h, w)
    if args.channels[22]:
        mndwi_channel = F.crop(mndwi_channel, i, j, h, w)
    if args.channels[23]:
        awei_channel = F.crop(awei_channel, i, j, h, w)
    if args.channels[24]:
        aweish_channel = F.crop(aweish_channel, i, j, h, w)

    if args.channels[25]:
        hue_channel = F.crop(hue_channel, i, j, h, w)
    if args.channels[26]:
        saturation_channel = F.crop(saturation_channel, i, j, h, w)
    if args.channels[27]:
        value_channel = F.crop(value_channel, i, j, h, w)

    label = F.crop(label, i, j, h, w)

    if random.random() > 0.5:
        if args.channels[0]:
            vv_channel = F.hflip(vv_channel)
        if args.channels[1]:
            vh_channel = F.hflip(vh_channel)
        if args.channels[2]:
            dem_channel = F.hflip(dem_channel)
        if args.channels[3]:
            gnorm_channel = F.hflip(gnorm_channel)
        if args.channels[4]:
            gy_channel = F.hflip(gy_channel)
        if args.channels[5]:
            gx_channel = F.hflip(gx_channel)
        if args.channels[6]:
            laplacian_channel = F.hflip(laplacian_channel)
        if args.channels[7]:
            flow_metric_channel = F.hflip(flow_metric_channel)
        if args.channels[8]:
            coastal_channel = F.hflip(coastal_channel)
        if args.channels[9]:
            blue_channel = F.hflip(blue_channel)
        if args.channels[10]:
            green_channel = F.hflip(green_channel)
        if args.channels[11]:
            red_channel = F.hflip(red_channel)
        if args.channels[12]:
            rededge1_channel = F.hflip(rededge1_channel)
        if args.channels[13]:
            rededge2_channel = F.hflip(rededge2_channel)
        if args.channels[14]:
            rededge3_channel = F.hflip(rededge3_channel)
        if args.channels[15]:
            nir_channel = F.hflip(nir_channel)
        if args.channels[16]:
            narrownir_channel = F.hflip(narrownir_channel)
        if args.channels[17]:
            watervapor_channel = F.hflip(watervapor_channel)
        if args.channels[18]:
            cirrus_channel = F.hflip(cirrus_channel)
        if args.channels[19]:
            swir1_channel = F.hflip(swir1_channel)
        if args.channels[20]:
            swir2_channel = F.hflip(swir2_channel)
        if args.channels[21]:
            ndwi_channel = F.hflip(ndwi_channel)
        if args.channels[22]:
            mndwi_channel = F.hflip(mndwi_channel)
        if args.channels[23]:
            awei_channel = F.hflip(awei_channel)
        if args.channels[24]:
            aweish_channel = F.hflip(aweish_channel)

        if args.channels[25]:
            hue_channel = F.hflip(hue_channel)
        if args.channels[26]:
            saturation_channel = F.hflip(saturation_channel)
        if args.channels[27]:
            value_channel = F.hflip(value_channel)

        label = F.hflip(label)
    if random.random() > 0.5:
        if args.channels[0]:
            vv_channel = F.vflip(vv_channel)
        if args.channels[1]:
            vh_channel = F.vflip(vh_channel)
        if args.channels[2]:
            dem_channel = F.vflip(dem_channel)
        if args.channels[3]:
            gnorm_channel = F.vflip(gnorm_channel)
        if args.channels[4]:
            gy_channel = F.vflip(gy_channel)
        if args.channels[5]:
            gx_channel = F.vflip(gx_channel)
        if args.channels[6]:
            laplacian_channel = F.vflip(laplacian_channel)
        if args.channels[7]:
            flow_metric_channel = F.vflip(flow_metric_channel)
        if args.channels[8]:
            coastal_channel = F.vflip(coastal_channel)
        if args.channels[9]:
            blue_channel = F.vflip(blue_channel)
        if args.channels[10]:
            green_channel = F.vflip(green_channel)
        if args.channels[11]:
            red_channel = F.vflip(red_channel)
        if args.channels[12]:
            rededge1_channel = F.vflip(rededge1_channel)
        if args.channels[13]:
            rededge2_channel = F.vflip(rededge2_channel)
        if args.channels[14]:
            rededge3_channel = F.vflip(rededge3_channel)
        if args.channels[15]:
            nir_channel = F.vflip(nir_channel)
        if args.channels[16]:
            narrownir_channel = F.vflip(narrownir_channel)
        if args.channels[17]:
            watervapor_channel = F.vflip(watervapor_channel)
        if args.channels[18]:
            cirrus_channel = F.vflip(cirrus_channel)
        if args.channels[19]:
            swir1_channel = F.vflip(swir1_channel)
        if args.channels[20]:
            swir2_channel = F.vflip(swir2_channel)
        if args.channels[21]:
            ndwi_channel = F.vflip(ndwi_channel)
        if args.channels[22]:
            mndwi_channel = F.vflip(mndwi_channel)
        if args.channels[23]:
            awei_channel = F.vflip(awei_channel)
        if args.channels[24]:
            aweish_channel = F.vflip(aweish_channel)

        if args.channels[25]:
            hue_channel = F.vflip(hue_channel)
        if args.channels[26]:
            saturation_channel = F.vflip(saturation_channel)
        if args.channels[27]:
            value_channel = F.vflip(value_channel)
        label = F.vflip(label)

    norm = transforms.Normalize(means, stds)
    stack_list = list()
    if args.channels[0]:
        vv_channel = transforms.ToTensor()(vv_channel).squeeze()
        stack_list.append(vv_channel)
    if args.channels[1]:
        vh_channel = transforms.ToTensor()(vh_channel).squeeze()
        stack_list.append(vh_channel)
    if args.channels[2]:
        dem_channel = transforms.ToTensor()(dem_channel).squeeze()
        stack_list.append(dem_channel)
    if args.channels[3]:
        gnorm_channel = transforms.ToTensor()(gnorm_channel).squeeze()
        stack_list.append(gnorm_channel)
    if args.channels[4]:
        gy_channel = transforms.ToTensor()(gy_channel).squeeze()
        stack_list.append(gy_channel)
    if args.channels[5]:
        gx_channel = transforms.ToTensor()(gx_channel).squeeze()
        stack_list.append(gx_channel)
    if args.channels[6]:
        laplacian_channel = transforms.ToTensor()(laplacian_channel).squeeze()
        stack_list.append(laplacian_channel)
    if args.channels[7]:
        flow_metric_channel = transforms.ToTensor()(flow_metric_channel).squeeze()
        stack_list.append(flow_metric_channel)
    if args.channels[8]:
        coastal_channel = transforms.ToTensor()(coastal_channel).squeeze()
        stack_list.append(coastal_channel)
    if args.channels[9]:
        blue_channel = transforms.ToTensor()(blue_channel).squeeze()
        stack_list.append(blue_channel)
    if args.channels[10]:
        green_channel = transforms.ToTensor()(green_channel).squeeze()
        stack_list.append(green_channel)
    if args.channels[11]:
        red_channel = transforms.ToTensor()(red_channel).squeeze()
        stack_list.append(red_channel)
    if args.channels[12]:
        rededge1_channel = transforms.ToTensor()(rededge1_channel).squeeze()
        stack_list.append(rededge1_channel)
    if args.channels[13]:
        rededge2_channel = transforms.ToTensor()(rededge2_channel).squeeze()
        stack_list.append(rededge2_channel)
    if args.channels[14]:
        rededge3_channel = transforms.ToTensor()(rededge3_channel).squeeze()
        stack_list.append(rededge3_channel)
    if args.channels[15]:
        nir_channel = transforms.ToTensor()(nir_channel).squeeze()
        stack_list.append(nir_channel)
    if args.channels[16]:
        narrownir_channel = transforms.ToTensor()(narrownir_channel).squeeze()
        stack_list.append(narrownir_channel)
    if args.channels[17]:
        watervapor_channel = transforms.ToTensor()(watervapor_channel).squeeze()
        stack_list.append(watervapor_channel)
    if args.channels[18]:
        cirrus_channel = transforms.ToTensor()(cirrus_channel).squeeze()
        stack_list.append(cirrus_channel)
    if args.channels[19]:
        swir1_channel = transforms.ToTensor()(swir1_channel).squeeze()
        stack_list.append(swir1_channel)
    if args.channels[20]:
        swir2_channel = transforms.ToTensor()(swir2_channel).squeeze()
        stack_list.append(swir2_channel)
    if args.channels[21]:
        ndwi_channel = transforms.ToTensor()(ndwi_channel).squeeze()
        stack_list.append(ndwi_channel)
    if args.channels[22]:
        mndwi_channel = transforms.ToTensor()(mndwi_channel).squeeze()
        stack_list.append(mndwi_channel)
    if args.channels[23]:
        awei_channel = transforms.ToTensor()(awei_channel).squeeze()
        stack_list.append(awei_channel)
    if args.channels[24]:
        aweish_channel = transforms.ToTensor()(aweish_channel).squeeze()
        stack_list.append(aweish_channel)

    if args.channels[25]:
        hue_channel = transforms.ToTensor()(hue_channel).squeeze()
        stack_list.append(hue_channel)
    if args.channels[26]:
        saturation_channel = transforms.ToTensor()(saturation_channel).squeeze()
        stack_list.append(saturation_channel)
    if args.channels[27]:
        value_channel = transforms.ToTensor()(value_channel).squeeze()
        stack_list.append(value_channel)
    im = torch.stack(stack_list)
    im = norm(im)
    label = transforms.ToTensor()(label).squeeze()
    if torch.sum(label.gt(.003) * label.lt(.004)):
        label *= 255
    return im, label


def processTestIm(data):
    (x, y, x_y_coord, depressed_dem, flow, depressed_flow) = data
    im, label, x_y_coord = x.copy(), y.copy(), x_y_coord
    norm = transforms.Normalize(means, stds)

    # [VV, VH, DEM, Gnorm, Gy, Gx, Laplacian]
    # convert to PIL for easier transforms
    if args.channels[0]:
        vv_channel = Image.fromarray(im[0]).resize((512, 512))
    if args.channels[1]:
        vh_channel = Image.fromarray(im[1]).resize((512, 512))
    if not args.depression:
        processed_dem = im[2]
    elif args.depression:
        processed_dem = depressed_dem
    # compute 1st & 2nd order derivative of DEM
    # train_data got created from train_data.append((arr_x, arr_y, coords))
    if args.gradient == 'degree':
        Gy, Gx, Gnorm, Laplacian = calc_derivates_deg(processed_dem)
    elif args.gradient == 'meter':
        Gy, Gx, Gnorm, Laplacian = calc_derivates_meter(processed_dem, x_y_coord)
    else:
        sys.exit('WRONG GRADIENT')
    if args.channels[2]:
        dem_channel = Image.fromarray(processed_dem).resize((512, 512))
    if args.channels[3]:
        gnorm_channel = Image.fromarray(Gnorm).resize((512, 512))
    if args.channels[4]:
        gy_channel = Image.fromarray(Gy).resize((512, 512))
    if args.channels[5]:
        gx_channel = Image.fromarray(Gx).resize((512, 512))
    if args.channels[6]:
        laplacian_channel = Image.fromarray(Laplacian).resize((512, 512))
    if args.channels[7]:
        if not args.depression:
            flow_metric_channel = flow
        if args.depression:
            flow_metric_channel = depressed_flow
        flow_metric_channel = Image.fromarray(flow_metric_channel).resize((512, 512))
    if args.channels[8]:
        coastal_channel = Image.fromarray(im[3]).resize((512, 512))
    if args.channels[9]:
        blue_channel = Image.fromarray(im[4]).resize((512, 512))
    if args.channels[10]:
        green_channel = Image.fromarray(im[5]).resize((512, 512))
    if args.channels[11]:
        red_channel = Image.fromarray(im[6]).resize((512, 512))
    if args.channels[12]:
        rededge1_channel = Image.fromarray(im[7]).resize((512, 512))
    if args.channels[13]:
        rededge2_channel = Image.fromarray(im[8]).resize((512, 512))
    if args.channels[14]:
        rededge3_channel = Image.fromarray(im[9]).resize((512, 512))
    if args.channels[15]:
        nir_channel = Image.fromarray(im[10]).resize((512, 512))
    if args.channels[16]:
        narrownir_channel = Image.fromarray(im[11]).resize((512, 512))
    if args.channels[17]:
        watervapor_channel = Image.fromarray(im[12]).resize((512, 512))
    if args.channels[18]:
        cirrus_channel = Image.fromarray(im[13]).resize((512, 512))
    if args.channels[19]:
        swir1_channel = Image.fromarray(im[14]).resize((512, 512))
    if args.channels[20]:
        swir2_channel = Image.fromarray(im[15]).resize((512, 512))
    if args.channels[21]:
        ndwi_channel = Image.fromarray(im[16]).resize((512, 512))
    if args.channels[22]:
        mndwi_channel = Image.fromarray(im[17]).resize((512, 512))
    if args.channels[23]:
        awei_channel = Image.fromarray(im[18]).resize((512, 512))
    if args.channels[24]:
        aweish_channel = Image.fromarray(im[19]).resize((512, 512))

    if args.channels[25]:
        hue_channel = Image.fromarray(im[20]).resize((512, 512))
    if args.channels[26]:
        saturation_channel = Image.fromarray(im[21]).resize((512, 512))
    if args.channels[27]:
        value_channel = Image.fromarray(im[22]).resize((512, 512))

    label = Image.fromarray(label.squeeze()).resize((512, 512))

    stack_list = list()
    if args.channels[0]:
        vv_channel = [F.crop(vv_channel, 0, 0, 256, 256), F.crop(vv_channel, 0, 256, 256, 256),
                      F.crop(vv_channel, 256, 0, 256, 256), F.crop(vv_channel, 256, 256, 256, 256)]
        vv_channel = [transforms.ToTensor()(elem).squeeze() for elem in vv_channel]
        stack_list.append(vv_channel)
    if args.channels[1]:
        vh_channel = [F.crop(vh_channel, 0, 0, 256, 256), F.crop(vh_channel, 0, 256, 256, 256),
                      F.crop(vh_channel, 256, 0, 256, 256), F.crop(vh_channel, 256, 256, 256, 256)]
        vh_channel = [transforms.ToTensor()(elem).squeeze() for elem in vh_channel]
        stack_list.append(vh_channel)
    if args.channels[2]:
        dem_channel = [F.crop(dem_channel, 0, 0, 256, 256), F.crop(dem_channel, 0, 256, 256, 256),
                       F.crop(dem_channel, 256, 0, 256, 256), F.crop(dem_channel, 256, 256, 256, 256)]
        dem_channel = [transforms.ToTensor()(elem).squeeze() for elem in dem_channel]
        stack_list.append(dem_channel)
    if args.channels[3]:
        gnorm_channel = [F.crop(gnorm_channel, 0, 0, 256, 256), F.crop(gnorm_channel, 0, 256, 256, 256),
                         F.crop(gnorm_channel, 256, 0, 256, 256), F.crop(gnorm_channel, 256, 256, 256, 256)]
        gnorm_channel = [transforms.ToTensor()(elem).squeeze() for elem in gnorm_channel]
        stack_list.append(gnorm_channel)
    if args.channels[4]:
        gy_channel = [F.crop(gy_channel, 0, 0, 256, 256), F.crop(gy_channel, 0, 256, 256, 256),
                      F.crop(gy_channel, 256, 0, 256, 256), F.crop(gy_channel, 256, 256, 256, 256)]
        gy_channel = [transforms.ToTensor()(elem).squeeze() for elem in gy_channel]
        stack_list.append(gy_channel)
    if args.channels[5]:
        gx_channel = [F.crop(gx_channel, 0, 0, 256, 256), F.crop(gx_channel, 0, 256, 256, 256),
                      F.crop(gx_channel, 256, 0, 256, 256), F.crop(gx_channel, 256, 256, 256, 256)]
        gx_channel = [transforms.ToTensor()(elem).squeeze() for elem in gx_channel]
        stack_list.append(gx_channel)
    if args.channels[6]:
        laplacian_channel = [F.crop(laplacian_channel, 0, 0, 256, 256), F.crop(laplacian_channel, 0, 256, 256, 256),
                             F.crop(laplacian_channel, 256, 0, 256, 256), F.crop(laplacian_channel, 256, 256, 256, 256)]
        laplacian_channel = [transforms.ToTensor()(elem).squeeze() for elem in laplacian_channel]
        stack_list.append(laplacian_channel)
    if args.channels[7]:
        flow_metric_channel = [F.crop(flow_metric_channel, 0, 0, 256, 256), F.crop(flow_metric_channel, 0, 256, 256, 256),
                             F.crop(flow_metric_channel, 256, 0, 256, 256), F.crop(flow_metric_channel, 256, 256, 256, 256)]
        flow_metric_channel = [transforms.ToTensor()(elem).squeeze() for elem in flow_metric_channel]
        stack_list.append(flow_metric_channel)
    if args.channels[8]:
        coastal_channel = [F.crop(coastal_channel, 0, 0, 256, 256), F.crop(coastal_channel, 0, 256, 256, 256),
                             F.crop(coastal_channel, 256, 0, 256, 256), F.crop(coastal_channel, 256, 256, 256, 256)]
        coastal_channel = [transforms.ToTensor()(elem).squeeze() for elem in coastal_channel]
        stack_list.append(coastal_channel)
    if args.channels[9]:
        blue_channel = [F.crop(blue_channel, 0, 0, 256, 256), F.crop(blue_channel, 0, 256, 256, 256),
                             F.crop(blue_channel, 256, 0, 256, 256), F.crop(blue_channel, 256, 256, 256, 256)]
        blue_channel = [transforms.ToTensor()(elem).squeeze() for elem in blue_channel]
        stack_list.append(blue_channel)
    if args.channels[10]:
        green_channel = [F.crop(green_channel, 0, 0, 256, 256), F.crop(green_channel, 0, 256, 256, 256),
                             F.crop(green_channel, 256, 0, 256, 256), F.crop(green_channel, 256, 256, 256, 256)]
        green_channel = [transforms.ToTensor()(elem).squeeze() for elem in green_channel]
        stack_list.append(green_channel)
    if args.channels[11]:
        red_channel = [F.crop(red_channel, 0, 0, 256, 256), F.crop(red_channel, 0, 256, 256, 256),
                             F.crop(red_channel, 256, 0, 256, 256), F.crop(red_channel, 256, 256, 256, 256)]
        red_channel = [transforms.ToTensor()(elem).squeeze() for elem in red_channel]
        stack_list.append(red_channel)
    if args.channels[12]:
        rededge1_channel = [F.crop(rededge1_channel, 0, 0, 256, 256), F.crop(rededge1_channel, 0, 256, 256, 256),
                             F.crop(rededge1_channel, 256, 0, 256, 256), F.crop(rededge1_channel, 256, 256, 256, 256)]
        rededge1_channel = [transforms.ToTensor()(elem).squeeze() for elem in rededge1_channel]
        stack_list.append(rededge1_channel)
    if args.channels[13]:
        rededge2_channel = [F.crop(rededge2_channel, 0, 0, 256, 256), F.crop(rededge2_channel, 0, 256, 256, 256),
                             F.crop(rededge2_channel, 256, 0, 256, 256), F.crop(rededge2_channel, 256, 256, 256, 256)]
        rededge2_channel = [transforms.ToTensor()(elem).squeeze() for elem in rededge2_channel]
        stack_list.append(rededge2_channel)
    if args.channels[14]:
        rededge3_channel = [F.crop(rededge3_channel, 0, 0, 256, 256), F.crop(rededge3_channel, 0, 256, 256, 256),
                             F.crop(rededge3_channel, 256, 0, 256, 256), F.crop(rededge3_channel, 256, 256, 256, 256)]
        rededge3_channel = [transforms.ToTensor()(elem).squeeze() for elem in rededge3_channel]
        stack_list.append(rededge3_channel)
    if args.channels[15]:
        nir_channel = [F.crop(nir_channel, 0, 0, 256, 256), F.crop(nir_channel, 0, 256, 256, 256),
                             F.crop(nir_channel, 256, 0, 256, 256), F.crop(nir_channel, 256, 256, 256, 256)]
        nir_channel = [transforms.ToTensor()(elem).squeeze() for elem in nir_channel]
        stack_list.append(nir_channel)
    if args.channels[16]:
        narrownir_channel = [F.crop(narrownir_channel, 0, 0, 256, 256), F.crop(narrownir_channel, 0, 256, 256, 256),
                             F.crop(narrownir_channel, 256, 0, 256, 256), F.crop(narrownir_channel, 256, 256, 256, 256)]
        narrownir_channel = [transforms.ToTensor()(elem).squeeze() for elem in narrownir_channel]
        stack_list.append(narrownir_channel)
    if args.channels[17]:
        watervapor_channel = [F.crop(watervapor_channel, 0, 0, 256, 256), F.crop(watervapor_channel, 0, 256, 256, 256),
                             F.crop(watervapor_channel, 256, 0, 256, 256), F.crop(watervapor_channel, 256, 256, 256, 256)]
        watervapor_channel = [transforms.ToTensor()(elem).squeeze() for elem in watervapor_channel]
        stack_list.append(watervapor_channel)
    if args.channels[18]:
        cirrus_channel = [F.crop(cirrus_channel, 0, 0, 256, 256), F.crop(cirrus_channel, 0, 256, 256, 256),
                             F.crop(cirrus_channel, 256, 0, 256, 256), F.crop(cirrus_channel, 256, 256, 256, 256)]
        cirrus_channel = [transforms.ToTensor()(elem).squeeze() for elem in cirrus_channel]
        stack_list.append(cirrus_channel)
    if args.channels[19]:
        swir1_channel = [F.crop(swir1_channel, 0, 0, 256, 256), F.crop(swir1_channel, 0, 256, 256, 256),
                             F.crop(swir1_channel, 256, 0, 256, 256), F.crop(swir1_channel, 256, 256, 256, 256)]
        swir1_channel = [transforms.ToTensor()(elem).squeeze() for elem in swir1_channel]
        stack_list.append(swir1_channel)
    if args.channels[20]:
        swir2_channel = [F.crop(swir2_channel, 0, 0, 256, 256), F.crop(swir2_channel, 0, 256, 256, 256),
                             F.crop(swir2_channel, 256, 0, 256, 256), F.crop(swir2_channel, 256, 256, 256, 256)]
        swir2_channel = [transforms.ToTensor()(elem).squeeze() for elem in swir2_channel]
        stack_list.append(swir2_channel)
    if args.channels[21]:
        ndwi_channel = [F.crop(ndwi_channel, 0, 0, 256, 256), F.crop(ndwi_channel, 0, 256, 256, 256),
                             F.crop(ndwi_channel, 256, 0, 256, 256), F.crop(ndwi_channel, 256, 256, 256, 256)]
        ndwi_channel = [transforms.ToTensor()(elem).squeeze() for elem in ndwi_channel]
        stack_list.append(ndwi_channel)
    if args.channels[22]:
        mndwi_channel = [F.crop(mndwi_channel, 0, 0, 256, 256), F.crop(mndwi_channel, 0, 256, 256, 256),
                             F.crop(mndwi_channel, 256, 0, 256, 256), F.crop(mndwi_channel, 256, 256, 256, 256)]
        mndwi_channel = [transforms.ToTensor()(elem).squeeze() for elem in mndwi_channel]
        stack_list.append(mndwi_channel)
    if args.channels[23]:
        awei_channel = [F.crop(awei_channel, 0, 0, 256, 256), F.crop(awei_channel, 0, 256, 256, 256),
                             F.crop(awei_channel, 256, 0, 256, 256), F.crop(awei_channel, 256, 256, 256, 256)]
        awei_channel = [transforms.ToTensor()(elem).squeeze() for elem in awei_channel]
        stack_list.append(awei_channel)
    if args.channels[24]:
        aweish_channel = [F.crop(aweish_channel, 0, 0, 256, 256), F.crop(aweish_channel, 0, 256, 256, 256),
                             F.crop(aweish_channel, 256, 0, 256, 256), F.crop(aweish_channel, 256, 256, 256, 256)]
        aweish_channel = [transforms.ToTensor()(elem).squeeze() for elem in aweish_channel]
        stack_list.append(aweish_channel)

    if args.channels[25]:
        hue_channel = [F.crop(hue_channel, 0, 0, 256, 256), F.crop(hue_channel, 0, 256, 256, 256),
                             F.crop(hue_channel, 256, 0, 256, 256), F.crop(hue_channel, 256, 256, 256, 256)]
        hue_channel = [transforms.ToTensor()(elem).squeeze() for elem in hue_channel]
        stack_list.append(hue_channel)
    if args.channels[26]:
        saturation_channel = [F.crop(saturation_channel, 0, 0, 256, 256), F.crop(saturation_channel, 0, 256, 256, 256),
                             F.crop(saturation_channel, 256, 0, 256, 256), F.crop(saturation_channel, 256, 256, 256, 256)]
        saturation_channel = [transforms.ToTensor()(elem).squeeze() for elem in saturation_channel]
        stack_list.append(saturation_channel)
    if args.channels[27]:
        value_channel = [F.crop(value_channel, 0, 0, 256, 256), F.crop(value_channel, 0, 256, 256, 256),
                             F.crop(value_channel, 256, 0, 256, 256), F.crop(value_channel, 256, 256, 256, 256)]
        value_channel = [transforms.ToTensor()(elem).squeeze() for elem in value_channel]
        stack_list.append(value_channel)

    labels = [F.crop(label, 0, 0, 256, 256), F.crop(label, 0, 256, 256, 256),
              F.crop(label, 256, 0, 256, 256), F.crop(label, 256, 256, 256, 256)]

    ims = list()
    for crop_idx in range(4):
        tmp = torch.stack([elem[crop_idx] for elem in stack_list])
        ims.append(tmp)


    ims = [norm(im) for im in ims]
    ims = torch.stack(ims)

    labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
    labels = torch.stack(labels)

    if torch.sum(labels.gt(.003) * labels.lt(.004)):
        labels *= 255

    return ims, labels


def getArrFlood(fname):
    return rasterio.open(fname).read()


def getDem(fname):
    return rasterio.open(fname).read()


def download_flood_water_data_from_list(l):
    i = 0
    tot_nan = 0
    tot_good = 0
    flood_data = []
    for (im_fname, mask_fname, dem_fname, s2_name) in l:
        if not os.path.exists(im_fname):
            continue
        arr_x = np.nan_to_num(getArrFlood(im_fname))
        arr_s2 = np.nan_to_num(getArrFlood(s2_name))
        arr_d = getDem(dem_fname)
        arr_y = getArrFlood(mask_fname)
        arr_y[arr_y == -1] = 255

        arr_x = np.clip(arr_x, -50, 1)
        arr_x = (arr_x + 50) / 51

        depressed_dem = rd.FillDepressions(rd.rdarray(np.array(arr_d[0]), no_data=-9999), in_place=False)
        flow = rd.FlowAccumulation(rd.rdarray(np.array(arr_d[0]), no_data=-9999), method=args.flow)
        depressed_flow = rd.FlowAccumulation(depressed_dem, method=args.flow)

        ndwi = (arr_s2[2]-arr_s2[7])/(arr_s2[2]+arr_s2[7])
        ndwi = ndwi.reshape((1, ndwi.shape[0], ndwi.shape[0]))
        mndwi = (arr_s2[2]-arr_s2[11])/(arr_s2[2]+arr_s2[11])
        mndwi = mndwi.reshape((1, mndwi.shape[0], mndwi.shape[0]))
        awei = 4*(arr_s2[2]-arr_s2[11])-((1/4)*(arr_s2[7]+(11*arr_s2[12])))
        awei = awei.reshape((1, awei.shape[0], awei.shape[0]))
        aweish = arr_s2[1] + ((5/2)*arr_s2[2])-((3/2)*(arr_s2[7]+arr_s2[11]))-(arr_s2[12]/4)
        aweish = aweish.reshape((1, aweish.shape[0], aweish.shape[0]))

        arr_x = np.concatenate([arr_x, arr_d, arr_s2, ndwi, mndwi, awei, aweish], axis=0)

        hsv_channels = get_hsv_channels(arr_x[args.hsv_channels[0]], arr_x[args.hsv_channels[1]], arr_x[args.hsv_channels[2]])
        hue = hsv_channels[0].reshape((1, hsv_channels.shape[1], hsv_channels.shape[2]))
        saturation = hsv_channels[1].reshape((1, hsv_channels.shape[1], hsv_channels.shape[2]))
        value = hsv_channels[2].reshape((1, hsv_channels.shape[1], hsv_channels.shape[2]))
        arr_x = np.concatenate([arr_x, hue, saturation, value], axis=0)
        coords = get_projection_coordinates(im_fname)

        i += 1
        flood_data.append((arr_x, arr_y, coords, depressed_dem, flow, depressed_flow))
    return flood_data


def get_hsv_channels(r_channel, g_channel, b_channel):
    width = r_channel.shape[0]
    height = r_channel.shape[1]
    res = np.zeros(shape=(3, r_channel.shape[0], r_channel.shape[1]))
    for w in range(width):
        for h in range(height):
            hsv = colorsys.rgb_to_hsv(r_channel[w, h], g_channel[w, h], b_channel[w, h])
            res[0, w, h] = hsv[0]
            res[1, w, h] = hsv[1]
            res[2, w, h] = hsv[2]
    return res

def get_projection_coordinates(fname):
    lon, lat = get_wgs84_coord(fname)
    centerx, centery = (np.mean(lon), np.mean(lat))
    epsg_utm = epsg_utm_from_wgs84(centerx, centery)
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:{}".format(epsg_utm), always_xy=True)
    x, y = transformer.transform(lon, lat)
    return (x, y)


def load_flood_train_data(input_root, label_root, dem_root, s2_root, input_root_weak, label_root_weak, dem_root_weak, s2_root_weak):
    fname = "flood_train_data.csv"
    fname_weak = "flood_train_data_weak.csv"
    training_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            folder_dem = line[0][:-11]
            file_dem = folder_dem + '_dem.tif'
            line_s2 = line[0]
            # switches s1 to s2
            line_s2 = line_s2[:-9] + '2' + line_s2[-8:]
            training_files.append(tuple((os.path.join(input_root, line[0]), os.path.join(label_root, line[1]),
                                         os.path.join(dem_root, folder_dem, file_dem),
                                         os.path.join(s2_root, line_s2))))
    if args.weakly_flag:
        with open(fname_weak) as f:
            for line in csv.reader(f):
                folder_dem = line[0][:-20]
                file_dem = folder_dem + '_dem.tif'
                line_s2 = line[2]
                training_files.append(tuple((os.path.join(input_root_weak, line[1]), os.path.join(label_root_weak, line[0]),
                                            os.path.join(dem_root_weak, folder_dem, file_dem),
                                            os.path.join(s2_root_weak, line_s2))))
    return download_flood_water_data_from_list(training_files)


def load_flood_valid_data(input_root, label_root, dem_root):
    fname = "flood_valid_data.csv"
    validation_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            folder_dem = '_'.join(line[0].split('.')[0].split('_')[:2])
            file_dem = folder_dem + '_dem.tif'
            line_s2 = line[0]
            line_s2 = line_s2[:-9] + '2' + line_s2[-8:]
            validation_files.append(tuple((os.path.join(input_root, line[0]), os.path.join(label_root, line[1]),
                                           os.path.join(dem_root, folder_dem, file_dem),
                                           os.path.join(s2_root, line_s2))))

    return download_flood_water_data_from_list(validation_files)


def load_flood_test_data(input_root, label_root, dem_root):
    fname = "flood_test_data.csv"
    testing_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            folder_dem = line[0][:-11]
            file_dem = folder_dem + '_dem.tif'
            line_s2 = line[0]
            line_s2 = line_s2[:-9] + '2' + line_s2[-8:]
            testing_files.append(tuple((os.path.join(input_root, line[0]), os.path.join(label_root, line[1]),
                                        os.path.join(dem_root, folder_dem, file_dem),
                                        os.path.join(s2_root, line_s2))))

    return download_flood_water_data_from_list(testing_files)


def convertBNtoGN(module, num_groups=16):
    if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        return nn.GroupNorm(num_groups, module.num_features,
                            eps=module.eps, affine=module.affine)
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        module.add_module(name, convertBNtoGN(child, num_groups=num_groups))

    return module


def computeIOU(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()

    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    intersection = torch.sum(output * target)
    union = torch.sum(target) + torch.sum(output) - intersection
    iou = (intersection + .0000001) / (union + .0000001)

    if iou != iou:
        print("failed, replacing with 0")
        iou = torch.tensor(0).float()

    return iou


def computeAccuracy(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()

    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    correct = torch.sum(output.eq(target))

    return correct.float() / len(target)


def truePositives(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    correct = torch.sum(output * target)

    return correct


def trueNegatives(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    output = (output == 0)
    target = (target == 0)
    correct = torch.sum(output * target)

    return correct


def falsePositives(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    output = (output == 1)
    target = (target == 0)
    correct = torch.sum(output * target)

    return correct


def falseNegatives(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    output = (output == 0)
    target = (target == 1)
    correct = torch.sum(output * target)

    return correct


def train_loop(inputs, labels, net, optimizer, scheduler):
    global running_loss
    global running_iou
    global running_count
    global running_accuracy

    # zero the parameter gradients
    optimizer.zero_grad()
    net = net.cuda()

    # forward + backward + optimize
    outputs = net(inputs.cuda())
    loss = criterion(outputs["out"], labels.long().cuda())
    loss.backward()
    optimizer.step()
    scheduler.step()

    running_loss += loss
    running_iou += computeIOU(outputs["out"], labels.cuda())
    running_accuracy += computeAccuracy(outputs["out"], labels.cuda())
    running_count += 1


def validation_loop(validation_data_loader, net):
    global running_loss
    global running_iou
    global running_count
    global running_accuracy
    global max_valid_iou

    global training_losses
    global training_accuracies
    global training_ious
    global valid_losses
    global valid_accuracies
    global valid_ious

    net = net.eval()
    net = net.cuda()
    count = 0
    iou = 0
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for (images, labels) in validation_data_loader:
            net = net.cuda()
            outputs = net(images.cuda())
            valid_loss = criterion(outputs["out"], labels.long().cuda())
            valid_iou = computeIOU(outputs["out"], labels.cuda())
            valid_accuracy = computeAccuracy(outputs["out"], labels.cuda())
            iou += valid_iou
            loss += valid_loss
            accuracy += valid_accuracy
            count += 1

    iou = iou / count
    accuracy = accuracy / count

    if iou > max_valid_iou:
        max_valid_iou = iou
        save_path = os.path.join(res_checkpoints, "{}_{}_{}.cp".format(RUNNAME, i, iou.item()))
        torch.save(net.state_dict(), save_path)
        print("model saved at", save_path)

    loss = loss / count
    print("Training Loss:", running_loss / running_count)
    print("Training IOU:", running_iou / running_count)
    print("Training Accuracy:", running_accuracy / running_count)
    print("Validation Loss:", loss)
    print("Validation IOU:", iou)
    print("Validation Accuracy:", accuracy)

    training_losses.append(running_loss / running_count)
    training_accuracies.append(running_accuracy / running_count)
    training_ious.append(running_iou / running_count)
    valid_losses.append(loss)
    valid_accuracies.append(accuracy)
    valid_ious.append(iou)


def test_loop(test_data_loader, net):
    net = net.eval()
    net = net.cuda()
    count = 0
    iou = 0
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for (images, labels) in tqdm(test_data_loader):
            net = net.cuda()
            outputs = net(images.cuda())
            #            valid_loss = criterion(outputs["out"], labels.long().cuda())
            valid_iou = computeIOU(outputs["out"], labels.cuda())
            iou += valid_iou
            accuracy += computeAccuracy(outputs["out"], labels.cuda())
            count += 1

    iou = iou / count
    print("Test IOU:", iou)
    print("Test Accuracy:", accuracy / count)
    return iou


def train_epoch(net, optimizer, scheduler, train_iter):
    for (inputs, labels) in tqdm(train_iter):
        train_loop(inputs.cuda(), labels.cuda(), net.cuda(), optimizer, scheduler)


def train_validation_loop(net, optimizer, scheduler, train_loader,
                          valid_loader, num_epochs, cur_epoch):
    global running_loss
    global running_iou
    global running_count
    global running_accuracy
    net = net.train()
    running_loss = 0
    running_iou = 0
    running_count = 0
    running_accuracy = 0

    for i in tqdm(range(num_epochs)):
        train_iter = iter(train_loader)
        train_epoch(net, optimizer, scheduler, train_iter)
    # clear_output()

    print("Current Epoch:", cur_epoch)
    validation_loop(iter(valid_loader), net)


def set_seed(se):
    """
    Sets the seed to have reproducible results

    Parameters
    ----------
    se: Integer, defining the seed value
    """

    torch.manual_seed(se)
    torch.cuda.manual_seed_all(se)
    torch.cuda.manual_seed(se)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(se)
    random.seed(se)


def parse_arugments():
    parser = argparse.ArgumentParser(description=
                                     'Run Sen1Floods11 code')
    parser.add_argument('-cf', '--configfile', dest='config', type=str, required=False,
                        help='Path to configuration file')
    parser.add_argument('-n', '--runs', dest='n_runs', type=int, required=False,
                        help='number of runs to be performed with different seeds')
    parser.add_argument('-s', '--source', dest='source', type=str, required=False,
                        help='Define which DEM datasource should '
                             'be used. Available options are:'
                             '\'SRTM\' and \'ASTER\'')
    parser.add_argument('-c', '--channels', dest='channels', type=int, required=False, nargs=28,
                        help='Define which channels should be used for training.\n'
                             'Example usage: 1, 1, 1, 0, 1, 1, 1, 1 where the positional int'
                             'corresponding to:\n'
                             'VV, VH, DEM, Gnorm, Gy, Gx, Laplacian, flow metric,\n'
                             'coastal, blue, green, red, redEdge1, redEdge2, RedEdge3, NIR, Narrow NIR, Water Vapor, Cirrus,'
                             'SWIR-1, SWIR-2, NDWI, MNDWI, AWEI, AWEISH, Hue, Saturation, Value.'
                             '1 is True, 0 is False')
    parser.add_argument('-hsv', '--hsv_channels', dest='hsv_channels', type=int, required=False, nargs=3,
                        help='Define the channel indices used for hsv transformation. Indices can be acquired from method get_channel_id()'
                        'e.g. 20, 15, 11 for \'SWIR-2\', \'NIR\', \'Red\'')
    parser.add_argument('-g', '--gradient', dest='gradient', type=str, required=False,
                        help='Define which gradient representation to use. Choose between:'
                             '\'degree\' and \'meter\'')
    parser.add_argument('-d', '--depression', dest='depression', type=int, required=False,
                        help='Set if depression filtering should be used (0 for no, 1 for yes)')
    parser.add_argument('-f', '--flow_direction', dest='flow', type=str, required=False,
                        help='Set which flow direction should be used')
                        # possible values are the method names given here: https://richdem.readthedocs.io/en/latest/flow_metrics.html
                        # need to give None if you don't want to use it
    parser.add_argument('-r', '--run_id', dest='run_id', type=int, required=False,
                        help='Set the id of the result folder')
    parser.add_argument('-w', '--weakly_flag', dest='weakly_flag', type=int, required=False,
                        help='set if weakly data should be used for training (0 for no, 1 for yes)')
    args = parser.parse_args()
    if args.config is not None:
        print('read argument values from file')
        args = read_arg_json(args.config, args)
    else:
        print('get arguments from command line')
        if args.n_runs is None:
            print('Missing Argument -n')
        if args.source is None:
            print('Missing Argument -s')
        if args.channels is None:
            print('Missing Argument -c')
        if args.gradient is None:
            print('Missing Argument -g')
        if args.depression is None:
            print('Missing Argument -d')
        if args.flow is None:
            print('Missing Argument -f')
        if args.run_id is None:
            print('Missing Argument -r')
        if args.weakly_flag is None:
            print('Missing Argument -w')
        if args.hsv_channels is None:
            print('Missing Argument -hsv')
    if args.flow == "None":
        args.flow = None
    args.channels = [True if elem == 1 else False for elem in args.channels]
    args.depression = True if args.depression == 1 else False
    args.weakly_flag = True if args.weakly_flag == 1 else False
    return args


def read_arg_json(file_path, arguments):
    print(file_path)
    with open(file_path, 'r') as f:
      data = json.load(f)
      arguments.n_runs = int(data['n_runs'])
      arguments.source = data['source']
      arguments.gradient = data['gradient']
      arguments.depression = int(data['depression'])
      arguments.flow = data['flow_direction']
      arguments.run_id = int(data['run_id'])
      arguments.channels = make_channel_array(data['channels'])
      arguments.hsv_channels = make_hsv_channel_array(data['HSV_channels'])
      arguments.weakly_flag = data['weakly_flag']
      return arguments

def make_hsv_channel_array(hsv_array_string):
    return [get_channel_id(hsv_array_string[0]), get_channel_id(hsv_array_string[1]), get_channel_id(hsv_array_string[2])]

def make_channel_array(c_dict):
    return [int(c_dict["VV"]), int(c_dict["VH"]), int(c_dict["DEM"]), int(c_dict["Gnorm"]), int(c_dict["Gy"]), int(c_dict["Gx"]),
            int(c_dict["Laplacian"]), int(c_dict["flow metric"]), int(c_dict["coastal"]), int(c_dict["blue"]), int(c_dict["green"]),
            int(c_dict["red"]), int(c_dict["redEdge1"]), int(c_dict["redEdge2"]), int(c_dict["RedEdge3"]), int(c_dict["NIR"]),
            int(c_dict["Narrow NIR"]), int(c_dict["Cirrus"]), int(c_dict["Water Vapor"]), int(c_dict["SWIR-1"]), int(c_dict["SWIR-2"]),
            int(c_dict["NDWI"]), int(c_dict["MNDWI"]), int(c_dict["AWEI"]), int(c_dict["AWEISH"]), int(c_dict["Hue"]),
            int(c_dict["Saturation"]), int(c_dict["Value"])]


def get_run_id(run_res_path):
    folders = []
    for name in os.listdir(run_res_path):
        folder = os.path.join(run_res_path, name)
        if os.path.isdir(folder):
            folders.append(folder)
    if len(folders) > 0:
        ids = [int(folder_name.split('_')[-1]) for folder_name in folders]
        ids.sort()
        highest_id = ids[-1]
    else:
        highest_id = -1
    return highest_id + 1


def make_res_folder(run_res_path, run_id):
    os.mkdir(os.path.join(run_res_path, f'run_{run_id}'))
    os.mkdir(os.path.join(run_res_path, f'run_{run_id}', 'checkpoints'))
    return os.path.join(run_res_path, f'run_{run_id}', 'res.csv'), os.path.join(run_res_path, f'run_{run_id}',
                                                                                'checkpoints'), os.path.join(
        run_res_path, f'run_{run_id}', 'plot.png')


def calc_derivates(im_data):
    Gy, Gx = np.gradient(im_data, 1., 1.)
    Gnorm = (Gx ** 2 + Gy ** 2) ** 0.5
    Laplacian = laplace(im_data)
    return Gy, Gx, Gnorm, Laplacian


def calc_derivates_deg(im_data):
    Gy, Gx = np.gradient(im_data, 0.000089315, 0.000089315)
    Gnorm = (Gx ** 2 + Gy ** 2) ** 0.5
    Laplacian = laplace(im_data)
    return Gy, Gx, Gnorm, Laplacian


def calc_derivates_meter(im_data, coords):
    x, y = coords
    dx_y, dx_x = np.gradient(x)
    dy_y, dy_x = np.gradient(y)

    dx = np.mean(dx_x)
    dy = np.mean(dy_y)

    Gy, Gx = np.gradient(im_data, dy, dx)
    Gnorm = (Gx ** 2 + Gy ** 2) ** 0.5
    Laplacian = laplace(im_data)
    return Gy, Gx, Gnorm, Laplacian


def calc_train_mean_std(train_data):
    ch0 = list()  # vv
    ch1 = list()  # vh
    ch2 = list()  # dem
    ch3 = list()  # gnorm
    ch4 = list()  # Laplacian
    ch_gx = list()  # gx
    ch_gy = list()  # gy
    ch_flow = list() # flow metrics
    ch_coastal = list()
    ch_blue = list()
    ch_green = list()
    ch_red = list()
    ch_rededge1 = list()
    ch_rededge2 = list()
    ch_rededge3 = list()
    ch_nir = list()
    ch_narrownir = list()
    ch_watervapor = list()
    ch_cirrus = list()
    ch_swir1 = list()
    ch_swir2 = list()
    ch_ndwi = list()
    ch_mndwi = list()
    ch_awei = list()
    ch_aweish = list()
    ch_hue = list()
    ch_saturation = list()
    ch_value = list()
    # train_data got created from train_data.append((arr_x, arr_y, coords, depressed_dem, flow, depressed_flow))
    # with arr_x = [s1 (2 channels), dem (1 channel), s2 (13 channels), ndwi, mndwi, awei, aweish, hue, saturation, value]
    for elem in train_data:
        im_data = elem[0]
        sar_channel0 = im_data[0]
        sar_channel1 = im_data[1]
        if not args.depression:
            dem_channel = im_data[2]
        elif args.depression:
            dem_channel = elem[3]
        ch0.append(sar_channel0)
        ch1.append(sar_channel1)
        ch2.append(dem_channel)
        coords = elem[2]
        if args.gradient == 'degree':
            Gy, Gx, Gnorm, Laplacian = calc_derivates_deg(dem_channel)
        elif args.gradient == 'meter':
            Gy, Gx, Gnorm, Laplacian = calc_derivates_meter(dem_channel, coords)
        else:
            sys.exit('WRONG GRADIENT')
        ch3.append(Gnorm)
        ch4.append(Laplacian)
        ch_gy.append(Gy)
        ch_gx.append(Gx)
        if args.flow is not None:
            flow_metric = elem[5] if args.depression else elem[4]
            ch_flow.append(flow_metric)
        if args.channels[7] and args.flow is None:
            sys.exit('Flow metric is chosen, but flow metric is set to None')
        # https://github.com/cloudtostreet/Sen1Floods11#dataset-information
        ch_coastal.append(im_data[3])
        ch_blue.append(im_data[4])
        ch_green.append(im_data[5])
        ch_red.append(im_data[6])
        ch_rededge1.append(im_data[7])
        ch_rededge2.append(im_data[8])
        ch_rededge3.append(im_data[9])
        ch_nir.append(im_data[10])
        ch_narrownir.append(im_data[11])
        ch_watervapor.append(im_data[12])
        ch_cirrus.append(im_data[13])
        ch_swir1.append(im_data[14])
        ch_swir2.append(im_data[15])
        ch_ndwi.append(im_data[16])
        ch_mndwi.append(im_data[17])
        ch_awei.append(im_data[18])
        ch_aweish.append(im_data[19])
        ch_hue.append(im_data[20])
        ch_saturation.append(im_data[21])
        ch_value.append(im_data[22])


    mean_list = list()
    std_list = list()
    # [VV, VH, DEM, Gnorm, Gy, Gx, Laplacian, flow_metric]
    if args.channels[0]:  # VV
        mean_list.append(np.mean(ch0))
        std_list.append(np.std(ch0))
    if args.channels[1]:  # VH
        mean_list.append(np.mean(ch1))
        std_list.append(np.std(ch1))
    if args.channels[2]:  # DEM
        mean_list.append(np.mean(ch2))
        std_list.append(np.std(ch2))
    if args.channels[3]:  # Gnorm
        mean_list.append(np.mean(ch3))
        std_list.append(np.std(ch3))
    if args.channels[4]:  # Gy
        mean_list.append(np.mean(ch_gy))
        std_list.append(np.std(ch_gy))
    if args.channels[5]:  # Gx
        mean_list.append(np.mean(ch_gx))
        std_list.append(np.std(ch_gx))
    if args.channels[6]:  # Laplacian
        mean_list.append(np.mean(ch4))
        std_list.append(np.std(ch4))
    if args.channels[7]: # flow metric
        mean_list.append(np.mean(ch_flow))
        std_list.append(np.std(ch_flow))
    if args.channels[8]: # coastal
        mean_list.append(np.mean(ch_coastal))
        std_list.append(np.std(ch_coastal))
    if args.channels[9]: #blue
        mean_list.append(np.mean(ch_blue))
        std_list.append(np.std(ch_blue))
    if args.channels[10]: #green
        mean_list.append(np.mean(ch_green))
        std_list.append(np.std(ch_green))
    if args.channels[11]: #red
        mean_list.append(np.mean(ch_red))
        std_list.append(np.std(ch_red))
    if args.channels[12]: #redEdge1
        mean_list.append(np.mean(ch_rededge1))
        std_list.append(np.std(ch_rededge1))
    if args.channels[13]: #redEdge2
        mean_list.append(np.mean(ch_rededge2))
        std_list.append(np.std(ch_rededge2))
    if args.channels[14]: #RedEdge3
        mean_list.append(np.mean(ch_rededge3))
        std_list.append(np.std(ch_rededge3))
    if args.channels[15]: #nir
        mean_list.append(np.mean(ch_nir))
        std_list.append(np.std(ch_nir))
    if args.channels[16]: #narrow nir
        mean_list.append(np.mean(ch_narrownir))
        std_list.append(np.std(ch_narrownir))
    if args.channels[17]: #water vapor
        mean_list.append(np.mean(ch_watervapor))
        std_list.append(np.std(ch_watervapor))
    if args.channels[18]: #cirrus
        mean_list.append(np.mean(ch_cirrus))
        std_list.append(np.std(ch_cirrus))
    if args.channels[19]: #swir1
        mean_list.append(np.mean(ch_swir1))
        std_list.append(np.std(ch_swir1))
    if args.channels[20]: #swir2
        mean_list.append(np.mean(ch_swir2))
        std_list.append(np.std(ch_swir2))
    if args.channels[21]: #ndwi
        mean_list.append(np.mean(ch_ndwi))
        std_list.append(np.std(ch_ndwi))
    if args.channels[22]: #mndwi
        mean_list.append(np.mean(ch_mndwi))
        std_list.append(np.std(ch_mndwi))
    if args.channels[23]: #awei
        mean_list.append(np.mean(ch_awei))
        std_list.append(np.std(ch_awei))
    if args.channels[24]: #aweish
        mean_list.append(np.mean(ch_aweish))
        std_list.append(np.std(ch_aweish))

    if args.channels[25]: #hue
        mean_list.append(np.mean(ch_hue))
        std_list.append(np.std(ch_hue))
    if args.channels[26]: #saturation
        mean_list.append(np.mean(ch_saturation))
        std_list.append(np.std(ch_saturation))
    if args.channels[27]: #value
        mean_list.append(np.mean(ch_value))
        std_list.append(np.std(ch_value))

    return mean_list, std_list


def get_utm_coord(sar_root, fname):
    utm_list = []
    with open(fname) as f:
        for line in csv.reader(f):
            fname = os.path.join(sar_root, line[0])
            lon, lat = get_wgs84_coord(fname)
            centerx, centery = (np.mean(lon), np.mean(lat))
            epsg_utm = epsg_utm_from_wgs84(centerx, centery)
            transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:{}".format(epsg_utm), always_xy=True)
            x_coord, y_coord = transformer.transform(lon, lat)
            utm_list.append((x_coord, y_coord))
    return utm_list


def get_wgs84_coord(fname):
    sar = rasterio.open(fname)
    band1 = sar.read(1)
    height = band1.shape[0]
    width = band1.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(sar.transform, rows, cols)
    lons = np.array(xs)
    lats = np.array(ys)
    return lons, lats


def epsg_utm_from_wgs84(longitude: float, latitude: float):
    utm_zone = str((math.floor((longitude + 180) / 6) % 60) + 1)
    if len(utm_zone) == 1:
        utm_zone = '0' + utm_zone
    if latitude >= 0:
        epsg_code = '326' + utm_zone
        return epsg_code
    epsg_utm = '327' + utm_zone
    return epsg_utm


def compute_gradient(data, coord_list):
    nData = len(data)
    dem_list = []
    for i in range(nData):
        dem = data[i][0][2]
        x, y = coord_list[i]
        dx_y, dx_x = np.gradient(x)
        dy_y, dy_x = np.gradient(y)

        dx = np.mean(dx_x)
        dy = np.mean(dy_y)

        Gy, Gx = np.gradient(dem, dy, dx)
        dem_list.append((Gx, Gy))
    return dem_list


def compute_gradient_degree(data):
    nData = len(data)
    dem_list = []
    for i in range(nData):
        dem = data[i][0][2]
        Gy, Gx = np.gradient(dem, 0.000089315, 0.000089315)
        dem_list.append((Gx, Gy))
    return dem_list


if __name__ == '__main__':

    args = parse_arugments()
    run_res_path = os.path.join('.', 'run_res')
    run_id = args.run_id
    res_csv_path, res_checkpoints, plot_path = make_res_folder(run_res_path, run_id)
    seeds = random.sample(range(1, 1000), args.n_runs)

    f = open(res_csv_path, 'w')
    writer = csv.writer(f)
    header = ['seed', 'trainIoU', 'valIoU', 'testIoU']
    writer.writerow(header)
    f.close()
    arg_file = open(os.path.join(run_res_path, f'run_{run_id}', 'args.csv'), 'w')
    arg_writer = csv.writer(arg_file)
    parameters = ['n_runs', 'weakly_flag', 'DEM Source', 'Gradient', 'Depression', 'VV', "VH", "DEM", "Gnorm", "Gy", "Gx", "Laplacian",
                  "flow metric", "flow direction", "coastal", "blue", "green", "red", "redEdge1", "redEdge2", "RedEdge3", "NIR",
                  "Narrow NIR", "Cirrus", "Water Vapor", "SWIR-1", "SWIR-2", "NDWI", "MNDWI", "AWEI", "AWEISH",
                  "Hue", "Saturation", "Value", "HSV_channels", "run_id"]
    values = [args.n_runs, args.weakly_flag, args.source, args.gradient, args.depression,
              args.channels[0], args.channels[1], args.channels[2], args.channels[3], args.channels[4], args.channels[5],
              args.channels[6], args.channels[7], args.flow, args.channels[8], args.channels[9], args.channels[10],
              args.channels[11], args.channels[12], args.channels[13], args.channels[14], args.channels[15], args.channels[16],
              args.channels[17], args.channels[18], args.channels[19], args.channels[20], args.channels[20], args.channels[21],
              args.channels[22], args.channels[23], args.channels[24], args.channels[25], args.channels[26],
              get_hsv_names(args.hsv_channels), args.run_id]
    arg_writer.writerow(parameters)
    arg_writer.writerow(values)
    arg_file.close()

    #data loading
    s1_root = os.path.join('..', 'data', 'sen1floods11_data', 'v1.1', 'data', 'flood_events', 'HandLabeled',
                           'S1Hand')
    s2_root = os.path.join('..', 'data', 'sen1floods11_data', 'v1.1', 'data', 'flood_events', 'HandLabeled',
                           'S2Hand')
    labels_root = os.path.join('..', 'data', 'sen1floods11_data', 'v1.1', 'data', 'flood_events', 'HandLabeled',
                               'LabelHand')

    s1_weak_root = os.path.join('..', 'data', 'sen1floods11_data', 'v1.1', 'data', 'flood_events', 'WeaklyLabeled',
                           'S1Weak')
    s2_weak_root = os.path.join('..', 'data', 'sen1floods11_data', 'v1.1', 'data', 'flood_events', 'WeaklyLabeled',
                           'S2Weak')
    labels_weak_root = os.path.join('..', 'data', 'sen1floods11_data', 'v1.1', 'data', 'flood_events', 'WeaklyLabeled',
                               'S1OtsuLabelWeak')
    if args.source == 'ASTER':
        dem_root = os.path.join('..', 'data', 'final_out_hand')
        dem_root_weak = os.path.join('..', 'data', 'final_out_weak')
    elif args.source == 'SRTM':
        dem_root = os.path.join('..', 'data', 'final_hand_srtm')
        dem_root_weak = os.path.join('..', 'data', 'final_weak_srtm')
    else:
        sys.exit('INVALID SOURCE')

    print('Load Training')
    train_data = load_flood_train_data(s1_root, labels_root, dem_root, s2_root, s1_weak_root, labels_weak_root, dem_root_weak, s2_weak_root)
    means, stds = calc_train_mean_std(train_data)

    train_dataset = InMemoryDataset(train_data, processAndAugment)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, sampler=None,
                                               batch_sampler=None, num_workers=0, collate_fn=None,
                                               pin_memory=True, drop_last=False, timeout=0,
                                               worker_init_fn=None)
    train_iter = iter(train_loader)
    print('Load Validation')
    valid_data = load_flood_valid_data(s1_root, labels_root, dem_root)
    valid_dataset = InMemoryDataset(valid_data, processTestIm)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, sampler=None,
                                               batch_sampler=None, num_workers=0, collate_fn=lambda x: (
            torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
                                               pin_memory=True, drop_last=False, timeout=0,
                                               worker_init_fn=None)
    valid_iter = iter(valid_loader)

    print('Load Test Data')
    test_data = load_flood_test_data(s1_root, labels_root, dem_root)
    test_dataset = InMemoryDataset(test_data, processTestIm)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, sampler=None,
                                              batch_sampler=None, num_workers=0, collate_fn=lambda x: (
            torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
                                              pin_memory=True, drop_last=False, timeout=0,
                                              worker_init_fn=None)

    print('Data loaded')
    print('Start iterating over seeds')

    for seed_id in seeds:
        f = open(res_csv_path, 'a')
        writer = csv.writer(f)
        # Model Parameters
        set_seed(seed_id)
        LR = 5e-4
        EPOCHS = 30  # TODO change to 30
        EPOCHS_PER_UPDATE = 1
        RUNNAME = "Sen1Floods11"


        channel_counter = 0
        for elem in args.channels:
            if elem:
                channel_counter += 1
        net = models.segmentation.fcn_resnet50(pretrained=False, num_classes=2, pretrained_backbone=False)
        net.backbone.conv1 = nn.Conv2d(channel_counter, 64, kernel_size=7, stride=2, padding=3, bias=False)

        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 8]).float().cuda(), ignore_index=255)
        optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader) * 10, T_mult=2,
                                                                         eta_min=0, last_epoch=-1)

        net = convertBNtoGN(net)

        running_loss = 0
        running_iou = 0
        running_count = 0
        running_accuracy = 0

        training_losses = []
        training_accuracies = []
        training_ious = []
        valid_losses = []
        valid_accuracies = []
        valid_ious = []

        max_valid_iou = 0
        start = 0

        epochs = []

        for i in range(start, EPOCHS):
            train_validation_loop(net, optimizer, scheduler, train_loader, valid_loader, 10, i)
            epochs.append(i)
            x = epochs
            plt.plot(x, [elem.detach().cpu().numpy() for elem in training_losses], label='training losses')
            plt.plot(x, [elem.detach().cpu().numpy() for elem in valid_losses], label='valid losses')
            plt.legend(loc="upper left")

            plt.savefig(plot_path)

            print("max valid iou:", max_valid_iou)

        test_iou = test_loop(test_loader, net)
        row = [seed_id, training_ious[-1].detach().cpu().numpy(), valid_ious[-1].detach().cpu().numpy(),
               test_iou.detach().cpu().numpy()]
        writer.writerow(row)
        f.close()
