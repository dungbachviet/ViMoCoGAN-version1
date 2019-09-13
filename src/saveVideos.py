"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz, MoCoGAN: Decomposing Motion and Content for Video Generation
https://arxiv.org/abs/1707.04993

Usage:
    saveVideos.py [options] <checkpoint_path>

Options:
    --resume                        when specified instance noise is used [default: False]
    --image_batch=<count>           number of images in image batch [default: 40]
    --video_batch=<count>           number of videos in video batch [default: 3]
    --factor=<count>                factor of total videos
    --dim_z_view=<count>            number of views
    --dim_z_category=<count>        number of categories
    --video_length=<count>          frames per video
    --performer_name=<path>         name of the performer [default: ]
    --output_folder=<path>          name of output folder [default: ]
    --is_6_dau                      when specified instance noise is used [default: False]

"""

import os
import time
import sys
import shutil

import numpy as np
import cv2
import torch

torch.backends.cudnn.enabled = False
from torch import nn

from torch.autograd import Variable
import torch.optim as optim

import os
from trainers import videos_to_numpy
import subprocess as sp
import models
import matplotlib.pyplot as plt
import PIL
import docopt


args = docopt.docopt(__doc__) # __doc__ is the first above comment of this file
print(args) # Display the parameters configuration

# checkpoint = torch.load("./log_folder/checkpoint_002023_000045.pth.tar")
checkpoint = torch.load(str(args['<checkpoint_path>']))

print("checkpoint['batch_num'] = ", checkpoint['batch_num'])
print("epoch_num = ", checkpoint["epoch_num"])

print("checkpoint['generator_loss'] = ", checkpoint['generator_loss'])

print("checkpoint['image_discriminator_loss'] = ", checkpoint['image_discriminator_loss'])

print("checkpoint['video_discriminator_loss'] = ", checkpoint['video_discriminator_loss'])


factor = int(args["--factor"])
dim_z_view = int(args["--dim_z_view"])
dim_z_category = int(args["--dim_z_category"])
num_class = dim_z_view * dim_z_category
video_length = int(args["--video_length"])
output_folder = str(args["--output_folder"])
performer_name = str(args["--performer_name"])
is_6_dau = args["--is_6_dau"]

offset = 0
if (is_6_dau == False) :
    offset = 6

# Create object of VideoGenerator
generator = models.VideoGenerator(n_channels=3, dim_z_content=50, dim_z_view=dim_z_view, dim_z_motion=10, dim_z_category=dim_z_category,
                 video_length=video_length, ngf=64)
print(generator)
print("torch.cuda.is_available() = ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch.cuda.is_available() = ", torch.cuda.is_available())
    generator.cuda()

generator.load_state_dict(checkpoint['generator'])
generator.eval()

# Create output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

videos, z_view_labels, z_category_labels = generator.test_sample_videos(factor, video_length)  # number of video, number of frames
videos = videos_to_numpy(videos)

integerToKinect = {1:"K1_original", 2:"K4_original", 3:"K5_original", 4:"K3_original", 5:"K2_original"}

print("range(videos.size(0)) = ", range(videos.shape[0]))
for video_id in range(videos.shape[0]):
    # ==> (depth, height, width, channel) ==> Split into each image to display
    video = videos[video_id, ::].transpose((1, 2, 3, 0))
    turn = video_id//num_class

    print("turn = ", turn)
    for image_index in range(video.shape[0]):
        # Resize image array to size of 171x128 (suit with C3D)
        resized = cv2.resize(video[image_index, ::], (128, 128), interpolation=cv2.INTER_AREA)
        image = PIL.Image.fromarray(resized)

        save_directory = os.path.join(output_folder,
                                      "%s/%s/%d/%d" %
                                      (integerToKinect[z_view_labels.data[video_id] + 1],
                                       performer_name,
                                       z_category_labels.data[video_id] + 1 + offset,
                                       turn+1))

        print("save_directory = ", save_directory)

        if (not os.path.exists(save_directory)):
            os.makedirs(save_directory)
        image.save(os.path.join(save_directory, '%06d.jpg' % (image_index+1)))






#
# # Create object of VideoGenerator
# generator = models.VideoGenerator(n_channels=3, dim_z_content=50, dim_z_view=2, dim_z_motion=10, dim_z_category=12,
#                  video_length=16, ngf=64)
# print(generator)
# print("torch.cuda.is_available() = ", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("torch.cuda.is_available() = ", torch.cuda.is_available())
#     generator.cuda()
#
# generator.load_state_dict(checkpoint['generator'])
# generator.eval()
#
# output_folder = "./test_generate_video"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
# num_videos = 1
# for i in range(num_videos):
#     v, _, _ = generator.sample_videos(1, 16)  # number of video, number of frames
#     video = videos_to_numpy(v).squeeze().transpose((1, 2, 3, 0))
#     for image_index in range(video.shape[0]):
#         image = PIL.Image.fromarray(video[image_index, ::])
#         image.save('./test_generate_video/out%002d_%002d.jpg' % (i, image_index))



