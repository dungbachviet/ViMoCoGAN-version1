#!/usr/bin/python
# -*- coding: latin-1 -*-
"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np

if torch.cuda.is_available():
    T = torch.cuda # Torch package support GPU
else:
    T = torch # Torch package support only CPU

# Create network for adding noise to input
# use_noise : boolean variable to indicate whether to add noise to input
# sigma : standard variance of noise added to input
class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * Variable(T.FloatTensor(x.size()).normal_(), requires_grad=False)
        return x


# Network of Image Discriminator ==> Conv2d (for image) ==> Output : (batchsize, channel=1, height=1, width=1)
# n_channels : number of channels (RGB) in original image (input)
# ndf : number unit used in number of channels (filters)
# use_noise : Boolean to indicate whether adds noise to input
# noise_sigma : Standard Variance of noise added to input
class ImageDiscriminator(nn.Module):
    def __init__(self, n_channels, dim_z_view, ndf=64, use_noise=False, noise_sigma=None):
        super(ImageDiscriminator, self).__init__()
        self.use_noise = use_noise
        # Sequence of networks to transform given input : batchsize of images 128x128x3 ==> (batchsize, channel=3, height=128, width=128)
        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),# Add noise to input
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),# output : (batchsize, channel=64, height=64, width=64)
            nn.LeakyReLU(0.2, inplace=True),# Activation

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # output : (batchsize, channel=64*2, height=32, width=32)
            nn.BatchNorm2d(ndf * 2),  # BatchNorm on ndf*2 channels
            nn.LeakyReLU(0.2, inplace=True),

            # Update to 128x128 image
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False),# output : (batchsize, channel=64*2, height=16, width=16)
            nn.BatchNorm2d(ndf * 2),# BatchNorm on ndf*2 channels
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),# output : (batchsize, channel=64*4, height=8, width=8)
            nn.BatchNorm2d(ndf * 4), # Batch Norm cho input
            nn.LeakyReLU(0.2, inplace=True),# activation

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),# output : (batchsize, channel=64*8, height=4, width=4)
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1 + dim_z_view, 4, 1, 0, bias=False), # output : (batchsize, channel=1, height=1, width=1)
        )

    def split(self, input):
        return input[:, : 1], input[:, 1 : input.size(1)]

    # To run this network
    def forward(self, input):
        h = self.main(input).squeeze() # tensor.squeeze() or torch.squeeze(tensor) ==> return tensor with dimension of size one eleminated
        labels, views = self.split(h)
        return labels, views, None




# Similar with ImageDiscrinator, but only have 4 conv2d to return output of size (batchsize, channel=1, height=4, width=4)
class PatchImageDiscriminator(nn.Module):
    def __init__(self, n_channels, dim_z_view, ndf=64, use_noise=False, noise_sigma=None):
        super(PatchImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, 1 + dim_z_view, 4, 2, 1, bias=False),
        )

    def split(self, input):
        return input[:, : 1], input[:, 1 : input.size(1)]

    # To run this network
    def forward(self, input):
        h = self.main(input).squeeze()  # tensor.squeeze() or torch.squeeze(tensor) ==> return tensor with dimension of size one eleminated
        labels, views = self.split(h)
        return labels, views, None



# # Model cho Discriminator cho video
# class PatchVideoDiscriminator(nn.Module):
#     def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
#         super(PatchVideoDiscriminator, self).__init__()
#
#         self.n_channels = n_channels
#         self.n_output_neurons = n_output_neurons
#         self.use_noise = use_noise
#         self.bn_use_gamma = bn_use_gamma
#
#         self.main = nn.Sequential(
#             Noise(use_noise, sigma=noise_sigma),
#             nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             Noise(use_noise, sigma=noise_sigma),
#             nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
#             nn.BatchNorm3d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             Noise(use_noise, sigma=noise_sigma),
#             nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
#             nn.BatchNorm3d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv3d(ndf * 4, 1, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
#         )
#
#     def forward(self, input):
#         h = self.main(input).squeeze()
#
#         return h, None



# Network of VideoDiscriminator : Input (batchsize, channel=3, depth=16, height=64, width=64) ==> Output (batchsize, channel=n_output_neurons=1, depth=1, height=1, width=1)
# n_channel : number of input channels (RGB)
# n_output_neurons : channels (filters) of output (after applying the network)
# use_noise : boolean whether to use noise to add to input
# bn_use_gamma : ???
# noise_sigma : standard variance of noise to add
# ndf = 64 : number unit used in formula of number of channels
class VideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(VideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        # From Input (batchsize, channel=3, depth=16, height=128, width=128)
        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),# Add noise to input
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),# Output (batchsize, channel=64, depth=13, height=64, width=64)
            nn.LeakyReLU(0.2, inplace=True), # Activation

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False), # Output (batchsize, channel=64*2, depth=10, height=32, width=32)
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Update to 128x128 image
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 2, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),# Output (batchsize, channel=64*2, depth=10, height=16, width=16)
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),


            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),# Output (batchsize, channel=64*4, depth=7, height=8, width=8)
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 4, ndf * 8, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),# Output (batchsize, channel=64*8, depth=4, height=4, width=4)
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 8, n_output_neurons, 4, 1, 0, bias=False),# Output (batchsize, channel=n_output_neurons=1, depth=1, height=1, width=1)
        )

    # To run this network
    def forward(self, input):
        h = self.main(input).squeeze()
        return h, None


# Network of CategoricalVideoDiscriminator : Input (batchsize, channel=3, depth=16, height=64, width=64) ==> Output (batchsize, channel=n_output_neurons + dim_categorical=1+6=7, depth=1, height=1, width=1)
# Install category prediction in the network
# Return vector with size = n_output_neurons + dim_categorical = 1 + 6 = 7 (for real/fake label and category label)
class CategoricalVideoDiscriminator(VideoDiscriminator):
    def __init__(self, n_channels, dim_z_view, dim_z_category, n_output_neurons=1, use_noise=False, noise_sigma=None):
        super(CategoricalVideoDiscriminator, self).__init__(n_channels=n_channels,
                                                            n_output_neurons=n_output_neurons + dim_z_view + dim_z_category,
                                                            use_noise=use_noise,
                                                            noise_sigma=noise_sigma)

        self.dim_z_category = dim_z_category
        self.dim_z_view = dim_z_view

    # input : (batchsize, channel= n_output_neurons + dim_categorical = 1 + 6 = 7)
    # Split input into : (batchsize, 1) and (batchsize, 6)
    def split(self, input):
        return input[:, : 1], input[:, 1 : 1 + self.dim_z_view], input[:, 1 + self.dim_z_view : input.size(1)]

    # Run the network ==> Ouput ==> Split output into 2 predictions: (batchsize, 1) and (batchsize, 6)
    def forward(self, input):
        h, _ = super(CategoricalVideoDiscriminator, self).forward(input)
        labels, views, categories = self.split(h)
        return labels, views, categories


# Network to generate a video (set of many frames of a video)
# Input (batchsize*video_length, channel= (dim_z_motion + dim_z_category + dim_z_content), height=1, width=1)
# Output (batchsize*video_length, channels=3, height=128, width=128)
class VideoGenerator(nn.Module):
    def __init__(self, n_channels, dim_z_content, dim_z_view, dim_z_motion, dim_z_category,
                 video_length, ngf=64):
        super(VideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_view = dim_z_view
        self.dim_z_category = dim_z_category
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length

        dim_z = dim_z_content + dim_z_view + dim_z_motion + dim_z_category

        self.recurrent = nn.GRUCell(dim_z_motion+ dim_z_category, dim_z_motion+ dim_z_category)

        # Input (batchsize*video_length, channel= (dim_z_motion + dim_z_category + dim_z_content), height=1, width=1)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False), # Output (batchsize*video_length, channels=64*8, height=4, width=4)
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),# Output (batchsize*video_length, channels=64*4, height=8, width=8)
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),# Output (batchsize*video_length, channels=64*2, height=16, width=16)
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # Update to 128x128 image
            nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False),# Output (batchsize*video_length, channels=64*2, height=32, width=32)
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),# Output (batchsize*video_length, channels=64, height=64, width=64)
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),# Output (batchsize*video_length, channels=n_channels=3, height=128, width=128)
            nn.Tanh()
        )

    # Generate batchsize of z_motion : (batchsize*video_len, dim_z_motion)
    # num_samples : batch_size
    # video_len : length of generated videos (video_len must be fixed in model)
    def sample_z_motion_category(self, num_samples, video_len=None):
        # Get fixed length of generated videos (default=video_length in configuration)
        video_len = video_len if video_len is not None else self.video_length


        # Generate a vector of size num_samples (batchsize), domain : 0-->(dim_z_category-1)
        # Each element represents a category label of an instance in batchsize
        classes_to_generate = np.random.randint(self.dim_z_category, size=num_samples)

        # Create one-hot matrix : each row is an one-hot of an instance in batchsize
        # Size : (batchsize, dim_z_category)
        one_hot_categories = np.zeros((num_samples, self.dim_z_category), dtype=np.float32)
        one_hot_categories[np.arange(num_samples), classes_to_generate] = 1.0

        # Convert numpy to tensor of torch
        one_hot_categories = torch.from_numpy(one_hot_categories)

        # Push one_hot_video to GPU
        if torch.cuda.is_available():
            one_hot_categories = one_hot_categories.cuda()

        # Wrap by Variable (batchsize, dim_z_category)
        one_hot_categories = Variable(one_hot_categories)

        # Generate batchsize of h_t at 0th time : Variable (batchsize, dim_z_motion+dim_z_category)
        h_t = [self.get_gru_initial_state(num_samples)]

        # Traverse each frames in videos (parallize on i'th frames of batchsize)
        for frame_num in range(video_len):
            e_t = self.get_iteration_noise(num_samples) # Generate random e_t : Variable(batchsize, dim_z_motion)
            et_categories = torch.cat([e_t, one_hot_categories], dim=1)
            h_t.append(self.recurrent(et_categories, h_t[-1])) # Output : z_motion : (batchsize, dim_z_motion)

        # At this time, h_t has dimension of (video_len, batch_size, dim_z_motion)

        # Design : Needed (batchsize*video_len, dim_z_motion+ self.dim_z_category)
        # Command 1 : Ouput (video_len, batchsize, 1, dim_z_motion)
        # Command 2 : torch.cat(z_m_t[1:], dim=1) ==> (batchsize, video_len, dim_z_motion+ dim_z_category)
        # Command 2, After that : .view(-1, self.dim_z_motion+dim_z_category) ==> (batchsize*video_len, dim_z_motion+dim_z_category)
        z_m_t = [h_k.view(-1, 1, self.dim_z_motion + self.dim_z_category) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion + self.dim_z_category)

        # (batchsize*video_len, dim_z_motion+dim_z_category)
        return z_m, classes_to_generate


    # Generate batchsize of categories : (batchsize, dim_z_category)
    # Category of frames in same videos is the same
    # num_samples : batchsize
    # video_len : fixed total frames of generated videos
    def sample_z_view(self, num_samples, video_len):
        video_len = video_len if video_len is not None else self.video_length

        # # In case of not using category
        # if self.dim_z_view <= 0:
        #     return None, np.zeros(num_samples)

        # Generate a vector of size num_samples (batchsize), domain : 0-->(dim_z_category-1)
        # Each element represents a category label of an instance in batchsize
        classes_to_generate = np.random.randint(self.dim_z_view, size=num_samples)

        # Create one-hot matrix : each row is an one-hot of an instance in batchsize
        # Size : (batchsize, dim_z_category)
        one_hot = np.zeros((num_samples, self.dim_z_view), dtype=np.float32)
        one_hot[np.arange(num_samples), classes_to_generate] = 1

        # Repeat each row (axis=0) more video_len times
        # (batchsize, dim_z_category) ==> (batchsize*video_len, dim_z_category)
        one_hot_video = np.repeat(one_hot, video_len, axis=0)

        # Convert numpy to tensor of torch
        one_hot_video = torch.from_numpy(one_hot_video)

        # Push one_hot_video to GPU
        if torch.cuda.is_available():
            one_hot_video = one_hot_video.cuda()

        # Variable(one_hot_video) : (batch_size*video_len, dim_z_category)
        # classes_to_generate : vector of (batch_size), each element is a scalar to indicate the category index of each instance in batchsize
        return Variable(one_hot_video), classes_to_generate





    # Generate batchsize of contents : (batchsize, dim_z_content)
    # Content of frames in same video is the same
    # num_samples : batchsize
    # video_len : fixed total frames of generated videos
    def sample_z_content(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        # Generate random uniform distribution of mean = 0, std=1
        # ==> Matrix of size (num_samples, self.dim_z_content)
        # Each row of matrix represents content of one video (in batchsize)
        content = np.random.normal(0, 1, (num_samples, self.dim_z_content)).astype(np.float32)

        # (num_samples, self.dim_z_content) ==> (num_samples*video_len, self.dim_z_content)
        # Repeat content to be same with all frames in the same video
        content = np.repeat(content, video_len, axis=0)

        # Convert from numpy to tensor of torch
        content = torch.from_numpy(content)
        if torch.cuda.is_available():
            content = content.cuda()

        # (num_samples*video_len, self.dim_z_content)
        return Variable(content)





    # Generate complete z : concatenate all of these z_content, z_category, z_motion
    def sample_z_video(self, num_samples, video_len=None):
        # Generate batchsize of z_content, z_category, z_motion_category
        # (batchsize*video_len, dim_z_xxx)
        # z_content, z_category, z_motion_category are Tensor "Variables"
        z_content = self.sample_z_content(num_samples, video_len)
        z_view, z_view_labels = self.sample_z_view(num_samples, video_len)
        z_motion_category, z_category_labels = self.sample_z_motion_category(num_samples, video_len)

        # Concatenate all of these z_content, z_category, z_motion_category
        # Ouput (batchsize*video_len, dim_z_content+dim_z_view+dim_z_motion_category)
        z = torch.cat([z_content, z_view, z_motion_category], dim=1)

        # z : (batchsize*video_len, dim_z_content+dim_z_category+dim_z_motion)
        # z_category_labels : vector of size (batch_size) (each element represents category of each instance in batchsize)
        return z, z_view_labels, z_category_labels



    # Generates batchsize of "fake" videos (each instance of batchsize will generate a video)
    # num_samples : batchsize of videos
    # video_len : fixed length of one video (total frames in one video)
    def sample_videos(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        # Generate z : (batchsize*video_len, dim_z_content+dim_z_category+dim_z_motion)
        # Generate z_category_labels : vector of size (batch_size) (each element represents category of each instance in batchsize)
        z, z_view_labels, z_category_labels = self.sample_z_video(num_samples, video_len)

        # Command : z.view(z.size(0), z.size(1), 1, 1) ==> (batchsize*video_len, dim_con+dim_cate+dim_mo, height=1, width=1)
        # Command self.main() ==> Run the network of Generator for batchsize of input
        # Ouput h of Generator : (batchsize*video_len, channels=3, height=64, width=64)
        h = self.main(z.view(z.size(0), z.size(1), 1, 1))

        # Transpose h to dimension : (batchsize, video_len, channel=3, height=64, width=64)
        h = h.view(h.size(0) / video_len, video_len, self.n_channels, h.size(3), h.size(3))

        # Convert numpy to tensor of torch
        z_view_labels = torch.from_numpy(z_view_labels)
        z_category_labels = torch.from_numpy(z_category_labels)

        # Push z_category_labels to GPU (all of z_cate, z_con, z_motion have been pushed before)
        if torch.cuda.is_available():
            z_view_labels = z_view_labels.cuda()
            z_category_labels = z_category_labels.cuda()

        # Need to transpose for suiting input to C3D network of video_discriminator
        # Transpose to dimension : (batchsize, channel=3, depth=video_len, height=64, width=64)
        h = h.permute(0, 2, 1, 3, 4)

        # Wrap z_category_labels (tensor) into Variable (h has been wrapped in Variable before...)
        return h, Variable(z_view_labels, requires_grad=False), Variable(z_category_labels, requires_grad=False)


    # Generate batchsize of "fake" images
    # num_samples : batchsize of images
    def sample_images(self, num_samples):
        # Generate batchsize of (num_samples * self.video_length * 2) videos
        # Each videos has video_length image frames
        # Dimension z : (num_samples*video_length*2*video_length, dim_z)
        # z_category_labels : vector (num_samples*video_length*2*video_length)
        # ??? Is this wrong? ==> choose num_samples frames from num_samples*video_length*2*video_length frames
        # z, z_view_labels, z_category_labels = self.sample_z_video(num_samples * self.video_length * 2)
	z, z_view_labels, z_category_labels = self.sample_z_video(num_samples)

        # Generate vetor of size num_samples,
        # containing values from 0-->z.size(0)=num_samples*video_length*2*video_length-1 (uniquely)
        # Sort vector to increasingly order (cast to type of np.int64)
        j = np.sort(np.random.choice(z.size(0), num_samples, replace=False)).astype(np.int64)

        # Get out num_samples rows have indices matching with values in vector j
        z = z[j, ::]

        # Get out corresponding views labels ==> Not care about the category because these is images
        indices = (j/self.video_length).astype(int)
        z_view_labels = z_view_labels[indices]

        # Transpose z to (num_samples, channels=dim_z, height=1, width=1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        # Take in z as a input of network Generator
        # Output : (num_samples, channels=3, height=64, width=64)
        h = self.main(z)

        # Convert numpy to tensor of torch
        z_view_labels = torch.from_numpy(z_view_labels)

        # Push z_category_labels to GPU (all of z_cate, z_con, z_motion have been pushed before)
        if torch.cuda.is_available():
            z_view_labels = z_view_labels.cuda()

        # Return h : (num_samples, channels=3, height=64, width=64)
        # None : means images don't aware about action category of image (only videos can do this)
        return h, Variable(z_view_labels, requires_grad=False), None

    # Initialize batchsize of hidden states h[0]s : (batchsize, dim_z_motion)
    # with values drawn from normal distribution of mean=0, std=1
    def get_gru_initial_state(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion + self.dim_z_category).normal_())

    # Initialize batchize of noise : (batchsize, dim_z_motion)
    # with values drawn from normal distribution of mean=0, std=1
    def get_iteration_noise(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())









    #######################################
    ########################################
    ##################################
    # For test ==> Save generated Videos
    def test_sample_z_motion_category(self, factor, video_len=None):
        # Get fixed length of generated videos (default=video_length in configuration)
        video_len = video_len if video_len is not None else self.video_length

        # Generate a vector of size num_samples (batchsize), domain : 0-->(dim_z_category-1)
        # Each element represents a category label of an instance in batchsize

        # Specify views to generate
        view_list = []
        for i in range(self.dim_z_view):
            view_list.extend(range(self.dim_z_category))
        print(view_list)

        view_list_factor = []
        for i in range(factor):
            view_list_factor.extend(view_list)
        print(view_list_factor)

        classes_to_generate = np.array(view_list_factor, dtype=int)



        # Create one-hot matrix : each row is an one-hot of an instance in batchsize
        # Size : (batchsize, dim_z_category)
        one_hot_categories = np.zeros((classes_to_generate.shape[0], self.dim_z_category), dtype=np.float32)
        one_hot_categories[np.arange(classes_to_generate.shape[0]), classes_to_generate] = 1.0

        # Convert numpy to tensor of torch
        one_hot_categories = torch.from_numpy(one_hot_categories)

        # Push one_hot_video to GPU
        if torch.cuda.is_available():
            one_hot_categories = one_hot_categories.cuda()

        # Wrap by Variable (batchsize, dim_z_category)
        one_hot_categories = Variable(one_hot_categories)

        # Generate batchsize of h_t at 0th time : Variable (batchsize, dim_z_motion+dim_z_category)
        h_t = [self.get_gru_initial_state(classes_to_generate.shape[0])]

        # Traverse each frames in videos (parallize on i'th frames of batchsize)
        for frame_num in range(video_len):
            e_t = self.get_iteration_noise(classes_to_generate.shape[0])  # Generate random e_t : Variable(batchsize, dim_z_motion)
            et_categories = torch.cat([e_t, one_hot_categories], dim=1)
            h_t.append(self.recurrent(et_categories, h_t[-1]))  # Output : z_motion : (batchsize, dim_z_motion)

        # At this time, h_t has dimension of (video_len, batch_size, dim_z_motion)

        # Design : Needed (batchsize*video_len, dim_z_motion+ self.dim_z_category)
        # Command 1 : Ouput (video_len, batchsize, 1, dim_z_motion)
        # Command 2 : torch.cat(z_m_t[1:], dim=1) ==> (batchsize, video_len, dim_z_motion+ dim_z_category)
        # Command 2, After that : .view(-1, self.dim_z_motion+dim_z_category) ==> (batchsize*video_len, dim_z_motion+dim_z_category)
        z_m_t = [h_k.view(-1, 1, self.dim_z_motion + self.dim_z_category) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion + self.dim_z_category)

        # (batchsize*video_len, dim_z_motion+dim_z_category)
        return z_m, classes_to_generate




    # For test ==> Save generated videos
    def test_sample_z_view(self, factor, video_len):
        video_len = video_len if video_len is not None else self.video_length

        # # In case of not using category
        # if self.dim_z_view <= 0:
        #     return None, np.zeros(num_samples)

        # Specify views to generate
        view_list = []
        for i in range(self.dim_z_view):
            view_list.extend([i] * self.dim_z_category)
        print(view_list)

        view_list_factor = []
        for i in range(factor):
            view_list_factor.extend(view_list)
        print(view_list_factor)

        classes_to_generate = np.array(view_list_factor, dtype=int)


        # Create one-hot matrix : each row is an one-hot of an instance in batchsize
        # Size : (batchsize, dim_z_category)
        one_hot = np.zeros((classes_to_generate.shape[0], self.dim_z_view), dtype=np.float32)
        one_hot[np.arange(classes_to_generate.shape[0]), classes_to_generate] = 1
        # one_hot[np.arange(classes_to_generate.shape[0]), (classes_to_generate+1)%self.dim_z_view] = 0.2

        print("*** one_hot views = : ", one_hot)

        # Repeat each row (axis=0) more video_len times
        # (batchsize, dim_z_category) ==> (batchsize*video_len, dim_z_category)
        one_hot_video = np.repeat(one_hot, video_len, axis=0)

        # Convert numpy to tensor of torch
        one_hot_video = torch.from_numpy(one_hot_video)

        # Push one_hot_video to GPU
        if torch.cuda.is_available():
            one_hot_video = one_hot_video.cuda()

        # Variable(one_hot_video) : (batch_size*video_len, dim_z_category)
        # classes_to_generate : vector of (batch_size), each element is a scalar to indicate the category index of each instance in batchsize
        return Variable(one_hot_video), classes_to_generate


    def test_sample_z_content(self, factor, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        # Generate random uniform distribution of mean = 0, std=1
        # ==> Matrix of size (num_samples, self.dim_z_content)
        # Each row of matrix represents content of one video (in batchsize)
        content = np.random.normal(0, 1, (self.dim_z_view*self.dim_z_category*factor, self.dim_z_content)).astype(np.float32)

        # (num_samples, self.dim_z_content) ==> (num_samples*video_len, self.dim_z_content)
        # Repeat content to be same with all frames in the same video
        content = np.repeat(content, video_len, axis=0)

        # Convert from numpy to tensor of torch
        content = torch.from_numpy(content)
        if torch.cuda.is_available():
            content = content.cuda()

        # (num_samples*video_len, self.dim_z_content)
        return Variable(content)



    # Generate complete z : concatenate all of these z_content, z_category, z_motion
    def test_sample_z_video(self, factor, video_len=None):
        # Generate batchsize of z_content, z_category, z_motion_category
        # (batchsize*video_len, dim_z_xxx)
        # z_content, z_category, z_motion_category are Tensor "Variables"
        z_content = self.test_sample_z_content(factor, video_len)
        z_view, z_view_labels = self.test_sample_z_view(factor, video_len)
        z_motion_category, z_category_labels = self.test_sample_z_motion_category(factor, video_len)

        # Concatenate all of these z_content, z_category, z_motion_category
        # Ouput (batchsize*video_len, dim_z_content+dim_z_view+dim_z_motion_category)

        print("z_content.size() = ", z_content.size())
        print("z_view.size() = ", z_view.size())
        print("z_motion_category.size() = ", z_motion_category.size())
        z = torch.cat([z_content, z_view, z_motion_category], dim=1)

        # z : (batchsize*video_len, dim_z_content+dim_z_category+dim_z_motion)
        # z_category_labels : vector of size (batch_size) (each element represents category of each instance in batchsize)
        return z, z_view_labels, z_category_labels



    def test_sample_videos(self, factor, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        # Generate z : (batchsize*video_len, dim_z_content+dim_z_category+dim_z_motion)
        # Generate z_category_labels : vector of size (batch_size) (each element represents category of each instance in batchsize)
        z, z_view_labels, z_category_labels = self.test_sample_z_video(factor, video_len)

        # Command : z.view(z.size(0), z.size(1), 1, 1) ==> (batchsize*video_len, dim_con+dim_cate+dim_mo, height=1, width=1)
        # Command self.main() ==> Run the network of Generator for batchsize of input
        # Ouput h of Generator : (batchsize*video_len, channels=3, height=64, width=64)
        h = self.main(z.view(z.size(0), z.size(1), 1, 1))

        # Transpose h to dimension : (batchsize, video_len, channel=3, height=64, width=64)
        h = h.view(h.size(0) / video_len, video_len, self.n_channels, h.size(3), h.size(3))

        # Convert numpy to tensor of torch
        z_view_labels = torch.from_numpy(z_view_labels)
        z_category_labels = torch.from_numpy(z_category_labels)

        # Push z_category_labels to GPU (all of z_cate, z_con, z_motion have been pushed before)
        if torch.cuda.is_available():
            z_view_labels = z_view_labels.cuda()
            z_category_labels = z_category_labels.cuda()

        # Need to transpose for suiting input to C3D network of video_discriminator
        # Transpose to dimension : (batchsize, channel=3, depth=video_len, height=64, width=64)
        h = h.permute(0, 2, 1, 3, 4)

        # Wrap z_category_labels (tensor) into Variable (h has been wrapped in Variable before...)
        return h, Variable(z_view_labels, requires_grad=False), Variable(z_category_labels, requires_grad=False)

