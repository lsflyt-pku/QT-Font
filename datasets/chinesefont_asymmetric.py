# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import ocnn_2d
import torch
import torch.nn.functional as F
import numpy as np
import copy
import cv2 as cv
from PIL import Image
import PIL
import json
from skimage import morphology

from ocnn_2d.octree import Octree, Points
from solver import Dataset
from .utils import collate_func
from diffusers import DDPMScheduler, DDIMScheduler
import random


class TransformShape:
  def __init__(self, flags):
    self.flags = flags

    self.depth = flags.depth
    self.image_size = 2 ** flags.depth
    self.mid = (self.image_size - 1.) / 2.
    self.full_depth = flags.full_depth
    self.num_classes = 3

    self.ref_num = flags.ref_num
    self.c_ref_num = flags.c_ref_num

    self.thres = 0.97

    num_train_timesteps = flags.num_train_timesteps
    self.num_train_timesteps = num_train_timesteps
    beta_start = flags.beta_start 
    beta_end = flags.beta_end

    self.points_scale = 1  # the points are in [-1, 1]
    if flags.beta_schedule == "linear":
      self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    elif flags.beta_schedule == "scaled_linear":
      # this schedule is very specific to the latent diffusion model.
      self.betas = (
          torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
      )
    elif flags.beta_schedule == "cos":
      # Glide cosine schedule
      steps = (np.arange(num_train_timesteps + 1, dtype=np.float64) / num_train_timesteps)
      alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
      betas = np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
      self.betas = torch.tensor(betas)
    elif flags.beta_schedule == "sigmoid":
      # GeoDiff sigmoid schedule
      betas = torch.linspace(-6, 6, num_train_timesteps)
      self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
      raise NotImplementedError
      
    self.Q = torch.ones([num_train_timesteps, self.num_classes, self.num_classes], dtype=torch.float32)
    for i in range(self.num_classes):
      for j in range(self.num_classes):
        if i == j:
          self.Q[:, i, j] = 1 - self.betas * (self.num_classes-1) / self.num_classes
        else:
          self.Q[:, i, j] = self.betas / self.num_classes

    self.Q_T = self.Q.permute(0, 2, 1)

    self.Q_ = torch.ones([num_train_timesteps, self.num_classes, self.num_classes], dtype=torch.float32)
    self.Q_[0] = self.Q[0]
    for i in range(1, num_train_timesteps):
      self.Q_[i] = torch.matmul(self.Q_[i-1], self.Q[i])

  def points2octree(self, points: Points, small=False):
    if small:
      octree = Octree(7, self.full_depth)
    else:
      octree = Octree(self.depth, self.full_depth)
    octree.build_octree(points)
    return octree

  def _at(self, a, t, x):
    t_broadcast = np.expand_dims(t, tuple(range(1, x.ndim)))
    return a[t_broadcast, x]

  def q_probs(self, x_0, t):
    '''
    x_0: L {0, 1}
    t: L numpy
    '''
    return self._at(self.Q_, t, x_0)

  def q_sample(self, x_0, t, noise):
    '''
    Compute logits of q(x_t | x_start).
    x_0: L {0, 1}
    noise: Lx2
    t: numpy
    '''
    logits = torch.log(self.q_probs(x_0, t) + 1e-8)
    noise = torch.clip(noise, min=1e-8, max=1.0)
    gumbel_noise = -torch.log(-torch.log(noise))
    return torch.argmax(logits+gumbel_noise, -1)

  def process_points_cloud(self, sample):
    # get the input
    image = sample['image']
    image = torch.from_numpy(image).long()# [0, 1, 2]
    image[image==3] = 1

    # x_0
    points = []
    feature_gt = []

    for label in range(1, self.num_classes):
      points.append(torch.stack(torch.where(image==label), -1))
      feature_gt.append(torch.zeros((points[-1].shape[0], self.num_classes-1)))
      feature_gt[-1][:, label-1] = 1
    points = torch.concat(points, 0)
    feature_gt = torch.concat(feature_gt, 0)
    points = (points - self.mid) / self.mid

    points_gt = Points(points=points, features=torch.cat((points, feature_gt), -1))
    octree_gt = self.points2octree(points_gt)

    # x_t
    t = np.random.randint(0, self.num_train_timesteps)
    timesteps = torch.tensor(t).long()

    L = self.image_size ** 2
    noise = torch.rand((L, self.num_classes))

    noise_image = self.q_sample(image.view(-1), np.array([t]*L), noise)
    noise_image = noise_image.view(self.image_size, self.image_size)

    # beta_t = self.Q_[t, 0, 1]
    # alpha_t = self.Q_[t, 0, 0]
    # noise_image = noise_image * (torch.rand(noise_image.shape) < (1-self.thres)/(beta_t*self.thres+alpha_t-alpha_t*self.thres))

    weight = torch.zeros((128, 128, 3))
    noise_image = noise_image.view(128, self.image_size // 128, 128, self.image_size // 128).permute(0, 2, 1, 3).reshape(128, 128, -1)
    zero_point = torch.stack(torch.where(noise_image.sum(-1)==0), -1)
    one_image = (noise_image == 1).sum(-1)
    two_image = (noise_image == 2).sum(-1)
    x = torch.linspace(0, 127, steps=128).long()
    y = torch.linspace(0, 127, steps=128).long()
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    weight[:, :, 1] = one_image / (one_image + two_image + 1e-8)
    weight[:, :, 2] = two_image / (one_image + two_image + 1e-8)
    weight[zero_point[:, 0], zero_point[:, 1], 0] = 1.0
    weight[zero_point[:, 0], zero_point[:, 1], 1] = 0.0
    weight[zero_point[:, 0], zero_point[:, 1], 2] = 0.0

    image_rand = torch.rand((128, 128))
    weight[:, :, 1] += weight[:, :, 0]
    weight[:, :, 2] += weight[:, :, 1]
    noise_image = (torch.gt(image_rand, weight[:, :, 0]) * torch.le(image_rand, weight[:, :, 1])) + torch.gt(image_rand, weight[:, :, 1]) * 2

    noise_points = []
    noise_feature = []
    for label in range(1, self.num_classes):
      noise_points.append(torch.stack(torch.where(noise_image==label), -1))
      noise_feature.append(torch.zeros((noise_points[-1].shape[0], self.num_classes-1)))
      noise_feature[-1][:, label-1] = 1
    noise_points = torch.concat(noise_points, 0)
    noise_feature = torch.concat(noise_feature, 0)
    noise_points = (noise_points - self.mid) / self.mid

    points_in = Points(points=noise_points, features=torch.cat((noise_points, noise_feature), -1))
    octree_in = self.points2octree(points_in, True)

    # style
    style_list = sample['style']
    octree_style_list = []
    for i in range(self.ref_num):
      style = style_list[i]
      style = torch.from_numpy(style).long()

      style_points = torch.stack(torch.where(style==1), -1)
      style_points = (style_points - self.mid) / self.mid

      style_points = style_points / self.points_scale  # scale to [-1.0, 1.0]
      # feature_style = torch.ones((style_points.shape[0], 1)).float()

      # points_style = Points(points=style_points, features=feature_style)
      points_style = Points(points=style_points, features=style_points)
      octree_style = self.points2octree(points_style)
      octree_style_list.append(octree_style)
      
    # content
    content_list = sample['content']
    octree_content_list = []

    for i in range(self.c_ref_num):
      content = content_list[i]
      content = torch.from_numpy(content).long()

      content_points = torch.stack(torch.where(content==1), -1)
      content_points = (content_points - self.mid) / self.mid

      content_points = content_points / self.points_scale  # scale to [-1.0, 1.0]
      points_content = Points(points=content_points, features=content_points)
      octree_content = self.points2octree(points_content)
      octree_content_list.append(octree_content)

    if len(octree_content_list) == 1:
      octree_content_list = octree_content_list[0]
    
    return {'octree_in': octree_in, 'octree_gt': octree_gt, 'pos': points, 'timesteps': timesteps, 'octree_style':octree_style_list, 'octree_content':octree_content_list} 

  def __call__(self, sample, idx):
    output = self.process_points_cloud(sample['point_cloud'])

    return output


class ReadFile:
  def __init__(self, flags):
    self.image_size = 2 ** flags.depth
    self.canny = flags.canny

    with open('data/VQ-Font128/train_unis.json', 'r') as f:
      ref_list = f.readlines()[0]
      ref_list = ref_list.split(',')
      ref_list = [x.strip()[1:5] for x in ref_list]
      ref_list = [r'\u'+x for x in ref_list]
      ref_list = [json.loads('"%s"' % x) for x in ref_list]

    with open('train_font.txt', 'r') as f:
      train_font_list = f.readlines()[0]
      train_font_list = train_font_list.split(',')

    self.ref_list = ref_list
    self.train_font_list = train_font_list
    self.ref_num = flags.ref_num
    self.c_ref_num = flags.c_ref_num


  def __call__(self, filename):
    filename = filename.replace('VQ-Font128', 'VQ-Font'+str(self.image_size))
    root, basename = os.path.split(filename)
    font, char = os.path.basename(filename)[:-4].split('_')
    
    filename = filename.replace('_', '/')
    image = Image.open(filename)
    
    image = image.resize((self.image_size, self.image_size))
    image = np.array(image)
    image = image[:, :, 0]
    image = (image > 127.5).astype(np.uint8)

    style_path = []
    for i in range(self.ref_num):
      ref_char = random.choice(self.ref_list)
      while ref_char == char:
        ref_char = random.choice(self.ref_list)
      style_path.append(os.path.join(root, '{}/{}.png'.format(font, ref_char)))        

    if 'val' in root:
      content_root = root.replace('val', 'content')
    elif 'train' in root:
      content_root = root.replace('train', 'content')

    train_root = content_root.replace('content', 'train')

    content_path = []
    if self.c_ref_num > 1:
      for i in range(self.c_ref_num):
        ref_font = random.choice(self.train_font_list)
        while ref_font == font:
          ref_font = random.choice(self.train_font_list)
        content_path.append(os.path.join(train_root, '{}/{}.png'.format(ref_font, char)))
    else:
        content_path.append(os.path.join(content_root, '{}/{}.png'.format(0, char)))

    style_image_list = []
    for i in range(self.ref_num):
      style_image = Image.open(style_path[i])
      style_image = style_image.resize((self.image_size, self.image_size))
      style_image = np.array(style_image)
      style_image = style_image[:, :, 0]
      style_image = (style_image > 127.5).astype(np.uint8)
      style_image_list.append(style_image)

    content_image_list = []
    for i in range(self.c_ref_num):
      content_image = Image.open(content_path[i])
      content_image = content_image.resize((self.image_size, self.image_size))
      content_image = np.array(content_image)
      content_image = content_image[:, :, 0]
      content_image = (content_image > 127.5).astype(np.uint8)
      content_image_list.append(content_image)

    if self.canny:
      image = (1 - image) * 255
      skeleton = morphology.skeletonize(image / 255)
      contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
      canvas = np.zeros((self.image_size, self.image_size))
      canvas = cv.drawContours(canvas, contours, -1, (1), 1) + skeleton * 2
      image = canvas


      for i in range(self.ref_num):
        style_image = style_image_list[i]
        style_image = (1 - style_image) * 255
        contours, hierarchy = cv.findContours(style_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        canvas = np.zeros((self.image_size, self.image_size))
        style_image = cv.drawContours(canvas, contours, -1, (1), 1)

        style_image_list[i] = style_image

      for i in range(self.c_ref_num):
        content_image = content_image_list[i]
        content_image = (1 - content_image) * 255
        contours, hierarchy = cv.findContours(content_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        canvas = np.zeros((self.image_size, self.image_size))
        content_image = cv.drawContours(canvas, contours, -1, (1), 1)

        content_image_list[i] = content_image

    point_cloud = {'image':image, 'style':style_image_list, 'content':content_image_list}
    
    output = {'point_cloud': point_cloud}
    return output


def get_chinesefont_asymmetric_dataset(flags):
  transform = TransformShape(flags)
  read_file = ReadFile(flags)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, collate_func
