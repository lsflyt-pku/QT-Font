# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from ocnn_2d.modules import InputFeature
 
def axis_loss(batch, model_out):
  max_depth = batch['octree_gt'].depth
  octree_out = batch['octree_gt']
  logits = model_out['logits']
  output = dict()
  octree_feature = InputFeature(feature = 'F', nempty=False)
  for d in logits.keys():
    logitd = logits[d]
    if d == max_depth:
      label_gt = octree_feature(octree_out)
      # pdb.set_trace()
      label_gt = label_gt[:, 2] * 1 + label_gt[:, 3] * 2
      label_gt = label_gt.long()
    else:
      label_gt = octree_out.nempty_mask(d).long()
    output['loss_%d' % d] = F.cross_entropy(logitd, label_gt)
    output['accu_%d' % d] = logitd.argmax(1).eq(label_gt).float().mean()
  return output

def leaf_pixel_loss(batch, model_out):
  # octree loss
  output = compute_octree_loss(model_out['logits'], model_out['octree_out'])
  return output

def compute_octree_loss(logits, octree_out):
  weights = [1.0] * 16
  # weights = [1.0] * 4 + [0.8, 0.6, 0.4] + [0.2] * 16

  output = dict()
  for d in logits.keys():
    logitd = logits[d]
    label_gt = octree_out.nempty_mask(d).long()
    output['loss_%d' % d] = F.cross_entropy(logitd, label_gt) * weights[d]
    output['accu_%d' % d] = logitd.argmax(1).eq(label_gt).float().mean()
  return output

