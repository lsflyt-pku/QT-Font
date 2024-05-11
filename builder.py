# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn_2d

import datasets
import models
import losses

from torch.nn import init


def get_model(flags):
  params = [flags.depth, flags.channel, flags.nout,
            flags.full_depth, flags.depth_stop, flags.depth_out]

  params.append(flags.resblock_type)
  params.append(flags.bottleneck)
  params.append(flags.resblk_num)
  params.append(flags.cond)

  if flags.name == 'graph_diffusion':
    model = models.graph_diffusion.Graph_diffusion(*params)
  else:
    raise ValueError
  return model


def get_dataset(flags):

  if flags.name.lower() == 'chinesefont_asymmetric':
    return datasets.get_chinesefont_asymmetric_dataset(flags)
  else:
    raise ValueError

def get_loss_function(flags):
  if flags.name.lower() == 'chinesefont':
    return losses.leaf_pixel_loss
  if flags.name.lower() == 'chinesefont_axis':
    return losses.axis_loss
  else:
    raise ValueError
