# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn_2d
import torch
from ocnn_2d.dataset import CollateBatch


def collate_func(batch):
  
  collate_batch = CollateBatch(merge_points=False)
  output = collate_batch(batch)
  # output = ocnn.collate_octrees(batch)

  if 'pos' in output:
    batch_idx = torch.cat([torch.ones(pos.size(0), 1) * i
                           for i, pos in enumerate(output['pos'])], dim=0)
    pos = torch.cat(output['pos'], dim=0)
    output['pos'] = torch.cat([pos, batch_idx], dim=1)

  for key in ['grad', 'sdf', 'occu', 'weight']:
    if key in output:
      output[key] = torch.cat(output[key], dim=0)

  return output
