# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from typing import Optional, Union


class KeyLUT:

  def __init__(self):
    r256 = torch.arange(256, dtype=torch.int64)
    r512 = torch.arange(512, dtype=torch.int64)
    zero = torch.zeros(256, dtype=torch.int64)
    device = torch.device('cpu')

    self._encode = {device: (self.xyz2key(r256, zero, 8),
                             self.xyz2key(zero, r256, 8))}
                            #  self.xyz2key(zero, zero, 8))}
    # check
    # for i in range(4):
    #   for j in range(4):
    #     print(i, j, self.xyz2key(torch.zeros(1).long()+i, torch.zeros(1).long()+j, 8))
    
    # for i in range(128*128):
      # print(i, self.key2xyz(torch.zeros(1).long()+i, 8))
    # input()
    
    self._decode = {device: self.key2xyz(r512, 9)}

  def encode_lut(self, device=torch.device('cpu')):
    if device not in self._encode:
      cpu = torch.device('cpu')
      self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
    return self._encode[device]

  def decode_lut(self, device=torch.device('cpu')):
    if device not in self._decode:
      cpu = torch.device('cpu')
      self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
    return self._decode[device]

  def xyz2key(self, x, y, depth):
    key = torch.zeros_like(x)
    for i in range(depth):
      mask = 1 << i
      key = (key | ((x & mask) << (1 * i + 1)) |
                   ((y & mask) << (1 * i + 0)))
                  #  ((z & mask) << (2 * i + 0)))
    return key

  def key2xyz(self, key, depth):
    x = torch.zeros_like(key)
    y = torch.zeros_like(key)
    # z = torch.zeros_like(key)
    for i in range(depth):
      x = x | ((key & (1 << (2 * i + 1))) >> (1 * i + 1))
      y = y | ((key & (1 << (2 * i + 0))) >> (1 * i + 0))
      # z = z | ((key & (1 << (3 * i + 0))) >> 2( * i + 0))
    return x, y


_key_lut = KeyLUT()


def xyz2key(x: torch.Tensor, y: torch.Tensor,
            b: Optional[Union[torch.Tensor, int]] = None, depth: int = 16):
  # 2d
  r'''Encodes :attr:`x`, :attr:`y` coordinates to the shuffled keys
  based on pre-computed look up tables. The speed of this function is much
  faster than the method based on for-loop.

  Args:
    x (torch.Tensor): The x coordinate.
    y (torch.Tensor): The y coordinate.
    b (torch.Tensor or int): The batch index of the coordinates, and should be 
        smaller than 32768. If :attr:`b` is :obj:`torch.Tensor`, the size of
        :attr:`b` must be the same as :attr:`x` and :attr:`y`.
    depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
  '''

  EX, EY = _key_lut.encode_lut(x.device)
  x, y = x.long(), y.long()
  # 255=11111111(2进制)
  mask = 255 if depth > 8 else (1 << depth) - 1
  key = EX[x & mask] | EY[y & mask]
  if depth > 8:
    mask = (1 << (depth-8)) - 1
    key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask]
    key = key16 << 16 | key

  # 48保留
  if b is not None:
    b = b.long()
    key = b << 48 | key

  return key


def key2xyz(key: torch.Tensor, depth: int = 16):
  # 2d
  r'''Decodes the shuffled key to :attr:`x`, :attr:`y`, :attr:`z` coordinates
  and the batch index based on pre-computed look up tables.

  Args:
    key (torch.Tensor): The shuffled key.
    depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
  '''
  
  DX, DY = _key_lut.decode_lut(key.device)
  x, y = torch.zeros_like(key), torch.zeros_like(key)

  # 48保留
  b = key >> 48
  key = key & ((1 << 48) - 1)

  n = (depth + 1) // 2 # 相当于上取整

  for i in range(n):
    k = key >> (i * 4) & 511 # 这个地方不清楚为什么原来是9，改成4, 猜测是xyz每个3bit，改成xy每个2bit
    x = x | (DX[k] << (i * 2))
    y = y | (DY[k] << (i * 2))
 
  return x, y, b

# check
# for i in range(4):
#   for j in range(4):
#     print(i, j, xyz2key(torch.zeros(1).long()+i, torch.zeros(1).long()+j, None, 8))
 
# for i in range(128*128):
  # print(i, key2xyz(torch.zeros(1).long()+i, 8))
# print(xyz2key(torch.zeros(1).long()+127, torch.zeros(1).long()+127, None, 16)) 
# print(key2xyz(torch.zeros(1).long()+128*128-1, 16))
