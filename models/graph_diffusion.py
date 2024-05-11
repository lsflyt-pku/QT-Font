# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn
from torch.nn import init
import torch.nn.functional as F

from . import modules_bn
from . import dual_octree
from ocnn_2d.octree import Octree, merge_octrees
import math
import pdb

class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(torch.nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = torch.nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = torch.nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb


class Graph_diffusion(torch.nn.Module):
    def __init__(self, depth, channel_in, nout, full_depth=2, depth_stop = 6, depth_out=8,
                resblk_type='bottleneck', bottleneck=4, resblk_num=3, cond='uncond'):
        super().__init__()
        self.depth = depth
        self.channel_in = channel_in
        self.nout = nout
        self.full_depth = full_depth
        self.depth_stop = depth_stop
        self.depth_out = depth_out
        self.resblk_type = resblk_type
        self.bottleneck = bottleneck
        self.resblk_num = resblk_num
        self._setup_channels_and_resblks()
        n_edge_type, avg_degree = 5, 5
        self.dropout = 0.0
        
        self.cond = cond
        print('cond', self.cond)

        self.inputdata = 'LF'
        
        # style & content encoder
        self.style_conv = modules_bn.GraphConv(4, self.channels[depth_out], n_edge_type, avg_degree, depth_out-1)
        self.style_encoder = torch.nn.ModuleList(
            [modules_bn.GraphResBlocks(self.channels[d], self.channels[d], self.dropout, 1, n_edge_type, avg_degree, d-1, self.channels[depth_stop]) 
            for d in range(depth_out, depth_stop-1, -1)])
        self.style_downsample = torch.nn.ModuleList([modules_bn.GraphDownsample(self.channels[d], self.channels[d-1]) for d in range(depth_out, depth_stop, -1)])

        self.content_conv = modules_bn.GraphConv(4, self.channels[depth_out], n_edge_type, avg_degree, depth_out-1)
        self.content_encoder = torch.nn.ModuleList(
            [modules_bn.GraphResBlocks(self.channels[d], self.channels[d], self.dropout, 1, n_edge_type, avg_degree, d-1, self.channels[depth_stop]) 
            for d in range(depth_out, depth_stop-1, -1)])
        self.content_downsample = torch.nn.ModuleList([modules_bn.GraphDownsample(self.channels[d], self.channels[d-1]) for d in range(depth_out, depth_stop, -1)])


        self.conv1 = modules_bn.GraphConv(channel_in, self.channels[depth], n_edge_type, avg_degree, depth-1)
        self.encoder = torch.nn.ModuleList(
            [modules_bn.GraphResBlocks(self.channels[d] * (1+(d!=self.depth)), self.channels[d],self.dropout, self.resblk_nums[d], n_edge_type, avg_degree, d-1, self.channels[depth_stop]) 
            for d in range(depth, depth_stop-1, -1)])
        self.downsample = torch.nn.ModuleList([modules_bn.GraphDownsample(self.channels[d], self.channels[d-1]) for d in range(depth, depth_stop, -1)])

        self.encoder_mid = torch.nn.Module()
        self.encoder_mid.block_1 = modules_bn.GraphResBlocks(self.channels[depth_stop], self.channels[depth_stop],self.dropout,
            self.resblk_nums[depth_stop], n_edge_type, avg_degree, depth_stop-1, self.channels[depth_stop])
        self.encoder_mid.block_2 = modules_bn.GraphResBlocks(self.channels[depth_stop], self.channels[depth_stop],self.dropout,
            self.resblk_nums[depth_stop], n_edge_type, avg_degree, depth_stop-1, self.channels[depth_stop])
        self.encoder_norm_out = torch.nn.BatchNorm1d(self.channels[depth_stop])

        # decoder
        self.upsample = torch.nn.ModuleList([modules_bn.GraphUpsample(self.channels[d-1], self.channels[d]) for d in range(depth_stop+1, depth_out+1)])

        self.nonlinearity = torch.nn.GELU()

        self.decoder_mid = torch.nn.Module()
        self.decoder_mid.block_1 = modules_bn.GraphResBlocks(self.channels[depth_stop], self.channels[depth_stop], self.dropout,
        self.resblk_nums[depth_stop], n_edge_type, avg_degree, depth_stop-1, self.channels[depth_stop])
        self.decoder_mid.block_2 = modules_bn.GraphResBlocks(self.channels[depth_stop], self.channels[depth_stop], self.dropout,
        self.resblk_nums[depth_stop], n_edge_type, avg_degree, depth_stop-1, self.channels[depth_stop])

        self.decoder = torch.nn.ModuleList(
            [modules_bn.GraphResBlocks(self.channels[d], self.channels[d], self.dropout,
            self.resblk_nums[d], n_edge_type, avg_degree, d-1, self.channels[depth_stop])
            for d in range(depth_stop, depth_out + 1)])

        self.predict = torch.nn.ModuleList(
            [self._make_predict_module(self.channels[d], 2 if d != depth_out else 3) for d in range(depth_stop, depth_out + 1)])
       
        self.time_embedding = TimeEmbedding(self.channels[depth_stop])
        self.style_fc = torch.nn.Linear(2 * self.channels[depth_stop], self.channels[depth_stop]//2 if 'char' in self.cond else self.channels[depth_stop])
        self.style_maxpool = torch.nn.AdaptiveMaxPool1d(1)
        self.style_avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.content_fc = torch.nn.Linear(2 * self.channels[depth_stop], self.channels[depth_stop]//2 if 'font' in self.cond else self.channels[depth_stop])
        self.content_maxpool = torch.nn.AdaptiveMaxPool1d(1)
        self.content_avgpool = torch.nn.AdaptiveAvgPool1d(1)

    def octree_encoder(self, octree, doctree, timesteps, cond):
        depth, depth_stop = self.depth, self.depth_stop
        data = self._get_input_feature(doctree, True, self.inputdata)
        convs = dict()

        cond, content_convs, doctree_content = cond
        
        convs[depth] = data 
        for i, d in enumerate(range(depth, depth_stop-1, -1)): 

            if d < self.depth:
                content_skip = modules_bn.doctree_align(content_convs[d], doctree_content.graph[d]['keyd'], doctree.graph[d]['keyd'])
                convs[d] = torch.cat((convs[d], content_skip), -1)

            convd = convs[d] 

            edge_idx = doctree.graph[d]['edge_idx']
            edge_type = doctree.graph[d]['edge_dir']
            node_type = doctree.graph[d]['node_type']
            batch_id = (doctree.graph[d]['keyd'] % (1 << 58)) >> 48
            
            if d == self.depth:
                convd = self.conv1(convd, edge_idx, edge_type, node_type)


            temb = timesteps + cond
            temb = temb[batch_id]

            convd = self.encoder[i](convd, octree, d, edge_idx, edge_type, node_type, temb)
            convs[d] = convd

            if d > depth_stop:
                nnum = doctree.nnum[d]
                lnum = doctree.lnum[d-1]
                leaf_mask = doctree.node_child(d-1) < 0
                convs[d-1] = self.downsample[i](convd, octree, d-1, leaf_mask, nnum, lnum)

        batch_id = (doctree.graph[depth_stop]['keyd'] % (1 << 58)) >> 48

        temb = timesteps + cond
        temb = temb[batch_id]

        convs[depth_stop] = self.encoder_mid.block_1(convs[depth_stop], octree, depth_stop, edge_idx, edge_type, node_type, temb)
        convs[depth_stop] = self.encoder_mid.block_2(convs[depth_stop], octree, depth_stop, edge_idx, edge_type, node_type, temb)

        return convs[self.depth_stop], convs

    def octree_decoder(self, feature, convs, doctree, doctree_out, timesteps, cond, update_octree=False):
        octree_out = doctree_out.octree

        logits = dict()
        node_types = dict()
        deconvs = dict()

        depth_stop = self.depth_stop

        deconvs[depth_stop] = feature

        edge_idx = doctree_out.graph[depth_stop]['edge_idx']
        edge_type = doctree_out.graph[depth_stop]['edge_dir']
        node_type = doctree_out.graph[depth_stop]['node_type']

        batch_id = (doctree_out.graph[depth_stop]['keyd'] % (1 << 58)) >> 48
        temb = timesteps + cond
        temb = temb[batch_id]

        deconvs[depth_stop] = self.decoder_mid.block_1(deconvs[depth_stop], octree_out, depth_stop, edge_idx, edge_type, node_type, temb)
        deconvs[depth_stop] = self.decoder_mid.block_2(deconvs[depth_stop], octree_out, depth_stop, edge_idx, edge_type, node_type, temb)

        for i, d in enumerate(range(self.depth_stop, self.depth_out+1)):
            if d > self.depth_stop:
                nnum = doctree_out.nnum[d-1]
                leaf_mask = doctree_out.node_child(d-1) < 0                
                deconvs[d] = self.upsample[i-1](deconvs[d-1], octree_out, d, leaf_mask, nnum)

                if 'align' in self.cond:
                    if d <= self.depth:
                        skip = modules_bn.doctree_align(convs[d], doctree.graph[d]['keyd'], doctree_out.graph[d]['keyd'])
                        deconvs[d] += skip

            edge_idx = doctree_out.graph[d]['edge_idx']
            edge_type = doctree_out.graph[d]['edge_dir']
            node_type = doctree_out.graph[d]['node_type']
            batch_id = (doctree_out.graph[d]['keyd'] % (1 << 58)) >> 48

            temb = timesteps + cond
            temb = temb[batch_id]

            deconvs[d] = self.decoder[i](deconvs[d], octree_out, d, edge_idx, edge_type, node_type, temb)

            # predict the splitting label
            logit = self.predict[i]([deconvs[d], octree_out, d])

            nnum = doctree_out.nnum[d]
            logits[d] = logit[-nnum:]

            # update the octree according to predicted labels

            if update_octree: 
                label = (logits[d].argmax(-1) > 0).to(torch.int32)

                octree_out = doctree_out.octree
                octree_out.octree_split(label, d)

                if d < self.depth_out:
                    octree_out.octree_grow(d + 1) 
                    octree_out.depth += 1

                doctree_out = dual_octree.DualOctree(octree_out)
                doctree_out.post_processing_for_docnn()

            # predict the signal
            node_type = doctree_out.graph[d]['node_type']

        return logits, doctree_out.octree, node_types

    def feature_encoder(self, octree_style, doctree_style, octree_content, doctree_content):
        depth_out, depth_stop = self.depth_out, self.depth_stop
        # style
        if 'font' in self.cond:
            style_list = []
            for idx in range(len(doctree_style)):
                data = self._get_input_feature(doctree_style[idx], True, self.inputdata)

                style_convs = dict()
                
                style_convs[depth_out] = data 
                for i, d in enumerate(range(depth_out, depth_stop-1, -1)):
                    convd = style_convs[d] 
                    edge_idx = doctree_style[idx].graph[d]['edge_idx']
                    edge_type = doctree_style[idx].graph[d]['edge_dir']
                    node_type = doctree_style[idx].graph[d]['node_type']
                    batch_id = (doctree_style[idx].graph[d]['keyd'] % (1 << 58)) >> 48
                    
                    if d == self.depth_out: 
                        convd = self.style_conv(convd, edge_idx, edge_type, node_type)

                    convd = self.style_encoder[i](convd, octree_style[idx], d, edge_idx, edge_type, node_type)
                    style_convs[d] = convd 

                    if d > depth_stop:
                        nnum = doctree_style[idx].nnum[d]
                        lnum = doctree_style[idx].lnum[d-1]
                        leaf_mask = doctree_style[idx].node_child(d-1) < 0
                        style_convs[d-1] = self.style_downsample[i](convd, octree_style[idx], d-1, leaf_mask, nnum, lnum)
                style = style_convs[depth_stop]
                style_list.append(style)
            style = torch.stack(style_list, 0)
        else:
            style = None

        # content
        if 'char' in self.cond:
            data = self._get_input_feature(doctree_content, True, self.inputdata)

            content_convs = dict()
            
            content_convs[depth_out] = data 
            for i, d in enumerate(range(depth_out, depth_stop-1, -1)):   
                # perform graph conv
                convd = content_convs[d] 
                edge_idx = doctree_content.graph[d]['edge_idx']
                edge_type = doctree_content.graph[d]['edge_dir']
                node_type = doctree_content.graph[d]['node_type']
                batch_id = (doctree_content.graph[d]['keyd'] % (1 << 58)) >> 48
                
                if d == self.depth_out: 
                    convd = self.content_conv(convd, edge_idx, edge_type, node_type)

                convd = self.content_encoder[i](convd, octree_content, d, edge_idx, edge_type, node_type)
                content_convs[d] = convd

                if d > depth_stop:
                    nnum = doctree_content.nnum[d]
                    lnum = doctree_content.lnum[d-1]
                    leaf_mask = doctree_content.node_child(d-1) < 0
                    content_convs[d-1] = self.content_downsample[i](convd, octree_content, d-1, leaf_mask, nnum, lnum)
            content = content_convs
        else:
            content = None

        return style, content

    def get_feature(self, octree_style, octree_content):
        doctree_style_list = []
        for i in range(len(octree_style)):
            doctree_style = dual_octree.DualOctree(octree_style[i])
            doctree_style.post_processing_for_docnn()
            doctree_style_list.append(doctree_style)
        doctree_style = doctree_style_list
        doctree_content = dual_octree.DualOctree(octree_content)
        doctree_content.post_processing_for_docnn()

        style_feature, content_feature = self.feature_encoder(octree_style, doctree_style, octree_content, doctree_content)
        ref_num = style_feature.shape[0]
        style_feature = style_feature.view(-1, (2**self.depth_stop)**2, self.channels[self.depth_stop]).permute(0, 2, 1)
        style_feature = torch.cat((self.style_maxpool(style_feature), self.style_avgpool(style_feature)), 1).squeeze(-1)
        style_feature = self.style_fc(style_feature)
        style_feature = style_feature.view(ref_num, -1, style_feature.shape[-1])
        style_feature = style_feature.mean(0)

        content_convs = content_feature
        content_feature = content_feature[self.depth_stop]
        content_feature = content_feature.view(-1, (2**self.depth_stop)**2, self.channels[self.depth_stop]).permute(0, 2, 1)
        content_feature = torch.cat((self.content_maxpool(content_feature), self.content_avgpool(content_feature)), 1).squeeze(-1)
        content_feature = self.content_fc(content_feature)

        return style_feature, content_feature, content_convs, doctree_content

    def forward(self, octree_in, timesteps, octree_style=None, octree_content=None, octree_out=None, pos=None, val_gen=False):
        doctree_style_list = []
        for i in range(len(octree_style)):
            doctree_style = dual_octree.DualOctree(octree_style[i])
            doctree_style.post_processing_for_docnn()
            doctree_style_list.append(doctree_style)
        doctree_style = doctree_style_list
        doctree_content = dual_octree.DualOctree(octree_content)
        doctree_content.post_processing_for_docnn()
        
        doctree_in = dual_octree.DualOctree(octree_in)
        doctree_in.post_processing_for_docnn()
        
        update_octree = octree_out is None
        if update_octree:
            octree_out = self.create_full_octree(octree_content)
            octree_out.depth = self.full_depth
        doctree_out = dual_octree.DualOctree(octree_out)
        doctree_out.post_processing_for_docnn()

        style_feature, content_feature = self.feature_encoder(octree_style, doctree_style, octree_content, doctree_content)

        if 'font' in self.cond and 'char' in self.cond:
            style_feature, content_feature = self.feature_encoder(octree_style, doctree_style, octree_content, doctree_content)
            ref_num = style_feature.shape[0]
            style_feature = style_feature.view(-1, (2**self.depth_stop)**2, self.channels[self.depth_stop]).permute(0, 2, 1)
            style_feature = torch.cat((self.style_maxpool(style_feature), self.style_avgpool(style_feature)), 1).squeeze(-1)
            style_feature = self.style_fc(style_feature)
            style_feature = style_feature.view(ref_num, -1, style_feature.shape[-1])
            style_feature = style_feature.mean(0)

            content_convs = content_feature
            content_feature = content_feature[self.depth_stop]
            content_feature = content_feature.view(-1, (2**self.depth_stop)**2, self.channels[self.depth_stop]).permute(0, 2, 1)
            content_feature = torch.cat((self.content_maxpool(content_feature), self.content_avgpool(content_feature)), 1).squeeze(-1)
            content_feature = self.content_fc(content_feature)
            
            cond = torch.cat((content_feature, style_feature), -1)
            cond = (cond, content_convs, doctree_content)
        elif 'font' in self.cond:
            style_feature = style_feature.view(-1, (2**self.depth_stop)**2, self.channels[self.depth_stop]).permute(0, 2, 1)
            style_feature = torch.cat((self.style_maxpool(style_feature), self.style_avgpool(style_feature)), 1).squeeze(-1)
            style_feature = self.style_fc(style_feature)
            cond = style_feature
        elif 'char' in self.cond:
            content_feature = content_feature.view(-1, (2**self.depth_stop)**2, self.channels[self.depth_stop]).permute(0, 2, 1)
            content_feature = torch.cat((self.content_maxpool(content_feature), self.content_avgpool(content_feature)), 1).squeeze(-1)
            content_feature = self.content_fc(content_feature)
            cond = content_feature
        
        timesteps = self.time_embedding(timesteps)

        feature, convs = self.octree_encoder(octree_in, doctree_in, timesteps, cond)

        out = self.octree_decoder(feature, convs, doctree_in, doctree_out, timesteps, cond[0], update_octree)

        output = {'logits': out[0], 'octree_out': out[1], 'node_types': out[2]}
        return output

    def generate(self, octree_in, timesteps, cond, octree_content):
        doctree_in = dual_octree.DualOctree(octree_in)
        doctree_in.post_processing_for_docnn()
        
        octree_out = self.create_full_octree(octree_in)
        octree_out.depth = self.full_depth
        doctree_out = dual_octree.DualOctree(octree_out)
        doctree_out.post_processing_for_docnn()
        
        timesteps = self.time_embedding(timesteps)

        feature, convs = self.octree_encoder(octree_in, doctree_in, timesteps, cond)

        out = self.octree_decoder(feature, convs, doctree_in, doctree_out, timesteps, cond[0], update_octree=True)

        output = {'logits': out[0], 'octree_out': out[1], 'node_types': out[2]}
        return output

    def create_full_octree(self, octree_in: Octree):
        ''' Initialize a full octree for decoding. '''

        device = octree_in.device
        batch_size = octree_in.batch_size
        octree = Octree(self.depth_out, self.full_depth, batch_size, device)
        for d in range(self.full_depth+1):
            octree.octree_grow_full(depth=d)
        return octree

    def _setup_channels_and_resblks(self):
        self.resblk_nums = [self.resblk_num] * 16  
        self.channels = [3, 512, 512, 256, 512, 256, 128, 64, 64, 64]  # 128, 256
        # self.channels = [3, 512, 512, 256, 512, 256, 256, 128, 128, 64]  # 512

    def _make_predict_module(self, channel_in, channel_out=2, num_hidden=32):
        return torch.nn.Sequential(
        modules_bn.Conv1x1BnGeluSequential(channel_in, num_hidden),
        modules_bn.Conv1x1(num_hidden, channel_out, use_bias=True))

    def _get_input_feature(self, doctree, all_leaf_nodes=True, label='L'):
        return doctree.get_input_feature(all_leaf_nodes, label)
