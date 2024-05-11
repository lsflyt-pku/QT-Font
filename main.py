# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import copy
import numpy as np
import cv2 as cv

import builder
from solver import Solver, get_config
import ocnn_2d
from ocnn_2d.modules import InputFeature

from ocnn_2d.octree import Octree, Points
from diffusers import DDPMScheduler, DDIMScheduler
from tqdm import tqdm
from losses import leaf_pixel_loss, axis_loss

torch.multiprocessing.set_sharing_strategy('file_system')


class DualOcnnSolver(Solver):

    def get_model(self, flags):
        return builder.get_model(flags)

    def get_dataset(self, flags):
        return builder.get_dataset(flags)

    def batch_to_cuda(self, batch):
        keys = ['octree_content', 'octree_style', 'octree_in', 'octree_gt', 'pos', 'sdf', 'grad', 'weight', 'occu', 'timesteps', 'font', 'char']
        for key in ['font', 'char', 'timesteps']:
            if key in batch:
                batch[key] = torch.stack(batch[key])
        for key in keys:
            if key in batch:
                if type(batch[key]) == list:
                    for i in range(len(batch[key])):
                        batch[key][i] = batch[key][i].cuda()
                else:
                    batch[key] = batch[key].cuda()
        batch['pos'].requires_grad_()

    def compute_loss(self, batch, model_out):
        flags = self.FLAGS.LOSS
        loss_func = builder.get_loss_function(flags)
        output = loss_func(batch, model_out)
        return output

    def model_forward(self, batch, val_gen=False):
        self.batch_to_cuda(batch)
        
        model_out = self.model(batch['octree_in'], batch['timesteps'], batch['octree_style'], batch['octree_content'], batch['octree_gt'], batch['pos'], val_gen=val_gen)

        output = self.compute_loss(batch, model_out)
        losses = [val for key, val in output.items() if 'loss' in key]
        output['loss'] = torch.sum(torch.stack(losses))
        if 'code_max' in model_out:
            output['code_max'] = model_out['code_max']
        if 'code_min' in model_out:
            output['code_min'] = model_out['code_min']
        return output

    def train_step(self, batch):
        output = self.model_forward(batch)
        output = {'train/' + key: val for key, val in output.items()}    # 这里把output字典里所有的key都加上了 train/ 前缀
        return output

    def test_step(self, batch):    # 在test的时候也是会输入八叉树的GT的
        output = self.model_forward(batch, val_gen=True)
        output = {'test/' + key: val for key, val in output.items()}
        return output

    def eval_step(self, batch):    # 在eval的时候则不会输入八叉树的GT，完全是通过网络预测的标签进行八叉树的生长
        self.batch_to_cuda(batch)
        style_feature, content_feature = self.model.get_feature(batch['octree_style'], batch['octree_content'])
        return style_feature, content_feature

    def discrete_step(self, batch):
        self.batch_to_cuda(batch)
        show_input = True
        style_image = {}
        content_image = {}
        gt_image = {}

        if show_input:
            octree_feature = InputFeature(feature = 'F', nempty=False)
            batch_size = len(batch['filename'])
            octree_out = batch['octree_gt']
            max_depth = octree_out.depth
            image_size = 2 ** max_depth
    
            pixel_label = octree_feature(octree_out)
            # pdb.set_trace()
            pixel_label = pixel_label[:, 2] * 1 + pixel_label[:, 3] * 2
            # pixel_label = octree_out.nempty_mask(max_depth).float()
            points = octree_out.xyzb(max_depth)
            for i in range(batch_size):
                x = points[0]
                y = points[1]
                b = points[2]
                batch_mask = torch.nonzero(b==i, as_tuple=False)
                x = x[batch_mask]
                y = y[batch_mask]
                fake = torch.zeros((image_size, image_size))
                fake[x, y] = pixel_label[batch_mask].cpu()
                gt_image[i] = fake.numpy()
                fake = fake.numpy() * 127.5
                fake = fake.astype(np.uint8)
                # cv.imwrite('gt_{}.png'.format(i), fake)

            octree_out = batch['octree_style'][0]
            max_depth = octree_out.depth
            image_size = 2 ** max_depth
            
            pixel_label = octree_out.nempty_mask(max_depth).float()
            points = octree_out.xyzb(max_depth)
            for i in range(batch_size):
                x = points[0]
                y = points[1]
                b = points[2]
                batch_mask = torch.nonzero(b==i, as_tuple=False)
                x = x[batch_mask]
                y = y[batch_mask]
                fake = torch.zeros((image_size, image_size))
                fake[x, y] = pixel_label[batch_mask].cpu()
                style_image[i] = fake.numpy()
                fake = fake.numpy() * 255
                # cv.imwrite('style_{}.png'.format(i), fake)

            octree_out = batch['octree_content']
            max_depth = octree_out.depth
            
            pixel_label = octree_out.nempty_mask(max_depth).float()
            points = octree_out.xyzb(max_depth)
            
            for i in range(batch_size):
                x = points[0]
                y = points[1]
                b = points[2]
                batch_mask = torch.nonzero(b==i, as_tuple=False)
                x = x[batch_mask]
                y = y[batch_mask]
                fake = torch.zeros((image_size, image_size))
                fake[x, y] = pixel_label[batch_mask].cpu()
                content_image[i] = fake.numpy()
                fake = fake.numpy() * 255
                # cv.imwrite('content_{}.png'.format(i), fake)

        num_train_timesteps = 1000
        num_classes = 3

        device = batch['timesteps'].device
        octree_style = batch['octree_style'][0]
        max_depth = octree_style.depth
        full_depth = octree_style.full_depth
        image_size = 2 ** max_depth
        mid = (image_size - 1.) / 2.
        batch_size = len(batch['filename'])

        steps = np.arange(num_train_timesteps + 1, dtype=np.float64) / num_train_timesteps
        alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
        betas = np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
        betas = torch.tensor(betas).to(device)

        Q = torch.zeros([num_train_timesteps, num_classes, num_classes], dtype=torch.float32).to(device)
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    Q[:, i, j] = 1 - betas  * (num_classes-1) / num_classes
                else:
                    Q[:, i, j] = betas / num_classes

        Q_T = Q.permute(0, 2, 1)

        Q_ = torch.ones([num_train_timesteps, num_classes, num_classes], dtype=torch.float32).to(device)
        Q_[0] = Q[0]
        for i in range(1, num_train_timesteps):
            Q_[i] = torch.matmul(Q_[i-1], Q[i])

        def _at(a, t, x):
            # print('_at', a.shape, t.shape, x.shape)
            t_broadcast = np.expand_dims(t, tuple(range(1, x.ndim)))
            return a[t_broadcast, x]

        def _at_onehot(a, t, x):
            # print('_at_onehot', a.shape, t.shape, x.shape)
            return torch.matmul(x.unsqueeze(1), a[t]).squeeze(1)

        def q_probs(x_0, t):
            '''
            x_0: L
            t: L
            '''
            # print('q_probs', x_0.shape, t.shape)
            return _at(Q_, t, x_0)

        def q_sample(x_0, t, noise):
            '''
            Compute logits of q(x_t | x_start).
            x_0: L
            noise: Lx2
            '''
            # print('q_sample', x_0.shape, t.shape, x_0.shape)
            logits = torch.log(q_probs(x_0, t) + 1e-8)
            noise = torch.clip(noise, min=1e-8, max=1.0)
            gumbel_noise = -torch.log(-torch.log(noise))
            return torch.argmax(logits+gumbel_noise, -1)

        def q_posterior_logits(x_start, x_t, t, x_start_logits):
            '''
            Compute logits of q(x_{t-1} | x_t, x_start).
            '''
            # print('q_posterior_logits', x_start.shape, x_t.shape, t.shape, x_start_logits)
            if x_start_logits:
                assert x_start.shape == x_t.shape + (3,), (x_start.shape, x_t.shape)
            else:
                assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)

            if gap != 1:
                Q_tmp = torch.ones([1000, num_classes, num_classes], dtype=torch.float32).to(device)
                for gap_t in range(t[0]-gap+1, t[0]+1):
                    Q_tmp[gap_t] = torch.matmul(Q_tmp[gap_t-1], Q[gap_t])
                Q_tmp = Q_tmp.permute(0, 2, 1)
                fact1 = _at(Q_tmp, t, x_t)
            else:
                fact1 = _at(Q_T, t, x_t)
            if x_start_logits:
                fact2 = _at_onehot(Q_, t-gap, torch.softmax(x_start, axis=-1))
                tzero_logits = x_start
            else:
                fact2 = _at(Q_, t-gap, x_start)
                tzero_logits = torch.log(torch.nn.functional.one_hot(x_start, num_classes) + 1e-8)

            # At t=0 we need the logits of q(x_{-1}|x_0, x_start)
            # where x_{-1} == x_start. This should be equal the log of x_0.
            out = torch.log(fact1 + 1e-8) + torch.log(fact2 + 1e-8)
            t_broadcast = np.expand_dims(t, tuple(range(1, out.ndim)))
            return torch.where(torch.tensor(t_broadcast, device=x_start.device) == 0, tzero_logits, out)

        def p_logits(model_out, x, t):
            '''
            Compute logits of p(x_{t-1} | x_t).
            '''
            # print('p_logits', model_out.shape, x.shape, t.shape)
            model_logits = model_out
            pred_x_start_logits = model_logits
            t_broadcast = np.expand_dims(t, tuple(range(1, model_logits.ndim)))
            model_logits = torch.where(torch.tensor(t_broadcast, device=x.device) == 0, pred_x_start_logits, q_posterior_logits(pred_x_start_logits, x, t, x_start_logits=True))

            return model_logits, pred_x_start_logits

        def p_sample(model_out, x, t, noise):
            '''
            Sample one timestep from the model p(x_{t-1} | x_t).
            '''
            # print('p_sample', model_out.shape, x.shape, t.shape, noise.shape)
            model_logits, pred_x_start_logits = p_logits(model_out, x, t)
            assert noise.shape == model_logits.shape, noise.shape
            nonzero_mask = torch.tensor((t != 0), dtype=x.dtype, device=x.device).reshape(x.shape[0], *([1] * (len(x.shape))))
            # For numerical precision clip the noise to a minimum value
            noise = torch.clip(noise, 1e-8, 1.)
            gumbel_noise = -torch.log(-torch.log(noise))
            sample = torch.argmax(model_logits + nonzero_mask * gumbel_noise, axis=-1)

            assert sample.shape == x.shape
            assert pred_x_start_logits.shape == model_logits.shape
            return sample, torch.softmax(pred_x_start_logits, axis=-1)   


        x_t = torch.randint(low=0, high=num_classes, size=(batch_size, image_size, image_size), device=device)
        
        L = batch_size * (image_size ** 2)
        x_0_list = {}
        x_t_list = {}
        module_flag = type(self.model) == torch.nn.parallel.DistributedDataParallel

        gap = 50 # timesteps=1000/50=20
        if module_flag:
            style_feature, content_feature, content_convs, doctree_content = self.model.module.get_feature(batch['octree_style'], batch['octree_content'])
        else:
            style_feature, content_feature, content_convs, doctree_content = self.model.get_feature(batch['octree_style'], batch['octree_content'])
        cond = torch.cat((content_feature, style_feature), -1)
        cond = (cond, content_convs, doctree_content)
        # pdb.set_trace()
        for t in tqdm(range(num_train_timesteps-gap, -1, -gap)):
            # x_t bxhxw
            x_t = x_t.view(batch_size, image_size, image_size)

            if t % 10 == 0 or t > 990:
                x_t_list[t] = x_t[0].cpu().numpy()

            octree_in_list = []

            for b in range(batch_size):
                noise_image = x_t[b]

                if max_depth != 7:
                    weight = torch.zeros((128, 128, 3))
                    noise_image = noise_image.view(128, image_size // 128, 128, image_size // 128).permute(0, 2, 1, 3).reshape(128, 128, -1)
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
                for label in range(1, num_classes):
                    noise_points.append(torch.stack(torch.where(noise_image==label), -1))
                    noise_feature.append(torch.zeros((noise_points[-1].shape[0], num_classes-1)))
                    noise_feature[-1][:, label-1] = 1
                noise_points = torch.concat(noise_points, 0)
                noise_feature = torch.concat(noise_feature, 0)
                noise_points = (noise_points - mid) / mid
                noise_points = noise_points.to(device)
                noise_feature = noise_feature.to(device)

                points_in = Points(points=noise_points.detach(), features=torch.cat((noise_points, noise_feature), -1))
                octree_in = Octree(7, full_depth) # depth, fulldepth
                octree_in.build_octree(points_in)
                octree_in = octree_in.to(device)
                octree_in_list.append(copy.deepcopy(octree_in))
            
            octree_in = ocnn_2d.octree.merge_octrees(octree_in_list)

            timesteps = torch.tensor([t] * batch_size).to(device)
            
            if module_flag:
                output = self.model.module.generate(octree_in, timesteps, cond, batch['octree_content'])
            else:
                output = self.model.generate(octree_in, timesteps, cond, batch['octree_content'])

            octree_out = output['octree_out']

            max_depth = octree_out.depth
            depth_stop = octree_out.full_depth
            
            logits = output['logits'][max_depth]

            gen_points = octree_out.xyzb(max_depth)

            x = gen_points[0]
            y = gen_points[1]
            batch_id = octree_out.batch_id(max_depth)

            fake_list = []

            for b in range(batch_size):
                batch_mask = torch.nonzero(batch_id==b, as_tuple=False)

                batch_x = x[batch_mask]
                batch_y = y[batch_mask]

                fake = torch.zeros((image_size, image_size, num_classes)).float().to(device)
                fake[:, :, 1] = torch.log(fake[:, :, 1]+1e-8)
                fake[:, :, 2] = torch.log(fake[:, :, 2]+1e-8)
                fake[batch_x, batch_y, :] = logits[batch_mask]

                fake_list.append(fake.detach().clone())

            x_0 = torch.stack(fake_list, 0)
            x_0_image = torch.argmax(x_0, -1)
            x_0_image = x_0_image.view(batch_size, image_size, image_size)

            if t % 10 == 0 or t > 990:
                x_0_list[t] = x_0_image[0].cpu().numpy()

            L = batch_size * image_size * image_size
            noise = torch.rand((L, num_classes), device=device)
            x_0 = x_0.view(-1, num_classes)
            x_t = x_t.view(-1)
            x_t, _ = p_sample(x_0, x_t, np.array([t]*L), noise)

        def get_image(image, num_classes):
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            H, W = image.shape
            new_image = np.zeros((H, W, 3))
            for label in range(1, num_classes):
                new_image[image==label] = color[label]
            return new_image

        if True:
            save_image = x_t.view(batch_size, image_size, image_size).cpu().numpy()

            for b in range(batch_size):
                filename = batch['filename'][b]
                pos = filename.rfind('.')
                if pos != -1: filename = filename[:pos]    # remove the suffix
                filename = os.path.join(self.logdir, 'results', filename + '.png')
                folder = os.path.dirname(filename)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                image = np.concatenate((
                get_image(save_image[b, :, :], num_classes),
                get_image(content_image[b], num_classes),
                get_image(style_image[b], num_classes),
                get_image(gt_image[b], num_classes)), 1
                )
                cv.imwrite(filename, image)

    def save_tensors(self, batch, output):
        iter_num = batch['iter_num']
        filename = os.path.join(self.logdir, '%04d.out.octree' % iter_num)
        output['octree_out'].cpu().numpy().tofile(filename)
        filename = os.path.join(self.logdir, '%04d.in.octree' % iter_num)
        batch['octree_in'].cpu().numpy().tofile(filename)
        filename = os.path.join(self.logdir, '%04d.in.points' % iter_num)
        batch['points_in'][0].cpu().numpy().tofile(filename)
        filename = os.path.join(self.logdir, '%04d.gt.octree' % iter_num)
        batch['octree_gt'].cpu().numpy().tofile(filename)
        filename = os.path.join(self.logdir, '%04d.gt.points' % iter_num)
        batch['points_gt'][0].cpu().numpy().tofile(filename)

    @classmethod
    def update_configs(cls):
        FLAGS = get_config()
        FLAGS.SOLVER.resolution = 128             # the resolution used for marching cubes
        FLAGS.SOLVER.save_sdf = False             # save the sdfs in evaluation
        FLAGS.SOLVER.sdf_scale = 0.9                # the scale of sdfs

        FLAGS.DATA.train.point_scale = 0.5    # the scale of point clouds
        FLAGS.DATA.train.load_sdf = True        # load sdf samples
        FLAGS.DATA.train.load_occu = False    # load occupancy samples
        FLAGS.DATA.train.point_sample_num = 10000
        FLAGS.DATA.train.sample_surf_points = False

        # FLAGS.MODEL.skip_connections = True
        FLAGS.DATA.test = FLAGS.DATA.train.clone()
        FLAGS.LOSS.loss_type = 'sdf_reg_loss'
        FLAGS.LOSS.codebook_weight = 1.0
        FLAGS.LOSS.kl_weight = 1.0


if __name__ == '__main__':
    DualOcnnSolver.main()
