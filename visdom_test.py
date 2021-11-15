# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:24:49 2020

@author: chenlin
"""

import torch
from visdom import Visdom
import numpy as np

from visdom import Visdom

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
    #plot for imagetensor
    def plot_images(self, var_name, split_name, title_name, img_tensor):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(img_tensor, env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.images(img_tensor, env=self.env, win=self.plots[var_name],
                            opts=dict(
                            legend=[split_name],
                            title=title_name,
                            xlabel='Epochs',
                            ylabel=var_name
                        ))
    def plot_embeddings(self, var_name, split_name, title_name, input_tensor, tensor_labels):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.embeddings(input_tensor, tensor_labels, env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.embeddings(input_tensor, tensor_labels, env=self.env, win=self.plots[var_name], 
                                name=split_name, opts=dict( legend=[split_name],
                                                            title=title_name,
                                                            xlabel='Epochs',
                                                            ylabel=var_name
                                                        ))
    def plot_histogram(self, var_name, split_name, title_name, input_tensor):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.histogram(X=input_tensor, env=self.env, opts=dict(
                numbins=40
            ))
        else:
            self.plots[var_name] = self.viz.histogram(X=input_tensor, env=self.env, 
                      win=self.plots[var_name], opts=dict(
                        numbins=40
                    ))
            
    def plot_surf(self, var_name, split_name, title_name, input_tensor):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.surf(X=input_tensor, env=self.env)
        else:
            self.plots[var_name] = self.viz.surf(X=input_tensor, env=self.env, 
                      win=self.plots[var_name])

#global plotter
#plotter = VisdomLinePlotter(env_name='Tutorial Plots')
#
#xx = torch.randn(10,10)
#plotter.plot_surf('gradient_w1', 'train', 'gradients', xx )

##viz = Visdom(env='demo')
##
##array = np.random.rand(10)
##
##viz.line(Y=array, X=np.arange(10))
#
#
#dtype = torch.float
#device = torch.device("cpu")
## device = torch.device("cuda:0") # Uncomment this to run on GPU
#
## N is batch size; D_in is input dimension;
## H is hidden dimension; D_out is output dimension.
#N, D_in, H, D_out = 64, 1000, 100, 10
#
## Create random Tensors to hold input and outputs.
## Setting requires_grad=False indicates that we do not need to compute gradients
## with respect to these Tensors during the backward pass.
#x = torch.randn(N, D_in, device=device, dtype=dtype)
#y = torch.randn(N, D_out, device=device, dtype=dtype)
#
## Create random Tensors for weights.
## Setting requires_grad=True indicates that we want to compute gradients with
## respect to these Tensors during the backward pass.
#w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
#w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
#
#learning_rate = 1e-6
#for t in range(10000):
#    # Forward pass: compute predicted y using operations on Tensors; these
#    # are exactly the same operations we used to compute the forward pass using
#    # Tensors, but we do not need to keep references to intermediate values since
#    # we are not implementing the backward pass by hand.
#    y_pred = x.mm(w1).clamp(min=0).mm(w2)
#
#    # Compute and print loss using operations on Tensors.
#    # Now loss is a Tensor of shape (1,)
#    # loss.item() gets the scalar value held in the loss.
#    loss = (y_pred - y).pow(2).sum()
#    if t % 100 == 99:
#        print(t, loss.item())
#
#    # Use autograd to compute the backward pass. This call will compute the
#    # gradient of loss with respect to all Tensors with requires_grad=True.
#    # After this call w1.grad and w2.grad will be Tensors holding the gradient
#    # of the loss with respect to w1 and w2 respectively.
#    loss.backward()
#
#    # Manually update weights using gradient descent. Wrap in torch.no_grad()
#    # because weights have requires_grad=True, but we don't need to track this
#    # in autograd.
#    # An alternative way is to operate on weight.data and weight.grad.data.
#    # Recall that tensor.data gives a tensor that shares the storage with
#    # tensor, but doesn't track history.
#    # You can also use torch.optim.SGD to achieve this.
#    with torch.no_grad():
#        w1 -= learning_rate * w1.grad
#        w2 -= learning_rate * w2.grad
#
#        # Manually zero the gradients after updating weights
#        w1.grad.zero_()
#        w2.grad.zero_()
#    if t%10 == 0:
#        plotter.plot('loss', 'train', 'Class Loss', t, loss.detach().numpy())
#        plotter.plot_images('images', 'train', 'images', x.view(64,1,20,50) )
#        plotter.plot_histogram('gradient_w1', 'train', 'gradients', y_pred.view(-1) )
##        plotter.plot_embeddings()
##        plotter.plot('y_pred', 'train', 'y prediction value', t, w1.grad)
##    viz.line(Y=loss.detach().numpy().reshape(1),X=np.array(t).reshape(1))