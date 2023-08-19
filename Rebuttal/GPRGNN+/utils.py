#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag is 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data


def GTV (y, z):
    lamda = 0.3
    alpha = 8   #.5
    T = lamda/2
    z = z.to(y.device) 
    ##for feature diff
    # diff = z[:, 1:] - z[:, 0:-1]
    # append = torch.cat((z[:,0:1], diff, z[:,-2:-1]), 1)
    # x = y - append
    # xdiff = x[:, 1:]-x[:, 0:-1]
    # zt = z + 1 / alpha * xdiff
    # TT = torch.zeros(z.shape[1]) + T
    # zt = torch.maximum(torch.minimum(zt[:,0:], TT), -TT)
    ##for node diff
    diff = z[1:,:] - z[0:-1,:]
    append = torch.cat((z[0:1,:],-diff,z[-2:-1,:]),0)
    x = y - append
    xdiff = x[1:,:]-x[0:-1,:]
    zt = z + 1/alpha*xdiff
    TT = torch.zeros(z.shape[1],device=zt.device) + T
    zt = torch.maximum(torch.minimum(zt[0:, :], TT), -TT)
    return x, zt
