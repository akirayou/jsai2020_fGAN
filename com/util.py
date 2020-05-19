#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyrihgt Shimadzu Corp.
This demonstration code can only be used for research purposes 
(Includes academic education). Commercial use is prohibited. 
Distribution of derivative software  for your research is permitted 
if the distribution is limited to research use. 
Other redistribution is prohibited.
Contact: a-noda@shimadzu.co.jp  

Created on Fri Jan 24 14:21:08 2020

@author: La-noda
"""
import chainer
import argparse
import os
import warnings
import numpy
def get_input(dummyData):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dummy_pattern', '-p', type=int, default=-1,
                        help="see dummyData")
    parser.add_argument('--z_order', '-z', type=int, default=2,
                        help="")
    parser.add_argument('--batchsize', '-b', type=int, default=10000,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100000,)
    parser.add_argument('--device', '-d', type=str, default='2')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str,
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval','-S', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')


    args = parser.parse_args()
    if chainer.get_dtype() == numpy.float16:
        warnings.warn(
            'This example may cause NaN in FP16 mode.', RuntimeWarning)

    device = chainer.get_device(args.device)
    device.use()

    print('Device: {}'.format(device))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')




    ###########################################
    ###Make dummy dataset
    ##########################################
    if args.dummy_pattern >= 0 :
        dummyData.set_pattern(args.dummy_pattern)
    nof_dummy_data=dummyData.NOF_DATA
    args.nof_pack=dummyData.NOF_PACK
    d,z=dummyData.makeDummy(nof_dummy_data,args.nof_pack)
    d=d.astype(chainer.get_dtype())

    l=numpy.array([0]*nof_dummy_data)
    l=l.reshape(-1,1)
    train=chainer.datasets.TupleDataset(d,l)
    train=d 
    ######################################
    args.out+="_z"+str(args.z_order)+"_npk"+str(args.nof_pack)+"_pat"+str(dummyData.DUMMY_PAT)
    
    return args,train,device

    