#!/usr/bin/env python
"""
Copyrihgt Shimadzu Corp.
This demonstration code can only be used for research purposes 
(Includes academic education). Commercial use is prohibited. 
Distribution of derivative software  for your research is permitted 
if the distribution is limited to research use. 
Other redistribution is prohibited.
Contact: a-noda@shimadzu.co.jp  

"""
import argparse
import os
import warnings

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
import net
from net import Generator
from net import Discriminator

from updater_for_check_generator import DCGANUpdater
from visualize import out_generated_image
import chainer.functions as F

import chainer.links as L

def main():



    nof_mesure=5
    nof_pack=2
    z_order=4
    x_order=3
    dtype=chainer.get_dtype()
    net.CHECK_INPUT=nof_mesure*nof_pack
    
    x=np.ones( (nof_mesure,nof_pack*x_order),dtype)
    for i in range(x_order):
        x[:,i::x_order]=i
    for i in range(nof_pack):
        x[:,i*x_order:(i+1)*x_order]+=i*10
    
    y=np.ones((nof_mesure,nof_pack),dtype)
    for i in range(nof_pack):
        y[:,i]=100*i
    
    z=np.ones((nof_mesure,z_order),dtype)
    for i in range(z_order):
        z[:,i]=1000*i
    
    for i in range(nof_mesure):
        x[i,:]+=10000*i
        y[i,:]+=10000*i
        z[i,:]+=10000*i

    
    gen = Generator(x_order=x_order,nof_pack=nof_pack, z_order=z_order ,wscale=0.5)
    x=chainer.Variable(x)
    z=chainer.Variable(z)
    gen(x,z)
    
    xy=F.concat([x,y])
    dis = Discriminator(inten_sigma=-0.01,nof_pack=nof_pack,x_order=x_order)
    dis(xy)
    
if __name__ == '__main__':
    main()
