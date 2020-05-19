#!/usr/bin/env python
"""
Original code is https://github.com/chainer/chainer/blob/master/examples/dcgan/visualize.py

"""
import os

import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.backends.cuda
from chainer import Variable
import matplotlib.cm as cm
def out_generated_image(gen,nof_pack_org , seed, dst,zrate=1):
    @chainer.training.make_extension()
    def make_image(trainer):
        make_image.count+=1
        if(make_image.count%10==9):plt.close("all")
        np.random.seed(seed)
        xp = gen.xp
        row=7
        col=7
        t_len=row*col
        z=zrate*np.random.randn(t_len, gen.z_order).astype( chainer.get_dtype())
        
        nof_pack=40        
        x_in=np.linspace(-2,2,nof_pack)
        nof_pack=nof_pack**2
        x_np=np.array(np.meshgrid(x_in,x_in)).transpose( (1,2,0)).reshape(-1,2)
        x_np=x_np.reshape(1,-1)*np.ones((t_len,1))        
        x_np=x_np.astype(chainer.get_dtype())
       
        
        with chainer.using_config('train', False):
            x = Variable(xp.asarray(x_np))
            z = Variable(xp.asarray(z))
            y = gen.forward_for_show(x,z)
        y = chainer.backends.cuda.to_cpu(y.array)
        np.random.seed()
        x=x_np
        
        
        fig, ax = plt.subplots(row, col, sharex="col", sharey=True,num="preview"+str(os.getcwd())+dst )
        fig.set_figheight(12)
        fig.set_figwidth(12)
        for i in range( x.shape[0]):  

            ax[i%col,i//col].set_xlim([-2,2])
            ax[i%col,i//col].set_ylim([-2,2])
            xx=x[i,:].reshape(-1,2)
            ax[i%col,i//col].scatter(xx[:,0],xx[:,1],c=y[i,:],marker=".",cmap=cm.RdBu,vmin=-1,vmax=1)
            
        fig.set_figheight(12)
        fig.set_figwidth(12)
        
        plt.pause(0.00001)

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir +\
            '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        plt.savefig(preview_path)
        
    make_image.count=0
    return make_image
