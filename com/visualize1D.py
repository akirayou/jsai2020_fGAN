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


def out_generated_image(gen,nof_pack_org , seed, dst,zrate=1):
    @chainer.training.make_extension()
    def make_image(trainer):
        make_image.count+=1
        if(make_image.count%10==9):plt.close("all")
        nof_pack=100
        np.random.seed(seed)
        xp = gen.xp
        xd,z=gen.make_hidden(40)
        x_order=xd.shape[1]//nof_pack_org
        if x_order!= 1 : return
        
        x_np=np.linspace(-3,3,nof_pack)
        
        x_np=np.ones(xd.shape[0]).reshape(-1,1)*x_np.reshape(1,-1)
        x_np=x_np.astype(chainer.get_dtype())
        z*=zrate
        x = Variable(xp.asarray(x_np))
        z = Variable(xp.asarray(z))
        with chainer.using_config('train', False):
            y = gen.forward_for_show(x,z)
        y = chainer.backends.cuda.to_cpu(y.array)
        np.random.seed()
        x=x_np
        """
        plt.figure("preview"+str(os.getcwd())+dst)
        
        plt.clf()
        for i in range( x.shape[0]):
            idx=np.argsort(x[i,:])
            plt.plot(x[i,idx],y[i,idx])
        
        plt.pause(0.00001)
        """
        fig, axs = plt.subplots(1, 2,  sharey=True,num="preview"+str(os.getcwd())+dst)
        fig.set_figheight(6)
        fig.set_figwidth(12)
        
        ax=axs[0]
        ax.set_title("generated")
        ax.set_xlim([-3,3])
        ax.set_ylim([-1.5,1.5])
        for i in range( x.shape[0]):
            idx=np.argsort(x[i,:])
            ax.plot(x[i,idx],y[i,idx])
        dy=y[:,1:]-y[:,:-1]
        margin=y.shape[1]//5
        for i in range(y.shape[0]):
            x[i,:]-= x[i,margin+np.argmax(dy[i,margin:-margin])]
        ax=axs[1]
        ax.set_title("aligined")
        ax.set_xlim([-3,3])
        ax.set_ylim([-1.5,1.5])
        for i in range( x.shape[0]):
            idx=np.argsort(x[i,:])
            ax.plot(x[i,idx],y[i,idx])
        
        plt.pause(0.00001)
        fig.set_figheight(6)
        fig.set_figwidth(12)
        plt.pause(0.00001)

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir +\
            '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        plt.savefig(preview_path)
    














        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir +\
            '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        plt.savefig(preview_path)
    
    make_image.count=0
    return make_image
