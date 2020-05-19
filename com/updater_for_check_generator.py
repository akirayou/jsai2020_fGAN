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
import chainer
import chainer.functions as F
from chainer import Variable
import numpy

class DCGANUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)
        self.count=0
        self.l=10
        self.nc=5
    def update_core(self):
        self.count+=1
        gen_optimizer = self.get_optimizer('gen')
        device = self.device
        gen = self.gen
        xp=gen.device.xp
        


        batch = self.get_iterator('main').next()
        batchsize=len(batch)

        xzy = Variable(self.converter(batch, device))   
        x_len=gen.x_order*gen.nof_pack
        y_len=gen.nof_pack
        z_len=gen.z_order
        xz=xzy[:,:x_len+z_len]
        x=xz[:,:x_len]
        z=xz[:,x_len:]
        y=xzy[:,x_len+z_len:]
        if False:
            def stat(x,name):
                print(name+" stat  min mean max std", float(xp.min(x)),float(xp.mean(x)),float(xp.max(x)),float(xp.std(x)))
            stat(x.data,"x")
            stat(y.data,"y")
            stat(z.data,"z")
        
        ey=gen(x,z)
        ey=ey[:,x_len:]
        loss_gen=F.mean_squared_error(ey,y)

        # update generator
        gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()
        chainer.report({'loss': loss_gen}, gen)
        gen_optimizer.update()
