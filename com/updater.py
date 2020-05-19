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
        self.gen, self.dis = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)
        self.count=0

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss
 
    def update_core(self):
        self.count+=1
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        batchsize=len(batch)
        device = self.device
        x_real = Variable(self.converter(batch, device))

        gen, dis = self.gen, self.dis
        
        
        y_real = dis(x_real)

        x_hidden,z=gen.make_hidden(batchsize)
        
        z = Variable(device.xp.asarray(z))
        x_hidden = Variable(device.xp.asarray(x_hidden))
        if(self.count%5!=0):
            xx= x_real[:,:x_hidden.shape[1]]
        else:
            xx=x_hidden
        x_fake = gen(xx,z)
        
        #print("x_fake,x_rean",x_fake.data.shape,x_real.data.shape)
        y_fake = dis(x_fake)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)
