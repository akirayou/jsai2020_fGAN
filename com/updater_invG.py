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
import random
class DCGANUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis,self.revg = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)
        self.count=0

    def loss_wgan(self,dis,gen,y_fake,y_real,x_fake,x_real):
        
        loss_real=F.average(y_real)
        loss_fake=F.average(y_fake)
       
        loss = loss_real-loss_fake
        loss += 0.01*(F.average( (y_real-y_fake)**2) -1)**2
        
        chainer.report({'loss': loss_real}, dis)
        chainer.report({'loss': loss_fake}, gen)
        loss=loss_real+loss_fake
        return loss
        
    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake,x_rev,x_real):
        batchsize = len(y_fake)
        loss_rev= F.mean_squared_error(x_rev,x_real) 
        loss_gan= F.sum(F.softplus(-y_fake)) / batchsize
        loss = loss_rev + loss_gan
        chainer.report({'loss': loss.data,'loss_gan': loss_gan.data,'loss_rev': loss_rev.data}, gen)
        return loss
    def loss_revg(self,revg, z1,z2):
        loss= F.mean_squared_error(z1,z2)
        chainer.report({'loss': loss.data}, revg)
        return loss

    def update_core(self):
        self.count+=1
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        revg_optimizer = self.get_optimizer('revg')

        batch = self.get_iterator('main').next()
        batchsize=len(batch)
        device = self.device     
        x_real = Variable(self.converter(batch, device))
        del batch
        gen, dis,revg = self.gen, self.dis,self.revg
        
   
     

        x_hidden,z=gen.make_hidden(batchsize)
        
        z = Variable(device.xp.asarray(z))
        x_hidden = Variable(device.xp.asarray(x_hidden))

        xy_part=x_hidden.shape[1]
        xx= x_real[:,:xy_part]
        x_fake = gen(xx,z)
        
        
        #print("x_fake,x_rean",x_fake.data.shape,x_real.data.shape)
        y_fake = dis(x_fake)
        y_real = dis(x_real)
        
        
        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        del y_real        
        
        
        rev_z=revg(x_fake)
        revg_optimizer.update(self.loss_revg,revg,z,rev_z)
        del rev_z
        
        rev_z=revg(x_real)
        #rev_z.unchain()
        x_rev=gen(xx,rev_z)
        del rev_z

        gen_optimizer.update(self.loss_gen, gen, y_fake,x_rev[:,xy_part:],x_real[:,xy_part:])
        del y_fake,x_rev,x_real
        
