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
import numpy

import chainer
import chainer.functions as F
import chainer.links as L
import chainerx

CHECK_INPUT=0

def add_noise(device, h, sigma=0.2):
    if chainer.config.train:
        xp = device.xp
        if device.xp is chainerx:
            fallback_device = device.fallback_device
            with chainer.using_device(fallback_device):
                randn = device.send(fallback_device.xp.random.randn(*h.shape))
        else:
            randn = xp.random.randn(*h.shape)
        return h + sigma * randn
    else:
        return h



class Generator(chainer.Chain):

    def __init__(self,x_order=1,nof_pack=10, z_order=3 ,wscale=1,
                 nof_hidden=7,hidden_order=10,
                 out_max=1.5,out_min=-0.5):
        super(Generator, self).__init__()
        self.x_order=x_order
        self.nof_pack=nof_pack
        self.hidden_order=hidden_order
        self.hidden_order_r=x_order
        self.z_order=z_order
        self.nof_hidden=nof_hidden
        self.out_max=out_max
        self.out_min=out_min
        self.t=None
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.fc = chainer.ChainList()
            self.fcr = chainer.ChainList()
                
            for i in range(self.nof_hidden):
                self.fc.add_link(  L.Linear(None,self.hidden_order ,initialW=w) )
                self.fcr.add_link(  L.Linear(None,self.hidden_order_r ,initialW=w) )
            self.ll = L.Linear(None,1,initialW=w )       
   
         
 
    def make_hidden(self, batchsize):
        dtype = chainer.get_dtype()
        return (numpy.random.rand(batchsize, self.x_order*self.nof_pack).astype(dtype)*4-2,numpy.random.randn(batchsize, self.z_order).astype(dtype) )

    def forward_one(self,xx,zz):
        
        h=F.concat((xx,zz),axis=1)
            
        if CHECK_INPUT>0:
            print("Generator forward_one h",h.shape,h[:CHECK_INPUT,:])
        hh=[]
        for f,fr in zip(self.fc,self.fcr):
            #skipped connection to avoid  gradient disappearance
            hh.append(F.elu(fr(h)) *0.01) 
            h=f(h)
            h=F.elu(h)        
        hh.append(h)        
        h=F.concat(hh)
        del hh
        
        h=self.ll(h)
        #max,min limit for faster learn
        return (self.out_max-self.out_min)*(1+F.tanh(h))/2+self.out_min 
        
    def forward_for_show(self,x,z):
        batchsize=z.data.shape[0]        
        tmp_nof_pack=x.shape[1]//self.x_order
        zz=F.reshape(F.broadcast_to(F.reshape(z, (batchsize,1,self.z_order)),(batchsize,tmp_nof_pack,self.z_order)),(-1,self.z_order))
        xx=F.reshape(x,(-1,self.x_order))
        h=self.forward_one(xx,zz)
        h=F.reshape(h,(batchsize,tmp_nof_pack))
        return h
        
    
    
    def forward(self,x,z=None):
        #len of x must be nof_pack*x_order
        """
        input data order is like this
        x[0]z[0]
        x[..]z[0]
        x[nof_pack]z[0]
        
        x[0]z[1]
        ...
        x[nof_pack]z[1]
        """
        batchsize=z.shape[0]
        zz=F.reshape(F.broadcast_to(F.reshape(z, (batchsize,1,self.z_order)),(batchsize,self.nof_pack,self.z_order)),(-1,self.z_order))        
        xx=F.reshape(x,(-1,self.x_order))
        h=self.forward_one(xx,zz)
        h=F.reshape(h,(batchsize,self.nof_pack))
        return F.concat([x,h])        

      
class Discriminator(chainer.Chain):

    def __init__(self,nof_pack,x_order,inten_sigma=0.01,out_order=1,
                 hidden_order=20,hidden_order2=20,hidden_last_order=20,
                 nof_hidden=3,nof_hidden2=3):
        self.hidden_order=hidden_order
        self.hidden_order2=hidden_order2
        self.hidden_last_order=hidden_last_order
        self.sigma=inten_sigma
        self.nof_hidden=nof_hidden
        self.nof_hidden2=nof_hidden2
        self.nof_pack=nof_pack
        self.x_order=x_order
        self.out_order=out_order
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.fc = chainer.ChainList()
            self.fc2 = chainer.ChainList()
            for i in range(self.nof_hidden):
                self.fc.add_link(  L.Linear(None,self.hidden_order ) )
            for i in range(self.nof_hidden2):
                self.fc2.add_link(  L.Linear(None,self.hidden_order2) )
            self.ll = L.Linear(None,self.hidden_last_order )
            self.ll2 = L.Linear(None,self.out_order )

    def forward_one(self,h):
        if CHECK_INPUT>0:
            print("Discriminator forward_one h",h.shape,h[:CHECK_INPUT,:])

        for i,f in enumerate(self.fc):
            h=f(h)
            h=F.leaky_relu(h)
        h=self.ll(h)
        if CHECK_INPUT>0:
            print("Discriminator forward_one out-h",h.shape,h[:CHECK_INPUT,:])
        return h
    
    def forward(self, x):
        device = self.device
        #Add noise because...
        #Only Real data have noise and Generated data have no noise
        #So having noise is leak information for Discriminator.
        if(self.sigma>0):
            x = add_noise(device, x,sigma=self.sigma) 
        #print("D  shape",x.shape)
        batchsize=x.shape[0]
        
        yy=x[:,self.x_order*self.nof_pack:].reshape(-1,1)
        xx=x[:,:self.x_order*self.nof_pack].reshape(-1,self.x_order)
        h=F.concat([xx,yy])
        h=self.forward_one(h)
        #h1=F.reshape(h,(batchsize,self.nof_pack*self.hidden_last_order))
        h=F.sum(F.reshape(h,(batchsize,self.nof_pack,self.hidden_last_order)),axis=1)
        #h=F.concat((h1,h2))
        if CHECK_INPUT>0:
            print("Discriminator forward_one out-h reshaped",h.shape,h[:CHECK_INPUT,:])
            
        for i,f in enumerate(self.fc2):
            h=f(h)
            h=F.leaky_relu(h)
        return self.ll2(h)
