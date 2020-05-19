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
import chainer.links as L

class RevG(chainer.Chain):

    def __init__(self,nof_pack,x_order,z_order,inten_sigma=0.01,
                 hidden_order=20,hidden_order2=20,hidden_last_order=10,
                 nof_hidden=4,nof_hidden2=4):
        self.hidden_order=hidden_order
        self.hidden_order2=hidden_order2
        self.hidden_last_order=hidden_last_order
        self.sigma=inten_sigma
        self.nof_pack=nof_pack
        self.nof_hidden=nof_hidden
        self.nof_hidden2=nof_hidden2
        self.x_order=x_order
        super(RevG, self).__init__()
        with self.init_scope():
            self.fc = chainer.ChainList()
            self.fc2 = chainer.ChainList()
            for i in range(self.nof_hidden):
                self.fc.add_link(  L.Linear(None,self.hidden_order ) )
            for i in range(self.nof_hidden2):
                self.fc2.add_link(  L.Linear(None,self.hidden_order2) )
            self.ll = L.Linear(None,self.hidden_last_order )
            self.ll2 = L.Linear(None,z_order )

    def forward_one(self,h):
        for i,f in enumerate(self.fc):
            h=f(h)
            h=F.leaky_relu(h)
        h=self.ll(h)
        return h
    
    def forward(self, x):
        batchsize=x.shape[0]
        
        yy=x[:,self.x_order*self.nof_pack:].reshape(-1,1)
        xx=x[:,:self.x_order*self.nof_pack].reshape(-1,self.x_order)
        h=F.concat([xx,yy])
        h=self.forward_one(h)
        #h1=F.reshape(h,(batchsize,self.nof_pack*self.hidden_last_order))
        h=F.sum(F.reshape(h,(batchsize,self.nof_pack,self.hidden_last_order)),axis=1)
        #h=F.concat((h1,h2))
            
        for i,f in enumerate(self.fc2):
            h=f(h)
            h=F.elu(h)
        return self.ll2(h)
