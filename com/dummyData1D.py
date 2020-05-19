#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyrihgt Shimadzu Corp.
This demonstration code can only be used for research purposes 
(Includes academic education). Commercial use is prohibited. 
Distribution of derivative software  for your research is permitted 
if the distribution is limited to research use. 
Other redistribution is prohibited.
Contact: a-noda@shimadzu.co.jp  

Created on Fri Jul 12 10:40:01 2019

@author: La-noda
"""
import numpy as np

NOF_PACK=4
NOF_DATA=10000
DUMMY_PAT=1
x_order=1
def set_pattern(x):
    global DUMMY_PAT
    DUMMY_PAT=x
    
    
def makeDummy(t_len=300,nof_pack=10,x_in=None):
    if x_in is None : x=np.random.randn(t_len,nof_pack)
    else : x=x_in.reshape(1,-1)*np.ones((t_len,1))
    
    
    if DUMMY_PAT==0:
        z=np.random.randn(t_len,1)
        p= 0.5*z[:,0].reshape(-1,1)
        x_org=x
        x=x+p
        s=1
        y=np.sin(2*x)
        
        
    
    if DUMMY_PAT==1:
        z=np.random.randn(t_len,2)
        
        p= 0.5*z[:,1].reshape(-1,1)
        r=1-np.exp(-0.75+0.75*np.tanh(z[:,0].reshape(-1,1)))
        print(r)
        
        x_org=x
        x=x+p
        s=1
        y=np.sin(2*x)
        
        for i in range(200):
            y+=r**(i+1)*np.sin( 2*(i+1)*x )
            s+= r**(i+1)
            
        y/=2

    return np.c_[x_org,y],z

    
    
if __name__ == "__main__":    
    import matplotlib.pyplot as plt
    nof_pack=NOF_PACK
    a,z=makeDummy(14,nof_pack)
    print(a.shape)
    x=a[:,:nof_pack]
    y=a[:,nof_pack:]
    plt.figure("sparse")    
    plt.clf()
    plt.xlim([-3,3])
    for i in range( x.shape[0]):
        idx=np.argsort(x[i,:])
        plt.plot(x[i,idx],y[i,idx],"+-",lw=1)

    nof_pack=100
    sx=x
    sy=y
    
    fig, axs = plt.subplots(1, 2,  sharey=True,num="1D_data")

    
    a,z=makeDummy(40,nof_pack,x_in=np.linspace(-3,3,nof_pack))
    print(a.shape)
    x=a[:,:nof_pack]
    y=a[:,nof_pack:]
    ax=axs[0]
    ax.set_title("Mesured")
    for i in range( sx.shape[0]):
        idx=np.argsort(sx[i,:])
        ax.plot(sx[i,idx],sy[i,idx],"-+")
    ax.set_xlim([-3,3])
    ax.set_ylim([-1.2,1.2])
 

    ax=axs[1]
    ax.set_title("True")
    ax.set_xlim([-3,3])
    ax.set_ylim([-1.2,1.2])
    for i in range( x.shape[0]):
        idx=np.argsort(x[i,:])
        ax.plot(x[i,idx],y[i,idx])
    dy=y[:,1:]-y[:,:-1]
    margin=y.shape[1]//5
    for i in range(y.shape[0]):
        x[i,:]-= x[i,margin+np.argmax(dy[i,margin:-margin])]
     
    fig.set_figheight(6)
    fig.set_figwidth(12)


    fig, axs = plt.subplots(1, 1,  sharey=True,num="1D_data_aligined")

    ax=axs
    ax.set_title("Aligined")
    ax.set_xlim([-3,3])
    ax.set_ylim([-1.2,1.2])
    for i in range( x.shape[0]):
        idx=np.argsort(x[i,:])
        ax.plot(x[i,idx],y[i,idx])
    
    fig.set_figheight(6)
    fig.set_figwidth(6)

    plt.pause(0.001)