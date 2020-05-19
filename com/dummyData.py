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
NOF_PACK=3*3
NOF_DATA=100000
DUMMY_PAT=4
x_order=2
def set_pattern(x):
    global DUMMY_PAT
    DUMMY_PAT=x
    
def gauss(x):
    print("gauss",x.shape)
    return np.exp(- np.sum(x**2,axis=0) )
def cau(x):
    return  1/ (1+np.sum( x**2,axis=0)) 
def makeDummy(t_len=300,nof_pack=10,x_in=None):
    if x_in is None : x=np.random.rand(t_len,nof_pack*x_order)*4-2
    else :
        x=np.array(np.meshgrid(x_in,x_in)).transpose( (1,2,0)).reshape(-1,2)
        x=x.reshape(1,-1)*np.ones((t_len,1))
        print("made x",x.shape)
    if(DUMMY_PAT==4):
        z=np.random.randn(t_len,4)
        z2=z[:,2].reshape(-1,1)
        z3=z[:,3].reshape(-1,1)

    else:
        z=np.random.randn(t_len,2)
    
    
    x0=x[:,0::x_order]
    x1=x[:,1::x_order]
    z0=z[:,0].reshape(-1,1)
    z1=z[:,1].reshape(-1,1)
  
    if DUMMY_PAT==1:
        posx=z0
        posy=z1
        
        y=cau( np.array( np.array( (  ( x0+posx)/2, (x1-posy)/1.3))  ))
    if DUMMY_PAT==2:
        posx=np.sin(3*z0)*1.5
        posy=np.cos(3*z0)*1.5
        
        y1=cau( np.array( np.array( (x0-posx,x1-posy))*1  ))
        posx=np.sin(3+3*z1)
        posy=np.cos(3+3*z1)
        y2=cau( np.array( np.array( (x0-posx,x1-posy))*0.7 ))
        y=np.maximum(y1,y2)
    if DUMMY_PAT==3:
        y=np.sin(x0*1.5+z0*3 )
        y+=np.sin(x1*1.5+z1*3  )
        y=y/4+0.5


    if DUMMY_PAT==4:
        posx=z0
        posy=z1
        rot=z2*2
        scale=np.exp(z3/3)
        
        s=np.sin(rot)
        c=np.cos(rot)
        
        ppx= s*x0+c*x1
        ppy= -s*x1+c*x0
        
        px=(ppx+posx)/2
        py=(ppy+posy)/scale
        
        
        y=cau( np.array(  (px ,py )  ))
    
    if DUMMY_PAT==5:
        posx=z0
        posy=z1
        rot=z0*1
        rot=0
        s=np.sin(rot)
        c=np.cos(rot)
        
        ppy= s*x0+c*x1
        ppx= c*x0-s*x1
        
        px=(ppx+posx)/2
        py=(ppy+posy)/1.3
        
        
        y=cau( np.array(  (px ,py )  ))
    print(y.shape)
    return np.c_[x,y],z  

    
    
if __name__ == "__main__":    
    set_pattern(4)
    import matplotlib.pyplot as plt
    plt.close("all")
    row=7
    col=7
    t_len=row*col
    nof_pack=NOF_PACK
    a,z=makeDummy(t_len,nof_pack)
    print(a.shape)
    x=a[:,:nof_pack*x_order]
    y=a[:,nof_pack*x_order:]
    print(x.shape,y.shape)
    
    fig, ax = plt.subplots(row, col, sharex="col", sharey=True,num="sample")
    fig.set_figheight(12)
    fig.set_figwidth(12)
    for i in range( t_len):  
        ax[i%col,i//col].set_xlim([-2,2])
        ax[i%col,i//col].set_ylim([-2,2])
        xx=x[i,:].reshape(-1,2)
        ax[i%col,i//col].scatter(xx[:,0],xx[:,1],c=y[i,:],marker=".",cmap="Blues",vmin=0,vmax=1)


    nof_pack=40
    a,z=makeDummy(t_len,nof_pack,x_in=np.linspace(-3,3,nof_pack))
    nof_pack=nof_pack**2
    print(a.shape)
    x=a[:,:nof_pack*x_order]
    y=a[:,nof_pack*x_order:]

    fig, ax = plt.subplots(row, col, sharex="col", sharey=True,num="true value")
    fig.set_figheight(12)
    fig.set_figwidth(12)
    for i in range( t_len):  
        ax[i%col,i//col].set_xlim([-2,2])
        ax[i%col,i//col].set_ylim([-2,2])
        xx=x[i,:].reshape(-1,2)
        ax[i%col,i//col].scatter(xx[:,0],xx[:,1],c=y[i,:],marker=".",cmap="Blues",vmin=0,vmax=1)
