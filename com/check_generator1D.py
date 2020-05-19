#!/usr/bin/env python

"""
Original code is 
https://github.com/chainer/chainer/blob/master/examples/dcgan/train_dcgan.py

"""
import argparse
import os
import warnings

import numpy

import chainer
from chainer import training
from chainer.training import extensions

from net import Generator
from updater_for_check_generator import DCGANUpdater
from visualize1D import out_generated_image
import chainer.functions as F

import chainer.links as L

def main():
    global d,l
    parser = argparse.ArgumentParser(description='Chainer example: DCGAN')
    parser.add_argument('--z_order', '-z', type=int, default=2,
                        help="")
    parser.add_argument('--batchsize', '-b', type=int, default=10000,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--device', '-d', type=str, default='1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device iniid are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='check1D',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str,
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval','-S', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    args.out+="_z"+str(args.z_order)
    if chainer.get_dtype() == numpy.float16:
        warnings.warn(
            'This example may cause NaN in FP16 mode.', RuntimeWarning)

    device = chainer.get_device(args.device)
    device.use()

    print('Device: {}'.format(device))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')




    ###########################################
    ###Make dummy dataset
    ##########################################
    import dummyData1D as dummyData
    nof_dummy_data=dummyData.NOF_DATA
    nof_pack=dummyData.NOF_PACK
    #dummyData.set_pattern(0)
    
    
    
    
    d,z=dummyData.makeDummy(nof_dummy_data,nof_pack)
    d=d.astype(chainer.get_dtype())
    z=z.astype(chainer.get_dtype())
    y=d[:,-nof_pack:]
    x=d[:,:-nof_pack]
    d=numpy.c_[x,z,y]
    print("xzy shape",x.shape,z.shape,y.shape)
    train=d
      ######################################
    
    
    
    # Set up a neural network to train
    gen = Generator(x_order=1,nof_pack=nof_pack, z_order=args.z_order ,wscale=0.5,out_max=1.5,out_min=-1.5)

    gen.to_device(device)  # Copy the model to the device

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        #optimizer.add_hook(
        #    chainer.optimizer_hooks.WeightDecay(0.0001), 'hook_dec')
        optimizer.add_hook(
            chainer.optimizer_hooks.Lasso(1e-5), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer(gen,alpha=0.0002)
    


    

    # Setup an iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    updater = DCGANUpdater(
        models=(gen,),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen},
        device=device)

    
    # Setup a trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss'
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_image(
            gen,nof_pack
            ,args.seed, args.out),
        trigger=snapshot_interval)

    if args.resume is not None:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
