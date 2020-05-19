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
import sys
sys.path.append("../com/")

import util

import chainer
from chainer import training
from chainer.training import extensions

from net import Discriminator
from net import Generator

from updater import DCGANUpdater
from visualize import out_generated_image
import dummyData


def main():
    args,train,device=util.get_input(dummyData)
    nof_pack=args.nof_pack


    
    
    # Set up a neural network to train
    gen = Generator(x_order=2,nof_pack=nof_pack, z_order=args.z_order ,wscale=0.5)
    dis = Discriminator(inten_sigma=0.01,nof_pack=nof_pack,x_order=2)

    gen.to_device(device)  # Copy the model to the device
    dis.to_device(device)

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
    opt_dis = make_optimizer(dis,alpha=0.0002)



    

    # Setup an iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Setup an updater
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
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
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',
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
        im=out_generated_image(
            gen,nof_pack
            ,args.seed, args.out+"zr0_5",zrate=0.5)
        im(trainer)

    else:
        trainer.run()


if __name__ == '__main__':
    main()
