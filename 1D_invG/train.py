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
from rev_net import RevG


from updater_invG import DCGANUpdater
from visualize1D import out_generated_image
import dummyData1D as dummyData

def main():
    args,train,device=util.get_input(dummyData)
    nof_pack=args.nof_pack

    # Set up a neural network to train
    gen = Generator(x_order=1,nof_pack=nof_pack, z_order=args.z_order ,wscale=0.5,out_max=1.5,out_min=-1.5)
    dis = Discriminator(nof_pack=nof_pack,x_order=1,inten_sigma=0.01,
                hidden_order=10,nof_hidden=4,hidden_last_order=10,
                hidden_order2=10,nof_hidden2=4
            )
    revg = RevG(inten_sigma=0.01,nof_pack=nof_pack,x_order=1,z_order=args.z_order,
                hidden_order=10,nof_hidden=4,hidden_last_order=10,
                hidden_order2=5,nof_hidden2=4
                )
    revg.to_device(device)


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
    opt_revg = make_optimizer(revg,alpha=0.0002)


    

    # Setup an iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Setup an updater
    updater = DCGANUpdater(
        models=(gen, dis,revg),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis,'revg':opt_revg},
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
    trainer.extend(extensions.snapshot_object(
        revg, 'revg_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
  
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration',
        'gen/loss',
        'gen/loss_rev',
        'gen/loss_gan',
        'dis/loss',
        'revg/loss'
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
        # Run the training
        trainer.run()


if __name__ == '__main__':
    main()
