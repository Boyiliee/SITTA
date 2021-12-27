# Copyright (c) Boyi Li. All rights reserved.
from utils import *
import models
import random
import shutil
import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


def main():
    # opts: alternative training settings, config: related hyperparamters and network design
    opts = get_arguments()
    config = get_config(opts.config)
    config_g = config['netG']
    config_d = config['netD']
    max_epochs = config['max_epochs']
    display_size = config['display_size']

    # config settings
    opts.device = torch.device("cpu" if opts.not_cuda else "cuda:0")
    if opts.manualseed is None:
        opts.manualseed = random.randint(1, 10000)
    print("Random Seed: ", opts.manualseed)
    random.seed(opts.manualseed)
    torch.manual_seed(opts.manualseed)

    if opts.net_savedir:
        net_savedir = opts.net_savedir
    else:
        net_savedir = config['net_savedir']

    logroot = os.path.join(net_savedir, 'log')
    if not os.path.exists(logroot):
        os.makedirs(logroot)
    summary_writer = SummaryWriter(log_dir=logroot)

    # model initialization
    netG_A = models.AutoGenerator(config_g).to(opts.device)
    netG_B = models.AutoGenerator(config_g).to(opts.device)
    netD_A = models.Discriminator(config_d).to(opts.device)
    netD_B = models.Discriminator(config_d).to(opts.device)


    # setup optimizer
    netG_A_optimizer = optim.Adam(netG_A.parameters(), lr=config['lr_g'], betas=(config['beta1'], config['beta2']))
    netG_B_optimizer = optim.Adam(netG_B.parameters(), lr=config['lr_g'], betas=(config['beta1'], config['beta2']))
    netD_A_optimizer = optim.Adam(netD_A.parameters(), lr=config['lr_d'], betas=(config['beta1'], config['beta2']))
    netD_B_optimizer = optim.Adam(netD_B.parameters(), lr=config['lr_d'], betas=(config['beta1'], config['beta2']))

    print('Resume: {}, path: {}'.format(opts.resume, os.path.join(net_savedir, 'netG_Current.pt')))
    if opts.resume and os.path.isfile(opts.pretrained_path):
        print('Resume from {}'.format(opts.pretrained_path))
        netG_checkpoint = torch.load(opts.pretrained_path)
        netD_checkpoint = torch.load(os.path.join(net_savedir, 'netD_Current.pt'))
        optimizer_checkpoint = torch.load(os.path.join(net_savedir, 'optimizer.pt'))

        netG_A.load_state_dict(netG_checkpoint['netG_A'])
        netG_B.load_state_dict(netG_checkpoint['netG_B'])
        netD_A.load_state_dict(netD_checkpoint['netD_A'])
        netD_B.load_state_dict(netD_checkpoint['netD_B'])
        netG_A_optimizer.load_state_dict(optimizer_checkpoint['netG_A_optimizer'])
        netG_B_optimizer.load_state_dict(optimizer_checkpoint['netG_B_optimizer'])
        netD_A_optimizer.load_state_dict(optimizer_checkpoint['netD_A_optimizer'])
        netD_B_optimizer.load_state_dict(optimizer_checkpoint['netD_B_optimizer'])

        opts.start_epoch = netG_checkpoint['epoch'] + 1

        netG_A_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=netG_A_optimizer, step_size=config['step_size'],
                                                         gamma=config['gamma'], last_epoch=opts.start_epoch - 1)
        netG_B_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=netG_B_optimizer, step_size=config['step_size'],
                                                         gamma=config['gamma'], last_epoch=opts.start_epoch - 1)
        netD_A_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=netD_A_optimizer, step_size=config['step_size'],
                                                         gamma=config['gamma'], last_epoch=opts.start_epoch - 1)
        netD_B_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=netD_B_optimizer, step_size=config['step_size'],
                                                         gamma=config['gamma'], last_epoch=opts.start_epoch - 1)
    elif opts.resume and os.path.isfile(os.path.join(net_savedir, 'netG_Current.pt')):
        print('Resume from {}'.format(os.path.join(net_savedir, 'netG_Current.pt')))
        netG_checkpoint = torch.load(os.path.join(net_savedir, 'netG_Current.pt'))
        netD_checkpoint = torch.load(os.path.join(net_savedir, 'netD_Current.pt'))
        optimizer_checkpoint = torch.load(os.path.join(net_savedir, 'optimizer.pt'))

        netG_A.load_state_dict(netG_checkpoint['netG_A'])
        netG_B.load_state_dict(netG_checkpoint['netG_B'])
        netD_A.load_state_dict(netD_checkpoint['netD_A'])
        netD_B.load_state_dict(netD_checkpoint['netD_B'])
        netG_A_optimizer.load_state_dict(optimizer_checkpoint['netG_A_optimizer'])
        netG_B_optimizer.load_state_dict(optimizer_checkpoint['netG_B_optimizer'])
        netD_A_optimizer.load_state_dict(optimizer_checkpoint['netD_A_optimizer'])
        netD_B_optimizer.load_state_dict(optimizer_checkpoint['netD_B_optimizer'])

        opts.start_epoch = netG_checkpoint['epoch'] + 1

        netG_A_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=netG_A_optimizer, step_size=config['step_size'],
                                                         gamma=config['gamma'], last_epoch=opts.start_epoch - 1)
        netG_B_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=netG_B_optimizer, step_size=config['step_size'],
                                                         gamma=config['gamma'], last_epoch=opts.start_epoch - 1)
        netD_A_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=netD_A_optimizer, step_size=config['step_size'],
                                                         gamma=config['gamma'], last_epoch=opts.start_epoch - 1)
        netD_B_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=netD_B_optimizer, step_size=config['step_size'],
                                                         gamma=config['gamma'], last_epoch=opts.start_epoch - 1)
    else:
        netG_A.apply(models.weights_init(config['init_method']))
        netG_B.apply(models.weights_init(config['init_method']))
        netD_A.apply(models.weights_init(config['init_method']))
        netD_B.apply(models.weights_init(config['init_method']))

        netG_A_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=netG_A_optimizer, step_size=config['step_size'],
                                                         gamma=config['gamma'], last_epoch=opts.start_epoch - 1)
        netG_B_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=netG_B_optimizer, step_size=config['step_size'],
                                                           gamma=config['gamma'], last_epoch=opts.start_epoch - 1)
        netD_A_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=netD_A_optimizer, step_size=config['step_size'],
                                                         gamma=config['gamma'], last_epoch=opts.start_epoch - 1)
        netD_B_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=netD_B_optimizer, step_size=config['step_size'],
                                                         gamma=config['gamma'], last_epoch=opts.start_epoch - 1)

    
    # dataloader
    print('*** Loading Data ***')
    train_loader = get_singleM_dataloaders(config, config['trainA_dir'], shuffle=False)
    val_loader = get_singleM_dataloaders(config, config['testA_dir'], shuffle=False)
    style_loader = get_singleM_dataloaders(config, config['trainB_dir'], shuffle=False)
   
   # Load VGG model if needed
    if 'vgg_w' in config.keys() and config['vgg_w'] > 0:
        compute_vgg_loss = models.VGGLoss()

    # begin training
    print('*** Begin Training ***')
    saveroot = os.path.join(net_savedir, 'train')
    if not os.path.exists(saveroot):
        os.mkdir(saveroot)
    print('Results save to {}'.format(saveroot))

    imagesAs = []
    imagesBs = []
    for it, data in enumerate(zip(train_loader, style_loader)):
        imagesA, imagesB = data
        imagesA = imagesA.cuda()
        imagesB = imagesB.cuda()
        imagesAs.append(imagesA)
        imagesBs.append(imagesB)
    for epoch in range(opts.start_epoch, max_epochs + 1):
        # progress display
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses_netG_A = AverageMeter('loss_netG_A', ':.4e')
        losses_netD_A = AverageMeter('loss_netD_A', ':.4e')
        losses_netG_A_idt = AverageMeter('loss_netG_A_idt', ':.4e')
        losses_netG_A_cycle = AverageMeter('loss_netG_A_cycle', ':.4e')
        losses_netG_A_vgg = AverageMeter('loss_netG_A_vgg', ':.4e')
        losses_netG_A_texture = AverageMeter('loss_netG_A_texture', ':.4e')
        losses_netG_B = AverageMeter('loss_netG_B', ':.4e')
        losses_netD_B = AverageMeter('loss_netD_B', ':.4e')
        losses_netG_B_idt = AverageMeter('loss_netG_idt_B', ':.4e')
        losses_netG_B_cycle = AverageMeter('loss_netG_cycle_B', ':.4e')
        losses_netG_B_vgg = AverageMeter('loss_netG_B_vgg', ':.4e')
        losses_netG_B_texture = AverageMeter('loss_netG_B_texture', ':.4e')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses_netG_A, losses_netG_A_idt, losses_netG_A_cycle, losses_netG_A_vgg, losses_netG_A_texture, losses_netD_A, \
                                    losses_netG_B, losses_netG_B_idt, losses_netG_B_cycle, losses_netG_B_vgg, losses_netG_B_texture, losses_netD_B],
            prefix="Epoch: [{}]".format(epoch))

        for it in range(len(imagesAs)):
            imagesA = imagesAs[it]
            imagesB = imagesBs[it]
            end = time.time()
            """Calculate losses, gradients, and update network weights; called in every training iteration"""
            ### ***********************
            ### update D_A and D_B first
            ### ***********************
            netD_A.zero_grad()
            netD_B.zero_grad()
            # end = time.time()
            s_A, stats_A = netG_A.encode_s(imagesA)
            s_B, stats_B = netG_A.encode_s(imagesB)
            t_A = netG_A.encode_t(imagesA)
            t_B = netG_A.encode_t(imagesB)

            x_ab = netG_B.decode(s_A, t_B, stats_A)
            x_ba = netG_A.decode(s_B, t_A, stats_B)

            loss_netD_A_adv = netD_A.calc_dis_loss(x_ba.detach(), imagesA) 
            loss_netD_B_adv = netD_B.calc_dis_loss(x_ab.detach(), imagesB) # realB fakeB
                
            loss_netD = config['gan_w'] * loss_netD_A_adv + config['gan_w'] * loss_netD_B_adv
            loss_netD.backward()
            netD_A_optimizer.step()
            netD_B_optimizer.step()

            ### ***********************
            ### update G_A and G_B first
            ### ***********************
            netG_A.zero_grad()
            netG_B.zero_grad()

            s_ab, stats_ab = netG_A.encode_s(x_ab)
            s_ba, stats_ba = netG_A.encode_s(x_ba)
            
            x_aba = netG_A.decode(s_ab, t_A, stats_ab) 
            x_bab = netG_B.decode(s_ba, t_B, stats_ba)
            x_aa = netG_A.decode(s_A, t_A, stats_A) 
            x_bb = netG_B.decode(s_B, t_B, stats_B)
            
            loss_netG_A_adv = netD_A.calc_gen_loss(x_ba) 
            loss_netG_B_adv = netD_B.calc_gen_loss(x_ab)

            if 'cycle_w' in config.keys() and config['cycle_w'] > 0:
                loss_netG_A_cycle = models.compute_recon_loss(x_aba, imagesA)
                loss_netG_B_cycle = models.compute_recon_loss(x_bab, imagesB) 
            else:
                loss_netG_A_cycle = torch.tensor(0)
                loss_netG_B_cycle = torch.tensor(0)

            if 'idt_w' in config.keys() and config['idt_w'] > 0:
                loss_netG_A_idt = models.compute_recon_loss(x_aa, imagesA)
                loss_netG_B_idt = models.compute_recon_loss(x_bb, imagesB)
            else:
                loss_netG_A_idt = torch.tensor(0)
                loss_netG_B_idt = torch.tensor(0)

            if 'texture_w' in config.keys() and config['texture_w'] > 0:
                t_ab = netG_A.encode_t(x_ab)
                t_ba = netG_A.encode_t(x_ba)
                loss_netG_A_texture = -0.5 * (F.kl_div(t_A, t_ba) + F.kl_div(t_ba, t_A))
                loss_netG_B_texture = -0.5 * (F.kl_div(t_B, t_ab) + F.kl_div(t_ab, t_B))
            else:
                loss_netG_A_texture = torch.tensor(0)
                loss_netG_B_texture = torch.tensor(0)

            if 'vgg_w' in config.keys() and config['vgg_w'] > 0:
                loss_netG_A_vgg = compute_vgg_loss(x_ab, imagesA)
                loss_netG_B_vgg = compute_vgg_loss(x_ba, imagesB)
            else:
                loss_netG_A_vgg = torch.tensor(0)
                loss_netG_B_vgg = torch.tensor(0)


            loss_netG = config['gan_w'] * loss_netG_A_adv + config['gan_w'] * loss_netG_B_adv + \
                        config['cycle_w'] * loss_netG_A_cycle + config['cycle_w'] * loss_netG_B_cycle + \
                        config['idt_w'] * loss_netG_A_idt + config['idt_w'] * loss_netG_B_idt + \
                        config['texture_w'] * loss_netG_A_texture + config['texture_w'] * loss_netG_B_texture + \
                        config['vgg_w'] * loss_netG_A_vgg + config['vgg_w'] * loss_netG_B_vgg

            loss_netG.backward() 

            netG_A_optimizer.step()
            netG_B_optimizer.step()

            if config['lambda_grad'] > 0:
                loss_netD_grad_penalty = netD_A.calc_gradient_penalty(imagesA) * config['lambda_grad'] + netD_B.calc_gradient_penalty(imagesB) * config['lambda_grad']
                loss_netD_grad_penalty.backward() 

            losses_netG_A.update(loss_netG_A_adv.item(), imagesA[0].size(0))
            losses_netG_A_cycle.update(loss_netG_A_cycle.item(), imagesA[0].size(0))
            losses_netG_A_idt.update(loss_netG_A_idt.item(), imagesA[0].size(0))
            losses_netG_A_vgg.update(loss_netG_A_vgg.item(), imagesA[0].size(0))
            losses_netG_A_texture.update(loss_netG_A_texture.item(), imagesA[0].size(0))
            losses_netD_A.update(loss_netD.item(), imagesA[0].size(0))
            losses_netG_B.update(loss_netG_B_adv.item(), imagesA[0].size(0))
            losses_netG_B_cycle.update(loss_netG_B_cycle.item(), imagesA[0].size(0))
            losses_netG_B_idt.update(loss_netG_B_idt.item(), imagesA[0].size(0))
            losses_netG_B_vgg.update(loss_netG_B_vgg.item(), imagesA[0].size(0))
            losses_netG_B_texture.update(loss_netG_B_texture.item(), imagesA[0].size(0))
            losses_netD_B.update(loss_netD.item(), imagesA[0].size(0))

            # schedule
            netG_A_scheduler.step()
            netG_B_scheduler.step()
            netD_A_scheduler.step()
            netD_B_scheduler.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # prgress display
        if (epoch) % config['print_freq'] == 0:
            progress.display(it)
        if (epoch) % config['display_freq'] == 0:
            # write image
            s_A_mean = s_A.detach().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1).type(torch.cuda.FloatTensor)
            s_A_mean = torch.nn.functional.interpolate(s_A_mean, size=(imagesA.shape[2], imagesA.shape[3]), mode='bilinear',  align_corners=False)
            s_B_mean = s_B.detach().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1).type(torch.cuda.FloatTensor)
            s_B_mean = torch.nn.functional.interpolate(s_B_mean, size=(imagesA.shape[2], imagesA.shape[3]), mode='bilinear',  align_corners=False)
            display_images = imagesA, imagesB, s_A_mean, s_B_mean, x_ab, x_ba, x_aba, x_bab, x_aa, x_bb

            write_display(display_images, display_size, saveroot, epoch, it) 
            save_image(x_ab[0], '{}/o_ab_{}.png'.format(saveroot, epoch), normalize=True)
            save_image(x_ba[0], '{}/o_ba_{}.png'.format(saveroot, epoch), normalize=True)

        if summary_writer is not None:
            # tensorboard logger
            summary_writer.add_scalar('loss_netG_A', losses_netG_A.avg, epoch)
            summary_writer.add_scalar('loss_netG_A_cycle', losses_netG_A_cycle.avg, epoch)
            summary_writer.add_scalar('loss_netG_A_idt', losses_netG_A_idt.avg, epoch)
            summary_writer.add_scalar('loss_netG_A_vgg', losses_netG_A_vgg.avg, epoch)
            summary_writer.add_scalar('loss_netG_A_latent', losses_netG_A_texture.avg, epoch)
            summary_writer.add_scalar('loss_netD_A', losses_netD_A.avg, epoch)
            summary_writer.add_scalar('losses_netG_B', losses_netG_A.avg, epoch)
            summary_writer.add_scalar('losses_netG_B_cycle', losses_netG_A_cycle.avg, epoch)
            summary_writer.add_scalar('losses_netG_B_idt', losses_netG_A_idt.avg, epoch)
            summary_writer.add_scalar('loss_netG_B_vgg', losses_netG_B_vgg.avg, epoch)
            summary_writer.add_scalar('loss_netG_B_latent', losses_netG_B_texture.avg, epoch)
            summary_writer.add_scalar('losses_netD_B', losses_netD_A.avg, epoch)
            summary_writer.add_scalar('netG_A_learning_rate', netG_A_optimizer.param_groups[0]['lr'], epoch)
            summary_writer.add_scalar('netD_A_learning_rate', netD_A_optimizer.param_groups[0]['lr'], epoch)
        
        # save model for every specific epoch
        if epoch % config['keypoint_save_epoch'] == 0:
            save(netG_A, netG_B, netD_A, netD_B, netG_A_optimizer, netG_B_optimizer, netD_A_optimizer, netD_B_optimizer, net_savedir, epoch)

    # begin eval
    print('*** Begin Evaluation ***')
    saveroot = os.path.join(net_savedir, 'test')
    if not os.path.exists(saveroot):
        os.mkdir(saveroot)
    
    for it, val_data in enumerate(zip(val_loader, style_loader)):
        val_images, imagesB = val_data
        val_images = val_images.cuda().detach()
        imagesB = imagesB.cuda().detach()

        s_A, stats_A = netG_A.encode_s(val_images)
        t_B = netG_A.encode_t(imagesB)
        val_x_ab = netG_B.decode(s_A, t_B, stats_A).detach()
        display_images = val_images, val_x_ab
        write_display(display_images, display_size, saveroot, 0, it)
        print('save to {}'.format(saveroot))
            

def save(netG_A, netG_B, netD_A, netD_B, netG_A_optimizer, netG_B_optimizer, netD_A_optimizer, netD_B_optimizer, net_savedir, epoch, iter=''):
    # Save generators, discriminators, and optimizers
    if iter:
        netG_name = os.path.join(net_savedir, 'netG_%02d_iter%06d.pt' % (epoch, iter))
        netD_name = os.path.join(net_savedir, 'netD_%02d_iter%06d.pt' % (epoch, iter))
    else:
        netG_name = os.path.join(net_savedir, 'netG_%02d.pt' % (epoch))
        netD_name = os.path.join(net_savedir, 'netD_%02d.pt' % (epoch))

    opt_name = os.path.join(net_savedir, 'optimizer.pt')

    torch.save({'netG_A': netG_A.state_dict(), 'netG_B': netG_B.state_dict(), 'epoch': epoch}, netG_name)
    torch.save({'netD_A': netD_A.state_dict(), 'netD_B': netD_B.state_dict(), 'epoch': epoch}, netD_name)

    shutil.copyfile(netG_name, os.path.join(net_savedir, 'netG_Current.pt'))
    shutil.copyfile(netD_name, os.path.join(net_savedir, 'netD_Current.pt'))
    torch.save({'netG_A_optimizer': netG_A_optimizer.state_dict(), 'netG_B_optimizer': netG_B_optimizer.state_dict(), 'netD_A_optimizer': netD_A_optimizer.state_dict(), 'netD_B_optimizer': netD_B_optimizer.state_dict()}, opt_name)

    print('Save info: Epoch {}'.format(epoch))


if __name__ == '__main__':
    opts = get_arguments()
    print('Select Func {}'.format(opts.func))
    if opts.func == 'main':
        main()
    










