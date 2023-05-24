import numpy as np 
import torch 
import shutil
import torch.nn.functional as F
import torchvision
import math
import time
import datasets
# from r2score import r2_score
import math
from typing import Tuple
from torch import Tensor
from torchvision.transforms import functional as FT
import wandb
import random
import itertools

def train(train_loader, models_ensemble, criterion, optimizer, scaler, model_ema, epoch, args, coeff, kl_coeff):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    ce_losses = AverageMeter('CE Loss', ':.4e')
    variance = AverageMeter('Variance', ':.4e')
    kl = AverageMeter('KL Div', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, ce_losses, variance, kl, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    models_ensemble.train()
    end = time.time()
    avg_sim = 0.0
    acc1_list, acc5_list = np.zeros(args.ensemble_size), np.zeros(args.ensemble_size)
    kl_loss = torch.nn.KLDivLoss(reduction = "batchmean")

    for iter, data in enumerate(train_loader):
        images, labels = data[0], data[1]
        data_time.update(time.time() - end)
        learning_rates.update(optimizer.param_groups[0]['lr'])
        if args.gpu is not None:
            # Size of images is args.batch_size * (args.num_augs + 1)
            images = images.reshape(-1, 3, datasets.img_size, datasets.img_size).cuda(args.gpu, non_blocking=True) 
            labels = labels.cuda(args.gpu, non_blocking=True)

        images = get_aug_wise_images(images, args)

        with torch.cuda.amp.autocast(False):
            logits, feats = models_ensemble(images)
            # Last 'batch_size' number of images are unaugmented, there are (num_augs+1)*args.batch_size number of images in one batch
            orig_image_logits = logits[:, args.batch_size*args.num_augs:, :]

            # orig image logits and feats is of shape -  [args.ensemble_size+1, .....], where last element corresponds to output of baseline model which is frozen
            ce_loss = get_loss(criterion, orig_image_logits[:-1], labels)
            # Adding KL divergence loss between baseline and models_ensemble predictions
            kl_div = get_kl_loss(orig_image_logits[:-1], orig_image_logits[-1], kl_loss,  T=6, alpha=args.kl_coeff)
            similarity_matrix = get_similarity_vector(feats[:-1], args)
            diff = get_pairwise_rowdiff(similarity_matrix).sum()
            loss =  (1 - kl_coeff) * ce_loss + coeff * diff + kl_div

            avg_sim += similarity_matrix.cpu().detach()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if model_ema and iter % args.model_ema_steps == 0:
            model_ema.update_parameters(models_ensemble)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)
        

        acc1, acc5 = [], []
        for i in range(0, args.ensemble_size):
            a1, a5 = accuracy(orig_image_logits[i], labels, topk=(1, 5))
            acc1.append(a1.item())
            acc5.append(a5.item())
            acc1_list[i] += a1.item()
            acc5_list[i] += a5.item()
        acc1 = np.mean(acc1)
        acc5 = np.mean(acc5)

        wandb.log({"Mean acc": acc1, "Mean ce loss": ce_loss, "Mean diff": diff})
        ce_losses.update(ce_loss.item(), images.size(0)//(args.num_augs+1))
        variance.update(diff.item(), images.size(0)//(args.num_augs+1))
        kl.update(kl_div.item(), images.size(0)//(args.num_augs+1))
        top1.update(acc1, images.size(0)//(args.num_augs+1))
        top5.update(acc5, images.size(0)//(args.num_augs+1))
    
        batch_time.update(time.time() - end)
        end = time.time()
        # torch.cuda.empty_cache()
        if iter % args.print_freq == 0:
            progress.display(iter)

    acc1_list /= (iter)
    acc5_list /= (iter)
    avg_sim /= (iter)
    torch.cuda.empty_cache()
   
    return avg_sim, acc1_list, acc5_list, ce_losses.avg


def get_kl_loss(inputs, target, criterion, T, alpha):
    kl_div = 0.0
    for k in range(0, inputs.shape[0]):
        kl_div += criterion(F.log_softmax(inputs[k]/T, dim=1), F.softmax(target/T, dim=1)) * (alpha * T * T)
    return kl_div

def evaluate(val_loader, models_ensemble, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    ce_losses = AverageMeter('CE Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, learning_rates, ce_losses, top1, top5],
        prefix="Test: [{}]".format(epoch))

    models_ensemble.eval()
    end = time.time()

    acc1_list, acc5_list = np.zeros(args.ensemble_size), np.zeros(args.ensemble_size)
    orig_feats_list = []
    all_feats_list = [[] for _ in range(args.num_augs)]
    for iter, (images, labels) in enumerate(val_loader):
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.reshape(-1, 3, datasets.img_size, datasets.img_size).cuda(args.gpu, non_blocking=True) 
            labels = labels.cuda(args.gpu, non_blocking=True)
        
        images = get_aug_wise_images(images, args)
        with torch.no_grad():
            with torch.cuda.amp.autocast(False):
                logits, feats = models_ensemble(images) # Pass the un-augmented image

                # Last 'batch_size' images are unaugmented, there are (num_augs+1)*args.batch_size number of images in one batch
                orig_image_logits = logits[:, args.batch_size * args.num_augs:, :]
                orig_image_feats = feats[:, args.batch_size * args.num_augs:, :]
                orig_feats_list.append(orig_image_feats.cpu().detach())
                for n in range(args.num_augs):
                    all_feats_list[n].append(feats[:, n*args.batch_size:(n+1)*args.batch_size, :].cpu().detach())
                ce_loss = get_loss(criterion, orig_image_logits, labels)

            acc1, acc5 = [], []
            for i in range(0, args.ensemble_size):
                a1, a5 = accuracy(orig_image_logits[i], labels, topk=(1, 5))
                acc1.append(a1.item())
                acc5.append(a5.item())
                acc1_list[i] += a1.item()
                acc5_list[i] += a5.item()
            acc1 = np.mean(acc1)
            acc5 = np.mean(acc5)

            ce_losses.update(ce_loss.item(), images.size(0)//(args.num_augs+1))
            top1.update(acc1, images.size(0)//(args.num_augs+1))
            top5.update(acc5, images.size(0)//(args.num_augs+1))

        batch_time.update(time.time() - end)
        end = time.time()
        # torch.cuda.empty_cache()
        if iter % args.print_freq == 0:
            progress.display(iter)
            
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    acc1_list /= (iter+1)
    acc5_list /= (iter+1)
    all_orig_feats = torch.cat(orig_feats_list, dim = 1)
    sim = torch.zeros((args.ensemble_size + 1, args.num_augs))

    for i in range(0, args.num_augs):
        all_aug_feats = all_feats_list[i]
        all_aug_feats = torch.cat(all_aug_feats, dim=1)
        sim[:, i] = F.cosine_similarity(all_orig_feats, all_aug_feats, dim = 2).mean(1)

    wandb.log({"val acc": top1.avg, "ce loss": ce_loss})
    return acc1_list, acc5_list, top1.avg, sim

def adjust_coeff(coeff, epochs, start_coeff, end_coeff = 5.0):
    coeff += (end_coeff - start_coeff)/epochs
    return coeff

def get_aug_wise_images(images, args):
    # Change the order of the images
    images_list = []
    for i in range(args.num_augs+1):
        l = images[i::args.num_augs+1, :, :, :]
        images_list.append(l)
    images_list = torch.cat(images_list, dim = 0)
    return images_list

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar') 

def get_loss(criterion, logits, labels):
    total_loss = 0.0
    for i in range(logits.shape[0]):
        loss = criterion(logits[i], labels)
        total_loss += loss
    total_loss /= logits.shape[0]
    return total_loss

def get_pairwise_rowdiff(sim_matrix, criterion = torch.nn.L1Loss()):
    diff = 0.0
    for i in range(0, sim_matrix.shape[0]-1):
        for j in range(i+1, sim_matrix.shape[0]):
            diff += torch.exp(-criterion(sim_matrix[i], sim_matrix[j]))
    return diff

def get_similarity_vector(feats, args):
    # returns N vectors of R^k, each element being the similarity between original and augmented image
    sim = torch.zeros((args.ensemble_size, args.num_augs)).cuda(args.gpu)

    # Unaugmented images 
    orig_feats = feats[:, args.batch_size * args.num_augs:, :]

    for i in range(0, args.num_augs):
        aug_feats = feats[:, args.batch_size * i: args.batch_size * (i+1), :]
        sim[:, i] = F.cosine_similarity(orig_feats, aug_feats, dim = 2).mean(1)
    return sim

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def adjust_learning_rate(optimizer, epoch, args, prev_lr, linear_gamma = 0.16, decay_gamma = 0.975):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        # lr = prev_lr + linear_gamma
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = prev_lr * decay_gamma
        # lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)
