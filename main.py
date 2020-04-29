import argparse
import time
import copy
import tqdm
import os
import re
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.optim import Adam
from utils.models import DarkDataset, UNet

import warnings
warnings.filterwarnings('ignore')

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for Seeing In The Dark")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=4000,
                                  help="number of training epochs, default is 4000")
    train_arg_parser.add_argument("--batch_size", type=int, default=1,
                                  help="batch size for training, default is 1")
    train_arg_parser.add_argument("--train_txt_file", type=str, default= './Sony/Sony_train_list.txt',
                                  help="path to the txt file corresponding to the training data, default is ./Sony/Sony_train_list.txt")
    train_arg_parser.add_argument("--patch_size", type=int, default=512,
                                  help="patch size used while training, default is 512")
    train_arg_parser.add_argument("--lr", type=float, default=1e-4,
                                  help="learning rate, default is 1e-4")
    train_arg_parser.add_argument("--cuda", type=int, default = 1,
                                  help="set it to 1 for running on GPU, 0 for CPU, default is 1")    
    train_arg_parser.add_argument("--log_interval", type=int, default=100,
                                  help="number of images after which the training loss is logged, default is 100")
    train_arg_parser.add_argument("--save_model_dir", type=str, default= './checkpoint_result/',
                                   help="path to folder where trained model will be saved, default is ./checkpoint_result/")
    train_arg_parser.add_argument("--checkpoint_model_dir", type=str, default='./checkpoints/',
                                  help="path to folder where checkpoints of trained models will be saved, default is ./checkpoints/")
    
    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--dark_image", type=str, required=True,
                                 help="path of the raw image (ARW) you want to evaluate")
    eval_arg_parser.add_argument("--output_image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model_path", type=str, required=True,
                                  help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, default=1,
                                 help="set it to 1 for running on GPU, 0 for CPU, default is 1")

    args = main_arg_parser.parse_args()
    
    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        evaluate(args)

def train(args):
    device = torch.device("cuda:0" if args.cuda  else 'cpu')
    
    trainset = DarkDataset(args.train_txt_file, ps = args.patch_size)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle = True, batch_size = 1)
    
    # Loading the UNet based Network model
    network = UNet().to(device)
    optimizer = Adam(network.parameters(), lr= args.lr)
    L1_loss = nn.L1Loss()

    # Loading checkpoint data (if available)
    if os.path.isfile(os.path.join(args.checkpoint_model_dir, 'checkpoint.pth.tar')):
        
        checkpoint = torch.load(os.path.join(args.checkpoint_model_dir, 'checkpoint.pth.tar'))
        start_epoch = checkpoint['epoch']
        network.load_state_dict(checkpoint['network_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {})"
                        .format(os.path.join(args.checkpoint_model_dir, 'checkpoint.pth.tar'), checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(os.path.join(args.checkpoint_model_dir, 'checkpoint.pth.tar')))
        start_epoch = 0

    #Start training
    for e in range(start_epoch, args.epochs):
        count = 0
        if e == 2000:
            optimizer.param_groups[0]['lr'] = 0.00001
  
        for batch_id, (input_patch, target_patch) in tqdm.tqdm(enumerate(trainloader)):

            count+= 1

            input_patch = input_patch.to(device)
            target_patch = target_patch.to(device)

            output = network(input_patch)
            loss = L1_loss(output, target_patch)
            loss.backward()
            optimizer.step()

            # Logging the losses
            if (batch_id + 1) % args.log_interval == 0:
                mesg = "\tEpoch {}:  [{}/{}]\tL1_loss: {:.6f}".format(
                    e + 1, count, len(trainset), loss.item())
                print(mesg)
        
        # Checkpointing models
        if e + 1 % args.checkpoint_interval == 0:
            network.eval().cpu()
            ckpt_model_path = os.path.join('checkpoints', 'checkpoint.pth.tar')
            torch.save({'epoch': e + 1, 'network_state_dict': network.state_dict(),
                        'optimizer' : optimizer.state_dict()}, ckpt_model_path)
            network.to(device).train()
    
    # save model
    network.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(args.patch_size) + ".pth.tar"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(network.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)

def produce(args):
    device = torch.device("cuda:0" if args.cuda  else 'cpu')
    network = UNet().to(device)
    network.load_state_dict(args.model_path)
    network.eval()
    
    input_ = torch.tensor(pack_raw(args.dark_image)).to(args.device)
    input_ = input_.permute(2, 0, 1).unsqueeze(0)
    H, W = input_.shape[2:]

    output = network(input_)
    output = (np.clip(np.transpose(output.detach().squeeze(0).cpu().numpy(), (1,2,0)), 0., 1.) * 255).astype(np.uint8)
    img = Image.fromarray(output).resize((W,H), Image.NEAREST)
    img.save(args.output_image)

def check_paths(args):

    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def pack_raw(filename):
    # pack Bayer image to 4 channels
    raw = rawpy.imread(filename)
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

if __name__ == "__main__":
    main()

