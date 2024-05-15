import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from prefetch_dataloader import DataLoaderX
import time
import math
import itertools
import utils
import model
import dataset
import loss
import numpy as np
import argparse
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--inDir", "-i", required=True)
parser.add_argument("--outDir", "-o", required=True)
parser.add_argument("--valInDir", "-v", required=True)
parser.add_argument("--avgDir", "-a", required=True)
parser.add_argument("--inlDir", type=str, required=True)
parser.add_argument("--valInlDir", type=str, required=True)
# training configs
parser.add_argument("--iterations", type=int, default=400001)
parser.add_argument("--batchSize", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--lrStep", type=int, default=100000)
parser.add_argument("--numSavedImgs", type=int, default=10, help="# of images to be saved during model saving")
parser.add_argument("--seed", type=int, default=990819)
parser.add_argument("--displayFreq", type=int, default=1, help="display losses (pbar)")
parser.add_argument("--summaryFreq", type=int, default=100, help="save losses")
parser.add_argument("--saveFreq", type=int, default=10000, help="save model every saveFreq steps, 0 to disable")
parser.add_argument("--valFreq", type=int, default=10000, help="test model every testFreq steps, 0 to disable")

parser.add_argument("--svbrdfMapLossWeight", type=float, default=0.1)
parser.add_argument("--renderLossWeight", type=float, default=1)
parser.add_argument("--svbrdfMapSwapLossWeight", type=float, default=0.1)
parser.add_argument("--renderSwapLossWeight", type=float, default=1)
parser.add_argument("--svbrdfMapUnsupLossWeight", type=float, default=0.1)
parser.add_argument("--numDiffRendering", type=int, default=3)
parser.add_argument("--numSpecRendering", type=int, default=6)

parser.add_argument("--isLoad", dest="loadModel", action="store_true")
parser.set_defaults(loadModel=False)
parser.add_argument("--modelPath", type=str, default=None)
# model or dataset parameters
parser.add_argument("--inCh", type=int, default=3)
parser.add_argument("--baseCh", type=int, default=64)
parser.add_argument("--latentCh", type=int, default=128)
parser.add_argument("--latentChN", type=int, default=32, help="end location")
parser.add_argument("--latentChD", type=int, default=64, help="end location")
parser.add_argument("--latentChR", type=int, default=96, help="end location")
parser.add_argument("--latentChS", type=int, default=128, help="end location")
parser.add_argument("--outCh", type=int, default=10)
parser.add_argument("--normType", type=str, default='batch')
parser.add_argument("--actType", type=str, default='relu')
parser.add_argument("--upType", type=str, default='bilinear')
parser.add_argument("--numHeads", type=int, default=8)
parser.add_argument("--attnDrop", type=int, default=0)
parser.add_argument("--projDrop", type=int, default=0)
parser.add_argument("--drop", type=int, default=0)
parser.add_argument("--groups", type=int, default=4)
parser.add_argument("--chReduct", type=int, default=1)
parser.add_argument("--bottleChReduct", type=int, default=2)
parser.add_argument("--jsonPath", type=str, default='group.json')
parser.add_argument("--numRenderings", type=int, default=1)
parser.add_argument("--mode", type=str, default='render')
parser.add_argument("--valNumRenderCount", type=int, default=1)
parser.add_argument("--renderSize", type=int, default=256)
parser.add_argument('--angleDelta', type=int, default=10)
parser.add_argument('--randScale', nargs='+', type=float, default=[0.7, 1])
parser.add_argument("--savePath", "-s", default=None)
parser.add_argument("--noFirstPosAsGuide", dest="noFirstPosAsGuide", action="store_true")
parser.set_defaults(noFirstPosAsGuide=False)
parser.add_argument("--noLogInput", dest="noLogInput", action="store_true")
parser.set_defaults(noLogInput=False)
parser.add_argument("--noLogOutputAlbedo", dest="noLogOutputAlbedo", action="store_true")
parser.set_defaults(noLogOutputAlbedo=False)
args, unknown = parser.parse_known_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
permutation = [0, 3, 1, 2]
depermutation = [0, 2, 3, 1]


def train(args, train_dataloader, val_dataloader, num_val_samples, root_save_path):
    in_unet = model.GatedDeformUNet(3, 3).to(device)
    print('in_unet')
    model.print_model_structure(in_unet)

    all_loss = loss.AllLoss(6, device, Lamda=10, lr=args.lr, betas=(0.0, 0.9))
    optimizer = optim.Adam(in_unet.parameters(), lr=args.lr, betas=(0.5, 0.9))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5, last_epoch=-1)

    print("\t-Start training")
    in_unet.eval()
    save_img_interval = num_val_samples // args.numSavedImgs
    total_iteration = 0
    while True:
        for _, batch_sample in enumerate(train_dataloader):
            iter_start = time.time()
            inputs, in_gt = batch_sample
            inputs = inputs.permute(permutation).to(device).contiguous()
            in_gt = in_gt.permute(permutation).to(device).contiguous()

            in_out = in_unet(inputs)

            if (total_iteration + 1) % 5 == 0:
                loss_all, G_loss_data, l1_loss_data, P_loss_data, D_loss_data = all_loss(inputs.detach(), in_out, in_gt, train_G=True)
                optimizer.zero_grad()
                loss_all.backward()
                optimizer.step()
            else:
                loss_all, G_loss_data, l1_loss_data, P_loss_data, D_loss_data = all_loss(inputs.detach(), in_out, in_gt, train_G=False)

            iter_took = time.time() - iter_start
            total_iteration += 1
            lr = optimizer.param_groups[0]['lr']

            if total_iteration % args.displayFreq == 0:
                print(
                    '[{}] iter_took:{:.3f}s  lr:{:.6f}  l1:{:.3f}  p:{:.3f}  G:{:.3f}  D:{:.3f}'.format(
                        total_iteration, iter_took, lr, l1_loss_data, P_loss_data, G_loss_data, D_loss_data))

            if total_iteration % args.summaryFreq == 0:
                f = open(os.path.join(root_save_path, 'loss.txt'), 'a')
                f.writelines(['\n', '[iteration: {}  lr:{:.8f}]\n'.format(total_iteration, lr),
                              'l1: {:.3f}\n'.format(l1_loss_data),
                              'p: {:.3f}\n'.format(P_loss_data),
                              'G: {:.3f}\n'.format(G_loss_data),
                              'D: {:.3f}\n'.format(D_loss_data)])
                f.close()

            if total_iteration % args.saveFreq == 0:
                current_save_path = utils.create_folder(os.path.join(root_save_path, 'iter%d' % total_iteration))
                torch.save(in_unet.state_dict(), os.path.join(current_save_path, "in_unet.pt"))

            if total_iteration % args.valFreq == 0:
                in_unet.eval()
                file_paths = []
                with torch.no_grad():
                    avg_rmse_in = 0
                    for i_batch, batch_sample in enumerate(val_dataloader):
                        inputs, in_gt, materials, name = batch_sample
                        inputs = inputs.permute(permutation).to(device)

                        in_out = in_unet(inputs)

                        inputs = inputs.permute(depermutation)
                        in_out = in_out.permute(depermutation)

                        if i_batch % save_img_interval == 0 and i_batch != 0:
                            current_save_path = utils.create_folder(os.path.join(root_save_path, 'iter%d' % total_iteration))
                            files = utils.save_maps(materials.cpu().numpy()[0],
                                                    utils.deprocess(inputs.cpu()).numpy()[0],
                                                    utils.deprocess(in_gt.cpu()).numpy()[0],
                                                    utils.deprocess(in_out.cpu()).numpy()[0],
                                                    args.noLogOutputAlbedo, current_save_path, i_batch)
                            files.append(name)
                            file_paths.append(files)
                        avg_rmse_in += utils.calculate_root_mse(utils.deprocess(in_out).cpu().numpy(),
                                                                utils.deprocess(in_gt).cpu().numpy())
                avg_rmse_in /= num_val_samples

                print('[Val {}] in_rmse:{:.5f}'.format(total_iteration, avg_rmse_in))
                # save validation results
                f = open(os.path.join(root_save_path, 'validation.txt'), 'a')
                f.writelines(['\n', '[Val {}]\n'.format(total_iteration),
                              'in_rmse: {:.5f}\n'.format(avg_rmse_in)])
                f.close()
                # save html
                utils.write_html(root_save_path, file_paths, total_iteration)

                in_unet.train()

            if total_iteration % args.lrStep == 0:
                scheduler.step()

            if total_iteration > args.iterations:
                return


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.autograd.set_detect_anomaly(True)

    utils.create_folder(args.outDir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    train_dataset = dataset.DesDataset(args.inDir, args.inlDir, log_input=not args.noLogInput, avg_path=args.avgDir,
                                       device=torch.device('cpu'))
    train_dataloader = DataLoaderX(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=7,
                                   drop_last=True)

    # Validation
    val_dataset = dataset.DesDatasetLoad(args.valInDir, args.valInlDir, log_input=not args.noLogInput)
    val_dataloader = DataLoaderX(val_dataset, batch_size=1, shuffle=True, num_workers=5)
    num_val_samples = len(val_dataset)

    # path to save model, imgs
    root_save_path = utils.create_folder(
        os.path.join(args.outDir, 'i_2in_deformconv_conditionalgan_perceptualloss_gammacorrection'), still_create=True)

    train(args, train_dataloader, val_dataloader, num_val_samples, root_save_path)
