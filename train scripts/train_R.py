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
parser.add_argument("--lrStep", type=int, default=60000)
parser.add_argument("--numSavedImgs", type=int, default=10, help="# of images to be saved during model saving")
parser.add_argument("--seed", type=int, default=990819)
parser.add_argument("--displayFreq", type=int, default=1, help="display losses (pbar)")
parser.add_argument("--summaryFreq", type=int, default=100, help="save losses")
parser.add_argument("--saveFreq", type=int, default=2000, help="save model every saveFreq steps, 0 to disable")
parser.add_argument("--valFreq", type=int, default=2000, help="test model every testFreq steps, 0 to disable")

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
    sm_net = model.MaskNet(1, device).to(device)
    print('sm_net')
    model.print_model_structure(sm_net)
    sm_net.load_state_dict(torch.load('sm_net.pt'))

    hm_net = model.MaskNet(1, device).to(device)
    print('hm_net')
    model.print_model_structure(hm_net)
    hm_net.load_state_dict(torch.load('hm_net.pt'))

    inl_unet = model.GatedDeformUNet(4, 3).to(device)
    print('inl_unet')
    model.print_model_structure(inl_unet)
    inl_unet.load_state_dict(torch.load('inl_unet.pt'))

    inlr_unet = model.GatedDeformUNet(4, 3).to(device)
    print('inlr_unet')
    model.print_model_structure(inlr_unet)
    inlr_unet.load_state_dict(torch.load('inlr_unet.pt'))

    R_unet = model.UNetGlobal(9, 1).to(device)
    print('R_unet')
    model.print_model_structure(R_unet)

    R_loss = torch.nn.L1Loss().to(device)
    render_loss = loss.RenderingLoss(args.batchSize, device, args.numDiffRendering, args.numSpecRendering,
                                     args.renderSize)
    optimizer = optim.Adam(R_unet.parameters(), lr=args.lr, betas=(0.5, 0.9))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5, last_epoch=-1)

    print("\t-Start training")
    sm_net.eval()
    hm_net.eval()
    inl_unet.eval()
    inlr_unet.eval()
    R_unet.train()
    save_img_interval = num_val_samples // args.numSavedImgs
    total_iteration = 0
    while True:
        for _, batch_sample in enumerate(train_dataloader):
            iter_start = time.time()
            inputs, materials = batch_sample
            inputs = inputs.permute(permutation).to(device).contiguous()
            materials = materials.permute(permutation).to(device).contiguous()

            with torch.no_grad():
                sm_out = sm_net(utils.deprocess(inputs))
                hm_out = hm_net(utils.deprocess(inputs))
                mask = utils.preprocess(torch.clip(sm_out + hm_out, 0, 1))
                inl_out = inl_unet(torch.cat([inputs, mask], dim=1))
                inlr_out = inlr_unet(torch.cat([inputs, mask], dim=1))
            R_out = R_unet(torch.cat([inputs, inl_out, inlr_out], dim=1))

            optimizer.zero_grad()
            loss_R_l1 = R_loss(R_out, materials[:, 6:7, :, :])
            render_materials = torch.cat([materials[:, :6, :, :], R_out, R_out, R_out,
                                          materials[:, 9:12, :, :]], dim=1)
            loss_render = render_loss(render_materials, materials)
            loss_all = loss_R_l1 + loss_render * 0.5
            loss_all.backward()
            optimizer.step()

            iter_took = time.time() - iter_start
            R_loss_l1_data = loss_R_l1.item()
            render_loss_data = loss_render.item()
            total_iteration += 1
            lr = optimizer.param_groups[0]['lr']

            if total_iteration % args.displayFreq == 0:
                print(
                    '[{}] iter_took:{:.3f}s  lr:{:.6f}  l1:{:.3f}  render:{:.3f}'.format(
                        total_iteration, iter_took, lr, R_loss_l1_data, render_loss_data))

            if total_iteration % args.summaryFreq == 0:
                f = open(os.path.join(root_save_path, 'loss.txt'), 'a')
                f.writelines(['\n', '[iteration: {}  lr:{:.8f}]\n'.format(total_iteration, lr),
                              'l1: {:.3f}\n'.format(R_loss_l1_data),
                              'render: {:.3f}\n'.format(render_loss_data)])
                f.close()

            if total_iteration % args.saveFreq == 0:
                current_save_path = utils.create_folder(os.path.join(root_save_path, 'iter%d' % total_iteration))
                torch.save(R_unet.state_dict(), os.path.join(current_save_path, "R_unet.pt"))

            if total_iteration % args.valFreq == 0:
                R_unet.eval()
                file_paths = []
                with torch.no_grad():
                    avg_rmse_R = 0
                    for i_batch, batch_sample in enumerate(val_dataloader):
                        inputs, inl_gt, inlr_gt, materials, name = batch_sample
                        inputs = inputs.permute(permutation).to(device)

                        with torch.no_grad():
                            sm_out = sm_net(utils.deprocess(inputs))
                            hm_out = hm_net(utils.deprocess(inputs))
                            mask = utils.preprocess(torch.clip(sm_out + hm_out, 0, 1))
                            inl_out = inl_unet(torch.cat([inputs, mask], dim=1))
                            inlr_out = inlr_unet(torch.cat([inputs, mask], dim=1))
                        R_out = R_unet(torch.cat([inputs, inl_out, inlr_out], dim=1))

                        inputs = inputs.permute(depermutation)
                        inl_out = inl_out.permute(depermutation)
                        inlr_out = inlr_out.permute(depermutation)
                        R_out = R_out.permute(depermutation)

                        if i_batch % save_img_interval == 0 and i_batch != 0:
                            current_save_path = utils.create_folder(os.path.join(root_save_path, 'iter%d' % total_iteration))
                            files = utils.save_maps(materials.cpu().numpy()[0],
                                                    R_out.cpu().numpy()[0],
                                                    utils.deprocess(inputs.cpu()).numpy()[0],
                                                    utils.deprocess(inl_gt.cpu()).numpy()[0],
                                                    utils.deprocess(inl_out.cpu()).numpy()[0],
                                                    utils.deprocess(inlr_gt.cpu()).numpy()[0],
                                                    utils.deprocess(inlr_out.cpu()).numpy()[0],
                                                    args.noLogOutputAlbedo, current_save_path, i_batch)
                            files.append(name)
                            file_paths.append(files)
                        avg_rmse_R += utils.calculate_root_mse(utils.deprocess(R_out).cpu().numpy(),
                                                               utils.deprocess(materials[:, :, :, 6:7]).cpu().numpy())
                avg_rmse_R /= num_val_samples

                print('[Val {}] R_rmse:{:.5f}'.format(total_iteration, avg_rmse_R))
                # save validation results
                f = open(os.path.join(root_save_path, 'validation.txt'), 'a')
                f.writelines(['\n', '[Val {}]\n'.format(total_iteration),
                              'R_rmse: {:.5f}\n'.format(avg_rmse_R)])
                f.close()
                # save html
                utils.write_html(root_save_path, file_paths, total_iteration)

                R_unet.train()

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
        os.path.join(args.outDir, 'i_hybridmaskgt_efficientB3_learnedDecoder_2inl_concatI_deformconv_conditionalgan_gammacorrection_NEWinl_[globalunet_R]'), still_create=True)

    train(args, train_dataloader, val_dataloader, num_val_samples, root_save_path)
