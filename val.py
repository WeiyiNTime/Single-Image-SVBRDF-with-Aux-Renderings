import glob
import os

import cv2
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
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--modelPath", type=str, default=None) # pre-trained models
parser.add_argument("--inDir", type=str, required=True) # directory of input images
parser.add_argument("--outDir", type=str, required=True) # directory of results
# parser.add_argument("--avgDir", type=str, required=True) 
parser.add_argument("--gtDir", type=str)
parser.add_argument("--TestType", type=str, default='Des18') # Des18, Des19, real

# training configs
parser.add_argument("--iterations", type=int, default=230001)
parser.add_argument("--batchSize", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--lrStep", type=int, default=50000)
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

def val(args, val_dataloader, num_val_samples, root_save_path):
    # load pre-trained models
    in_unet = model.GatedDeformUNet(3, 3).to(device)
    in_unet.load_state_dict(torch.load(args.modelPath+'/in_unet.pt'))
    N_unet = model.UNetGlobal(6, 2).to(device)
    N_unet.load_state_dict(torch.load(args.modelPath+'/N_unet.pt'))
    sm_net = model.MaskNet(1, device).to(device)
    sm_net.load_state_dict(torch.load(args.modelPath+'/sm_net.pt'))
    hm_net = model.MaskNet(1, device).to(device)
    hm_net.load_state_dict(torch.load(args.modelPath+'/hm_net.pt'))
    inl_unet = model.GatedDeformUNet(4, 3).to(device)
    inl_unet.load_state_dict(torch.load(args.modelPath+'/inl_unet.pt'))
    inlr_unet = model.GatedDeformUNet(4, 3).to(device)
    inlr_unet.load_state_dict(torch.load(args.modelPath+'/inlr_unet.pt'))
    DS_unet = model.UNetGlobal(9, 6).to(device)
    DS_unet.load_state_dict(torch.load(args.modelPath+'/DS_unet0.pt', map_location='cuda:0'))
    R_unet = model.UNetGlobal(9, 1).to(device)
    R_unet.load_state_dict(torch.load(args.modelPath+'/R_unet.pt'))
    in_unet.eval()
    N_unet.eval()
    sm_net.eval()
    hm_net.eval()
    inl_unet.eval()
    inlr_unet.eval()
    DS_unet.eval()
    R_unet.eval()

    file_paths = []
    iter = 0
    with torch.no_grad():
        avg_rmse_N = 0
        avg_rmse_D = 0
        avg_rmse_R = 0
        avg_rmse_S = 0
        for i_batch, batch_sample in enumerate(val_dataloader):
            inputs, inl_gt, inlr_gt, materials, name = batch_sample
            inputs = inputs.permute(permutation).to(device)

            with torch.no_grad():
                # compute mask images
                sm_out = sm_net(utils.deprocess(inputs))
                hm_out = hm_net(utils.deprocess(inputs))
                mask = utils.preprocess(torch.clip(sm_out + hm_out, 0, 1))
                # compute I_bf
                in_out = in_unet(inputs)
                # compute I_hr, I_hr^2
                inl_out = inl_unet(torch.cat([inputs, mask], dim=1))
                inlr_out = inlr_unet(torch.cat([inputs, mask], dim=1))
                # compute N
                diff = (inputs - in_out) / 2  # to -1 ~ 1
                N_out = N_unet(torch.cat([inputs, diff], dim=1))
                N_out = utils.deprocess_N(N_out, device)
                # compute D and S
                DS_out = DS_unet(torch.cat([inputs, inl_out, inlr_out], dim=1))
                # compute R
                R_out = R_unet(torch.cat([inputs, inl_out, inlr_out], dim=1))
                
                materials_out = torch.cat([N_out, DS_out[:, :3, :, :], R_out, R_out, R_out, DS_out[:, 3:6, :, :]], dim=1)

            # save results
            inputs = inputs.permute(depermutation)
            inl_out = inl_out.permute(depermutation)
            inlr_out = inlr_out.permute(depermutation)
            N_out = N_out.permute(depermutation)
            DS_out = DS_out.permute(depermutation)
            R_out = R_out.permute(depermutation)
            materials_out = materials_out.permute(depermutation)

            current_save_path = utils.create_folder(os.path.join(root_save_path, 'iter%d' % iter))
            fea_out_save_path = utils.create_folder(os.path.join(root_save_path, 'fea'))
            utils.save_maps_fea(materials_out.clone().cpu().numpy()[0], args.noLogOutputAlbedo, fea_out_save_path, i_batch, name)

            files = utils.save_maps(materials.cpu().numpy()[0],
                                    materials_out.cpu().numpy()[0],
                                    utils.deprocess(inputs.cpu()).numpy()[0],
                                    utils.deprocess(inl_gt.cpu()).numpy()[0],
                                    utils.deprocess(inl_out.cpu()).numpy()[0],
                                    utils.deprocess(inlr_gt.cpu()).numpy()[0],
                                    utils.deprocess(inlr_out.cpu()).numpy()[0],
                                    args.noLogOutputAlbedo, current_save_path, i_batch, name[0].split(';')[0])
            files.append(name)
            file_paths.append(files)

            # compute RMSE
            avg_rmse_N += utils.calculate_root_mse(utils.deprocess(N_out).cpu().numpy(),
                                                   utils.deprocess(materials[:, :, :, :3]).cpu().numpy())

            avg_rmse_R += utils.calculate_root_mse(utils.deprocess(R_out).cpu().numpy(),
                                                   utils.deprocess(materials[:, :, :, 6:7]).cpu().numpy())
            avg_rmse_D += utils.calculate_root_mse(utils.deprocess(DS_out[:, :, :, :3]).cpu().numpy(),
                                                   utils.deprocess(materials[:, :, :, 3:6]).cpu().numpy())
            avg_rmse_S += utils.calculate_root_mse(utils.deprocess(DS_out[:, :, :, 3:6]).cpu().numpy(),
                                                   utils.deprocess(materials[:, :, :, 9:12]).cpu().numpy())
    avg_rmse_N /= num_val_samples
    avg_rmse_R /= num_val_samples
    avg_rmse_D /= num_val_samples
    avg_rmse_S /= num_val_samples

    print('[Val {}] N_rmse:{:.5f}  D_rmse:{:.5f}  R_rmse:{:.5f}  S_rmse:{:.5f}'.format(iter, avg_rmse_N, avg_rmse_D,
                                                                                       avg_rmse_R, avg_rmse_S))
    # save validation results
    f = open(os.path.join(root_save_path, 'validation.txt'), 'a')
    f.writelines(['\n', '[Val {}]\n'.format(iter),
                  'N_rmse: {:.5f}\n'.format(avg_rmse_N),
                  'D_rmse: {:.5f}\n'.format(avg_rmse_D),
                  'R_rmse: {:.5f}\n'.format(avg_rmse_R),
                  'S_rmse: {:.5f}\n'.format(avg_rmse_S)])
    f.close()
    # save html
    utils.write_html(root_save_path, file_paths, iter)

def val_real(args, val_dataloader, num_val_samples, root_save_path):
    in_unet = model.GatedDeformUNet(3, 3).to(device)
    in_unet.load_state_dict(torch.load(args.modelPath+'/in_unet.pt'))
    N_unet = model.UNetGlobal(6, 2).to(device)
    N_unet.load_state_dict(torch.load(args.modelPath+'/N_unet.pt'))
    sm_net = model.MaskNet(1, device).to(device)
    sm_net.load_state_dict(torch.load(args.modelPath+'/sm_net.pt'))
    hm_net = model.MaskNet(1, device).to(device)
    hm_net.load_state_dict(torch.load(args.modelPath+'/hm_net.pt'))
    inl_unet = model.GatedDeformUNet(4, 3).to(device)
    inl_unet.load_state_dict(torch.load(args.modelPath+'/inl_unet.pt'))
    inlr_unet = model.GatedDeformUNet(4, 3).to(device)
    inlr_unet.load_state_dict(torch.load(args.modelPath+'/inlr_unet.pt'))
    DS_unet = model.UNetGlobal(9, 6).to(device)
    DS_unet.load_state_dict(torch.load(args.modelPath+'/DS_unet0.pt', map_location='cuda:0'))
    R_unet = model.UNetGlobal(9, 1).to(device)
    R_unet.load_state_dict(torch.load(args.modelPath+'/R_unet.pt'))
    in_unet.eval()
    N_unet.eval()
    sm_net.eval()
    hm_net.eval()
    inl_unet.eval()
    inlr_unet.eval()
    DS_unet.eval()
    R_unet.eval()

    file_paths = []
    iter = 0
    with torch.no_grad():
        for i_batch, batch_sample in enumerate(val_dataloader):
            inputs, name = batch_sample
            inputs = inputs.permute(permutation).to(device)

            with torch.no_grad():
                current_save_path = utils.create_folder(os.path.join(root_save_path, 'mask'))
                sm_out = sm_net(utils.deprocess(inputs))
                hm_out = hm_net(utils.deprocess(inputs))
                
                mask = utils.preprocess(torch.clip(sm_out + hm_out, 0, 1))
                inl_out = inl_unet(torch.cat([inputs, mask], dim=1))
                inlr_out = inlr_unet(torch.cat([inputs, mask], dim=1))

                in_out = in_unet(inputs)
                diff = (inputs - in_out) / 2  # to -1 ~ 1
                N_out = N_unet(torch.cat([inputs, diff], dim=1))

                N_out = utils.deprocess_N(N_out, device)
                DS_out = DS_unet(torch.cat([inputs, inl_out, inlr_out], dim=1))
                R_out = R_unet(torch.cat([inputs, inl_out, inlr_out], dim=1))
                materials_out = torch.cat([N_out, DS_out[:, :3, :, :], R_out, R_out, R_out, DS_out[:, 3:6, :, :]], dim=1)

            inputs = inputs.permute(depermutation)
            materials_out = materials_out.permute(depermutation)

            current_save_path = utils.create_folder(os.path.join(root_save_path, "estimated-maps"))
            files = utils.save_maps_real(materials_out.cpu().numpy()[0],
                                         utils.deprocess(inputs.cpu()).numpy()[0],
                                         args.noLogOutputAlbedo, current_save_path, name[0])
            print('\r%d / %d' % (i_batch+1, num_val_samples), end='')

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.autograd.set_detect_anomaly(True)

    utils.create_folder(args.outDir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # path to save model, imgs
    root_save_path = utils.create_folder(
        os.path.join(args.outDir, 'result-'+ args.TestType), still_create=True)


    if args.TestType=='real':
        val_dataset = dataset.RealDataset(args.inDir)
        val_dataloader = DataLoaderX(val_dataset, batch_size=1, shuffle=True, num_workers=5)
        num_val_samples = len(val_dataset)
        val_real(args, val_dataloader, num_val_samples, root_save_path)
        
    elif args.TestType=='Des18':
        val_dataset = dataset.Des18DatasetLoad(args.gtDir, args.inDir, log_input=not args.noLogInput)
        val_dataloader = DataLoaderX(val_dataset, batch_size=1, shuffle=False, num_workers=5)
        num_val_samples = len(val_dataset)
        val(args, val_dataloader, num_val_samples, root_save_path)
    else:
        val_dataset = dataset.Des19DatasetLoad(args.gtDir, args.inDir, log_input=not args.noLogInput)
        val_dataloader = DataLoaderX(val_dataset, batch_size=1, shuffle=False, num_workers=5)
        num_val_samples = len(val_dataset)
        val(args, val_dataloader, num_val_samples, root_save_path)

