import cv2
import numpy as np
import torch
from einops import rearrange
from skimage import transform
from torchvision.transforms import functional as F
import math
import random
import os
import itertools
import time
import warnings
from PIL import Image
import matplotlib.pyplot as plt


def read_material(material):
    materials = cv2.imread(material)
    materials = cv2.cvtColor(materials, cv2.COLOR_BGR2RGB) / 255.0
    return materials


def read_material_tiled(material):
    materials = cv2.imread(material)
    materials = cv2.cvtColor(materials, cv2.COLOR_BGR2RGB)
    materials = np.array(np.array_split(materials, 4, axis=1)) / 255.0
    return materials


def read_image_material_tiled(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array_split(image / 255.0, 5, axis=1)
    return image


def read_img_materials_auxrend(image, split):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array_split(image / 255.0, split, axis=1)
    return image


def read_material_separated(normal, diffuse, roughness, specular, part_count=16):
    part_idx = random.randint(1, part_count)
    normal = normal.replace('.png', '#%d.png' % part_idx)
    diffuse = diffuse.replace('.png', '#%d.png' % part_idx)
    roughness = roughness.replace('.png', '#%d.png' % part_idx)
    specular = specular.replace('.png', '#%d.png' % part_idx)

    normal = cv2.imread(normal)
    normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB) / 255.0

    diffuse = cv2.imread(diffuse)
    diffuse = cv2.cvtColor(diffuse, cv2.COLOR_BGR2RGB) / 255.0

    roughness = cv2.imread(roughness)
    roughness = cv2.cvtColor(roughness, cv2.COLOR_BGR2RGB) / 255.0

    specular = cv2.imread(specular)
    specular = cv2.cvtColor(specular, cv2.COLOR_BGR2RGB) / 255.0
    return normal, diffuse, roughness, specular


def read_material_group(maps, part_count=16):
    temp = maps
    part_idx = random.randint(1, part_count)
    for i in range(4):
        if isinstance(maps[i], list):
            for p in range(2):
                t = cv2.imread(maps[i][p].replace('.png', '#%d.png' % part_idx))
                temp[i][p] = cv2.cvtColor(t, cv2.COLOR_BGR2RGB) / 255.0
        else:
            t = cv2.imread(maps[i].replace('.png', '#%d.png' % part_idx))
            temp[i] = cv2.cvtColor(t, cv2.COLOR_BGR2RGB) / 255.0
    return temp


# Make sure the roughness is between 0.1 and 1.0 to avoid having pure mirror (with nothing to reflect it will be black)
def adapt_roughness(material, device):
    multiplier = torch.tensor([1.0, 1.0, 0.9, 1.0], requires_grad=False).to(device)
    addition = torch.tensor([0.0, 0.0, 0.1, 0.0], requires_grad=False).to(device)
    multiplier = torch.reshape(multiplier, [4, 1, 1, 1])
    addition = torch.reshape(addition, [4, 1, 1, 1])
    return (material * multiplier) + addition


# Mix the materials, particularly careful on the normals as they are vector fields
def mix_materials(n1, r1, b1, m1, n2, r2, b2, m2, alpha):
    n1_corrected = (n1 - 0.5) * 2.0  # go between -1 and 1
    n2_corrected = (n2 - 0.5) * 2.0
    n1_projected = n1_corrected / torch.unsqueeze(torch.clamp(n1_corrected[:, :, 2], min=0.01), dim=-1)
    n2_projected = n2_corrected / torch.unsqueeze(torch.clamp(n2_corrected[:, :, 2], min=0.01), dim=-1)

    mixed_n = alpha * n1_projected + (1.0 - alpha) * n2_projected
    normalized_n = mixed_n / torch.sqrt(torch.sum(torch.square(mixed_n), dim=-1, keepdim=True))
    n_mix = (normalized_n * 0.5) + 0.5

    r_mix = alpha * r1 + (1.0 - alpha) * r2
    b_mix = alpha * b1 + (1.0 - alpha) * b2
    m_mix = alpha * m1 + (1.0 - alpha) * m2
    return n_mix, r_mix, b_mix, m_mix


# ================================basecolor_metallic_to_diffuse_specular================================
def convert_to_sRGB(img):
    return torch.pow(img, 0.4545)


def convert_to_linear(img):
    return torch.pow(img, 2.2)


def blend_add(foreground, background, opacity=None):
    if opacity:
        out = background + foreground * opacity
    else:
        out = background + foreground
    return torch.clamp(out, max=1.0)


def blend_copy(foreground, background, opacity):
    return foreground * opacity + background * (1 - opacity)


# according to substance designer's "bm_to_ds" node
def basecolor_metallic_to_diffuse_specular(basecolor, metallic):
    # level_out = level(metallic, 0, 1, 0.5, 1, 0)
    level_out = 1 - metallic
    diff_level_srgb = convert_to_sRGB(level_out)
    # diff_level_out = level(diff_level_srgb, 0, 1, 0.5, 1, 0)
    diff_level_out = 1 - diff_level_srgb
    uniform_color = torch.zeros_like(basecolor, device=basecolor.device)
    basecolor_linear = convert_to_linear(basecolor)

    # spec_level_out = level(level_out, 0, 1, 0.5, 0, 0.04)
    spec_level_out = level_out * 0.04
    spec_blend1 = blend_copy(uniform_color, basecolor_linear, level_out)
    spec_blend2 = blend_add(spec_level_out, spec_blend1)
    specular = convert_to_sRGB(spec_blend2)

    diffuse = blend_copy(uniform_color, basecolor, diff_level_out)

    return diffuse, specular


# ================================basecolor_metallic_to_diffuse_specular================================


def angle_random_picker(delta):
    a = range(0, delta)
    b = range(90 - delta, 90 + delta)
    c = range(180 - delta, 180 + delta)
    d = range(270 - delta, 270 + delta)
    e = range(360 - delta, 360)
    merged_range = list(itertools.chain(a, b, c, d, e))
    value = random.choice(merged_range)
    return value


def preprocess(image):
    # [0, 1] => [-1, 1]
    return image * 2 - 1


def deprocess(image):
    # [-1, 1] => [0, 1]
    return (image + 1) / 2


# Log a tensor and normalize it
def log_tensor(tensor):
    return (torch.log(torch.add(tensor, 0.01)) - np.log(0.01)) / (np.log(1.01) - np.log(0.01))


def de_log_tensor(tensor):
    return torch.exp(tensor * (np.log(1.01) - np.log(0.01)) + np.log(0.01)) - 0.01


def np_log_tensor(map):
    return (np.log(map + 0.01) - np.log(0.01)) / (np.log(1.01) - np.log(0.01))


def rand_range(shape, low, high, dtype=torch.float32):
    return ((high - low) * torch.rand(shape, dtype=dtype))


def randn_range(shape, mean, std, dtype=torch.float32):
    return (torch.randn(shape, dtype=dtype) * std + mean)


def material_reshape(material):
    # Here [4, 256, 256, 3] and we want to go to [256, 256, 12]
    return rearrange(material, 'n h w c -> h w (n c)')


def material_deshape(material):
    # Here [256, 256, 12] and we want to go to [4, 256, 256, 3]
    return rearrange(material, 'h w (n c) -> n h w c', c=3)


def convert(image, squeeze=False):
    if squeeze:
        def tempLog(imageValue):
            imageValue = torch.log(imageValue + 0.01)
            imageValue = imageValue - torch.min(imageValue)
            imageValue = imageValue / torch.max(imageValue)
            return imageValue

        image = [tempLog(imageVal) for imageVal in image]

    return F.convert_image_dtype(image, dtype=torch.uint8)


# Normalizes a tensor throughout the Channels dimension (Width, Height, Channels)
# Keeps 3rd dimension to 1. Output will be (Width, Height, 1)
def normalize(tensor, dim=-1):
    length = torch.sqrt(torch.sum(torch.square(tensor), dim=dim, keepdim=True))
    return torch.div(tensor, length)


# Computes the dot product between 2 tensors (Width, Height, Channels)
# Keeps 3rd dimension to 1. Output will be (Width, Height, 1).
def dot_product(tensor_A, tensor_B):
    return torch.sum(torch.multiply(tensor_A, tensor_B), dim=-1, keepdim=True)


# Clip values between min an max
def squeeze_values(tensor, min, max):
    return torch.clip(tensor, min, max)


# Adds a little bit of noise
def add_noise(renderings_shape, device):
    stddevNoise = np.exp(np.random.normal(np.log(0.005), 0.3, size=()))
    noise = torch.normal(0.0, stddevNoise, size=renderings_shape).to(device)
    return noise


# Physically based lamp attenuation
def lamp_attenuation_pbr(distance):
    return 1.0 / torch.square(distance)


# Put the normals and roughness back to 3 channel for easier processing.
# model output [B, 9, H, W] -> [B, 12, H, W]
def deprocess_outputs(outputs, device):
    # The multiplication here gives space to generate direction with angle > pi/4
    partialOutputedNormals = outputs[:, 0:2, :, :]  # * 3.0
    outputedDiffuse = outputs[:, 2:5, :, :]
    outputedRoughness = outputs[:, 5:6, :, :]
    outputedSpecular = outputs[:, 6:9, :, :]

    normalShape = partialOutputedNormals.shape
    newShape = [normalShape[0], 1, normalShape[2], normalShape[3]]
    tmpNormals = torch.ones(newShape).to(device)
    normNormals = normalize(torch.cat([partialOutputedNormals, tmpNormals], dim=1), dim=1)

    return torch.cat(
        [normNormals, outputedDiffuse, outputedRoughness, outputedRoughness, outputedRoughness, outputedSpecular],
        dim=1)


def deprocess_N(outputs, device):
    shape = outputs.shape
    new_shape = [shape[0], 1, shape[2], shape[3]]
    tmp_N = torch.ones(new_shape, device=device)
    norm_N = normalize(torch.cat([outputs, tmp_N], dim=1), dim=1)
    return norm_N


# Put the normals and roughness back to 3 channel for easier processing.
# model output [B, 7, H, W] -> [B, 9, H, W]
def deprocess_outputs_drs(outputs):
    # The multiplication here gives space to generate direction with angle > pi/4
    outputedDiffuse = outputs[:, :3, :, :]
    outputedRoughness = outputs[:, 3:4, :, :]
    outputedSpecular = outputs[:, 4:7, :, :]

    return torch.cat([outputedDiffuse, outputedRoughness, outputedRoughness, outputedRoughness, outputedSpecular],
                     dim=1)


# Deprocess an image to be visible
def save_maps(targets, outputs, inputs, inl_gt, inl_out, inlr_gt, inlr_out, no_log_albedo, current_save_path, i_batch,
              name):
    targets = np.clip(deprocess(targets), 0, 1)
    outputs = np.clip(deprocess(outputs), 0, 1)
    inputs *= 255
    inl_gt = np.clip(inl_gt, 0, 1)
    inl_out = np.clip(inl_out, 0, 1)
    inl_gt *= 255
    inl_out *= 255
    inlr_gt = np.clip(inlr_gt, 0, 1)
    inlr_out = np.clip(inlr_out, 0, 1)
    inlr_gt *= 255
    inlr_out *= 255

    if not no_log_albedo:
        targets[:, :, 3:6] = np.power(targets[:, :, 3:6], 0.4545)
        targets[:, :, 9:] = np.power(targets[:, :, 9:], 0.4545)
        outputs[:, :, 3:6] = np.power(outputs[:, :, 3:6], 0.4545)
        outputs[:, :, 9:] = np.power(outputs[:, :, 9:], 0.4545)
    targets *= 255
    target_N = Image.fromarray(np.uint8(targets[:, :, :3]))
    target_N_path = os.path.join(current_save_path, 'gt_N_%d.png' % i_batch)
    target_N.save(target_N_path)
    target_D = Image.fromarray(np.uint8(targets[:, :, 3:6]))
    target_D_path = os.path.join(current_save_path, 'gt_D_%d.png' % i_batch)
    target_D.save(target_D_path)
    target_R = Image.fromarray(np.uint8(targets[:, :, 6:9]))
    target_R_path = os.path.join(current_save_path, 'gt_R_%d.png' % i_batch)
    target_R.save(target_R_path)
    target_S = Image.fromarray(np.uint8(targets[:, :, 9:]))
    target_S_path = os.path.join(current_save_path, 'gt_S_%d.png' % i_batch)
    target_S.save(target_S_path)
    outputs *= 255
    output_N = Image.fromarray(np.uint8(outputs[:, :, :3]))
    output_N_path = os.path.join(current_save_path, '%s_Nout.png' % name)
    output_N.save(output_N_path)
    output_D = Image.fromarray(np.uint8(outputs[:, :, 3:6]))
    output_D_path = os.path.join(current_save_path, '%s_Dout.png' % name)
    output_D.save(output_D_path)
    output_R = Image.fromarray(np.uint8(outputs[:, :, 6:9]))
    output_R_path = os.path.join(current_save_path, '%s_Rout.png' % name)
    output_R.save(output_R_path)
    output_S = Image.fromarray(np.uint8(outputs[:, :, 9:]))
    output_S_path = os.path.join(current_save_path, '%s_Sout.png' % name)
    output_S.save(output_S_path)

    img = Image.fromarray(np.uint8(inputs))
    img_path = os.path.join(current_save_path, '%s_input.png' % name)
    img.save(img_path)
    inl_gt = Image.fromarray(np.uint8(inl_gt))
    inl_gt_path = os.path.join(current_save_path, 'inl_gt_%d.png' % i_batch)
    inl_gt.save(inl_gt_path)
    inl_out = Image.fromarray(np.uint8(inl_out))
    inl_out_path = os.path.join(current_save_path, '%s_inlout.png' % name)
    inl_out.save(inl_out_path)
    inlr_gt = Image.fromarray(np.uint8(inlr_gt))
    inlr_gt_path = os.path.join(current_save_path, 'inlr_gt_%d.png' % i_batch)
    inlr_gt.save(inlr_gt_path)
    inlr_out = Image.fromarray(np.uint8(inlr_out))
    inlr_out_path = os.path.join(current_save_path, '%s_inlrout.png' % name)
    inlr_out.save(inlr_out_path)

    target_N_path = os.path.join(current_save_path.split('\\')[-1], 'gt_N_%d.png' % i_batch)
    target_D_path = os.path.join(current_save_path.split('\\')[-1], 'gt_D_%d.png' % i_batch)
    target_R_path = os.path.join(current_save_path.split('\\')[-1], 'gt_R_%d.png' % i_batch)
    target_S_path = os.path.join(current_save_path.split('\\')[-1], 'gt_S_%d.png' % i_batch)
    output_N_path = os.path.join(current_save_path.split('\\')[-1], '%s_Nout.png' % name)
    output_D_path = os.path.join(current_save_path.split('\\')[-1], '%s_Dout.png' % name)
    output_R_path = os.path.join(current_save_path.split('\\')[-1], '%s_Rout.png' % name)
    output_S_path = os.path.join(current_save_path.split('\\')[-1], '%s_Sout.png' % name)
    img_path = os.path.join(current_save_path.split('\\')[-1], '%s_input.png' % name)
    inl_gt_path = os.path.join(current_save_path.split('\\')[-1], 'inl_gt_%d.png' % i_batch)
    inl_out_path = os.path.join(current_save_path.split('\\')[-1], '%s_inlout.png' % name)
    inlr_gt_path = os.path.join(current_save_path.split('\\')[-1], 'inlr_gt_%d.png' % i_batch)
    inlr_out_path = os.path.join(current_save_path.split('\\')[-1], '%s_inlrout.png' % name)
    return [target_N_path, target_D_path, target_R_path, target_S_path, output_N_path, output_D_path, output_R_path,
            output_S_path, img_path, inl_gt_path, inl_out_path, inlr_gt_path, inlr_out_path]


def save_maps_fea(outputs, no_log_albedo, current_save_path, i_batch, name):
    outputs = np.clip(deprocess(outputs), 0, 1)  # 256*256*12

    if not no_log_albedo:
        outputs[:, :, 3:6] = np.power(outputs[:, :, 3:6], 0.4545)
        outputs[:, :, 9:] = np.power(outputs[:, :, 9:], 0.4545)
    outputs *= 255

    outputs = rearrange(outputs, 'h w (n c) -> h (n w) c', c=3)

    output_img = Image.fromarray(np.uint8(outputs))
    output_path = os.path.join(current_save_path, '%s.png' % name)
    output_img.save(output_path)


# def save_maps_real(outputs, inputs, inl_out, inlr_out, no_log_albedo, current_save_path, name):
def save_maps_real(outputs, inputs, no_log_albedo, current_save_path, name):
    outputs = np.clip(deprocess(outputs), 0, 1) # 256*256*12
    inputs *= 255
    inputs = np.uint8(inputs)
    # inl_out = np.clip(inl_out, 0, 1)
    # inl_out *= 255
    # inlr_out = np.clip(inlr_out, 0, 1)
    # inlr_out *= 255

    if not no_log_albedo:
        outputs[:, :, 3:6] = np.power(outputs[:, :, 3:6], 0.4545)
        outputs[:, :, 9:] = np.power(outputs[:, :, 9:], 0.4545)
    outputs *= 255
    outputs = np.uint8(outputs)

    outputs = rearrange(outputs, 'h w (n c)-> h (n w) c', c=3)

    # output_N = Image.fromarray(np.uint8(outputs[:, :, :3]))
    # output_N_path = os.path.join(current_save_path, '%s_Nout.png' % name)
    # output_N.save(output_N_path)
    # output_D = Image.fromarray(np.uint8(outputs[:, :, 3:6]))
    # output_D_path = os.path.join(current_save_path, '%s_Dout.png' % name)
    # output_D.save(output_D_path)
    # output_R = Image.fromarray(np.uint8(outputs[:, :, 6:9]))
    # output_R_path = os.path.join(current_save_path, '%s_Rout.png' % name)
    # output_R.save(output_R_path)
    # output_S = Image.fromarray(np.uint8(outputs[:, :, 9:]))
    # output_S_path = os.path.join(current_save_path, '%s_Sout.png' % name)
    # output_S.save(output_S_path)

    outputs_img = Image.fromarray(outputs)
    outputs_save_path = os.path.join(current_save_path,'%s.png' % name)
    outputs_img.save(outputs_save_path)


    # img = Image.fromarray(np.uint8(inputs))
    # img_path = os.path.join(current_save_path, '%s_input.png' % name)
    # img.save(img_path)
    # inl_out = Image.fromarray(np.uint8(inl_out))
    # inl_out_path = os.path.join(current_save_path, '%s_inlout.png' % name)
    # inl_out.save(inl_out_path)
    # inlr_out = Image.fromarray(np.uint8(inlr_out))
    # inlr_out_path = os.path.join(current_save_path, '%s_inlrout.png' % name)
    # inlr_out.save(inlr_out_path)

    output_N_path = os.path.join(current_save_path, '%s_Nout.png' % name)
    output_D_path = os.path.join(current_save_path, '%s_Dout.png' % name)
    output_R_path = os.path.join(current_save_path, '%s_Rout.png' % name)
    output_S_path = os.path.join(current_save_path, '%s_Sout.png' % name)
    img_path = os.path.join(current_save_path, '%s_input.png' % name)
    # inl_out_path = os.path.join(current_save_path.split('\\')[-1], '%s_inlout.png' % name)
    # inlr_out_path = os.path.join(current_save_path.split('\\')[-1], '%s_inlrout.png' % name)
    # return [output_N_path, output_D_path, output_R_path, output_S_path, img_path, inl_out_path, inlr_out_path]
    return [output_N_path, output_D_path, output_R_path, output_S_path, img_path]


# ================================================ teaser ==============================================================
def save_maps_teaser(outputs, sm_out, hm_out, inputs, in_out, inl_out, inlr_out, renderings, renderings_gt,
                     current_save_path, name):
    outputs = np.clip(deprocess(outputs), 0, 1)
    inputs *= 255
    sm_out *= 255
    sm_out = np.concatenate([sm_out, sm_out, sm_out], axis=-1)
    hm_out *= 255
    hm_out = np.concatenate([hm_out, hm_out, hm_out], axis=-1)
    in_out = np.clip(in_out, 0, 1)
    in_out *= 255
    inl_out = np.clip(inl_out, 0, 1)
    inl_out *= 255
    inlr_out = np.clip(inlr_out, 0, 1)
    inlr_out *= 255

    renderings = np.power(renderings, 0.4545)
    renderings *= 255
    renderings_gt = np.power(renderings_gt, 0.4545)
    renderings_gt *= 255

    outputs[:, :, 3:6] = np.power(outputs[:, :, 3:6], 0.4545)
    outputs[:, :, 9:] = np.power(outputs[:, :, 9:], 0.4545)
    outputs *= 255
    outputs_tiled = np.concatenate([np.concatenate([outputs[:, :, :3], outputs[:, :, 3:6]], axis=1),
                                    np.concatenate([outputs[:, :, 6:9], outputs[:, :, 9:]], axis=1)], axis=0)
    outputs_tiled = Image.fromarray(np.uint8(outputs_tiled))
    outputs_tiled_path = os.path.join(current_save_path, '%s_out.png' % name)
    outputs_tiled.save(outputs_tiled_path)

    img = Image.fromarray(np.uint8(inputs))
    img_path = os.path.join(current_save_path, '%s_input.png' % name)
    img.save(img_path)
    sm_out = Image.fromarray(np.uint8(sm_out))
    sm_out_path = os.path.join(current_save_path, '%s_sm_out.png' % name)
    sm_out.save(sm_out_path)
    hm_out = Image.fromarray(np.uint8(hm_out))
    hm_out_path = os.path.join(current_save_path, '%s_hm_out.png' % name)
    hm_out.save(hm_out_path)
    in_out = Image.fromarray(np.uint8(in_out))
    in_out_path = os.path.join(current_save_path, '%s_inout.png' % name)
    in_out.save(in_out_path)
    inl_out = Image.fromarray(np.uint8(inl_out))
    inl_out_path = os.path.join(current_save_path, '%s_inlout.png' % name)
    inl_out.save(inl_out_path)
    inlr_out = Image.fromarray(np.uint8(inlr_out))
    inlr_out_path = os.path.join(current_save_path, '%s_inlrout.png' % name)
    inlr_out.save(inlr_out_path)

    renderings_1 = Image.fromarray(np.uint8(renderings[0]))
    renderings_1_path = os.path.join(current_save_path, '%s_render1.png' % name)
    renderings_1.save(renderings_1_path)
    renderings_2 = Image.fromarray(np.uint8(renderings[1]))
    renderings_2_path = os.path.join(current_save_path, '%s_render2.png' % name)
    renderings_2.save(renderings_2_path)
    renderings_3 = Image.fromarray(np.uint8(renderings[2]))
    renderings_3_path = os.path.join(current_save_path, '%s_render3.png' % name)
    renderings_3.save(renderings_3_path)
    renderings_4 = Image.fromarray(np.uint8(renderings[3]))
    renderings_4_path = os.path.join(current_save_path, '%s_render4.png' % name)
    renderings_4.save(renderings_4_path)
    # gt
    renderings_gt_1 = Image.fromarray(np.uint8(renderings_gt[0]))
    renderings_gt_1_path = os.path.join(current_save_path, '%s_render_gt1.png' % name)
    renderings_gt_1.save(renderings_gt_1_path)
    renderings_gt_2 = Image.fromarray(np.uint8(renderings_gt[1]))
    renderings_gt_2_path = os.path.join(current_save_path, '%s_render_gt2.png' % name)
    renderings_gt_2.save(renderings_gt_2_path)
    renderings_gt_3 = Image.fromarray(np.uint8(renderings_gt[2]))
    renderings_gt_3_path = os.path.join(current_save_path, '%s_render_gt3.png' % name)
    renderings_gt_3.save(renderings_gt_3_path)
    renderings_gt_4 = Image.fromarray(np.uint8(renderings_gt[3]))
    renderings_gt_4_path = os.path.join(current_save_path, '%s_render_gt4.png' % name)
    renderings_gt_4.save(renderings_gt_4_path)

    output_path = os.path.join(current_save_path.split('\\')[-1], '%s_out.png' % name)
    img_path = os.path.join(current_save_path.split('\\')[-1], '%s_input.png' % name)
    sm_out_path = os.path.join(current_save_path.split('\\')[-1], '%s_sm_out.png' % name)
    hm_out_path = os.path.join(current_save_path.split('\\')[-1], '%s_hm_out.png' % name)
    in_out_path = os.path.join(current_save_path.split('\\')[-1], '%s_inout.png' % name)
    inl_out_path = os.path.join(current_save_path.split('\\')[-1], '%s_inlout.png' % name)
    inlr_out_path = os.path.join(current_save_path.split('\\')[-1], '%s_inlrout.png' % name)
    renderings_1_path = os.path.join(current_save_path.split('\\')[-1], '%s_render1.png' % name)
    renderings_2_path = os.path.join(current_save_path.split('\\')[-1], '%s_render2.png' % name)
    renderings_3_path = os.path.join(current_save_path.split('\\')[-1], '%s_render3.png' % name)
    renderings_4_path = os.path.join(current_save_path.split('\\')[-1], '%s_render4.png' % name)
    renderings_gt_1_path = os.path.join(current_save_path.split('\\')[-1], '%s_render_gt1.png' % name)
    renderings_gt_2_path = os.path.join(current_save_path.split('\\')[-1], '%s_render_gt2.png' % name)
    renderings_gt_3_path = os.path.join(current_save_path.split('\\')[-1], '%s_render_gt3.png' % name)
    renderings_gt_4_path = os.path.join(current_save_path.split('\\')[-1], '%s_render_gt4.png' % name)
    return [img_path, sm_out_path, hm_out_path, in_out_path, inl_out_path, inlr_out_path, output_path,
            renderings_1_path, renderings_2_path, renderings_3_path, renderings_4_path,
            renderings_gt_1_path, renderings_gt_2_path, renderings_gt_3_path, renderings_gt_4_path]


def write_html_teaser(root_save_path, file_paths, iteration):
    html_path = os.path.join(root_save_path, "real.html")
    if os.path.exists(html_path):
        index = open(html_path, "a")
    else:
        titles = ['Input/Normal_gt/Normal', 'I_n_out/Diffuse_gt/Diffuse', 'I_n_gt/Roughness_gt/Roughness',
                  'Diff_out/Specular_gt/Specular', 'Diff_gt//']
        index = open(html_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>Material Name</th>")
        for title in titles:
            index.write("<th>" + title + "</th>")
        index.write("</tr>\n")

    for file_path in file_paths:
        index.write("<tr>")
        index.write("<td>===========================</td>")
        index.write("<td>===========================</td>")
        index.write("<td>===========================</td>")
        index.write("<td>===========================</td>")
        index.write("<td>===========================</td>")
        index.write("</tr>")
        # 1st row
        index.write("<tr>")
        index.write("<td>%s</td>" % file_path[-1])
        index.write("<td><img src='" + file_path[0] + "'></td>")
        index.write("<td><img src='" + file_path[1] + "'></td>")
        index.write("<td><img src='" + file_path[2] + "'></td>")
        index.write("<td><img src='" + file_path[3] + "'></td>")
        index.write("<td><img src='" + file_path[4] + "'></td>")
        index.write("<td><img src='" + file_path[5] + "'></td>")
        index.write("</tr>")
        # 2nd row
        index.write("<tr>")
        index.write("<td>Iteration: {}____{}____{}____{}</td>".format(file_path[-5], file_path[-4], file_path[-3],
                                                                      file_path[-2]))
        index.write("<td><img src='" + file_path[7] + "'></td>")
        index.write("<td><img src='" + file_path[8] + "'></td>")
        index.write("<td><img src='" + file_path[9] + "'></td>")
        index.write("<td><img src='" + file_path[10] + "'></td>")
        index.write("<td><img src='" + file_path[6] + "'></td>")
        index.write("</tr>")
        # 3rd row
        index.write("<tr>")
        index.write("<td>Iteration: {}____{}____{}____{}</td>".format(file_path[-9], file_path[-8], file_path[-7],
                                                                      file_path[-6]))
        index.write("<td><img src='" + file_path[11] + "'></td>")
        index.write("<td><img src='" + file_path[12] + "'></td>")
        index.write("<td><img src='" + file_path[13] + "'></td>")
        index.write("<td><img src='" + file_path[14] + "'></td>")
        index.write("</tr>\n")

    index.write("<tr>")
    index.write("<td>=============</td>")
    index.write("<td>===========================</td>")
    index.write("<td>===========================</td>")
    index.write("<td>===========================</td>")
    index.write("<td>===========================</td>")
    index.write("</tr>\n")


def save_maps_teaser_real(outputs, sm_out, hm_out, inputs, in_out, inl_out, inlr_out, renderings, current_save_path,
                          name):
    outputs = np.clip(deprocess(outputs), 0, 1)
    inputs *= 255
    sm_out *= 255
    sm_out = np.concatenate([sm_out, sm_out, sm_out], axis=-1)
    hm_out *= 255
    hm_out = np.concatenate([hm_out, hm_out, hm_out], axis=-1)
    in_out = np.clip(in_out, 0, 1)
    in_out *= 255
    inl_out = np.clip(inl_out, 0, 1)
    inl_out *= 255
    inlr_out = np.clip(inlr_out, 0, 1)
    inlr_out *= 255

    renderings = np.power(renderings, 0.4545)
    renderings *= 255

    outputs[:, :, 3:6] = np.power(outputs[:, :, 3:6], 0.4545)
    outputs[:, :, 9:] = np.power(outputs[:, :, 9:], 0.4545)
    outputs *= 255
    outputs_tiled = np.concatenate([np.concatenate([outputs[:, :, :3], outputs[:, :, 3:6]], axis=1),
                                    np.concatenate([outputs[:, :, 6:9], outputs[:, :, 9:]], axis=1)], axis=0)
    outputs_tiled = Image.fromarray(np.uint8(outputs_tiled))
    outputs_tiled_path = os.path.join(current_save_path, '%s_out.png' % name)
    outputs_tiled.save(outputs_tiled_path)

    img = Image.fromarray(np.uint8(inputs))
    img_path = os.path.join(current_save_path, '%s_input.png' % name)
    img.save(img_path)
    sm_out = Image.fromarray(np.uint8(sm_out))
    sm_out_path = os.path.join(current_save_path, '%s_sm_out.png' % name)
    sm_out.save(sm_out_path)
    hm_out = Image.fromarray(np.uint8(hm_out))
    hm_out_path = os.path.join(current_save_path, '%s_hm_out.png' % name)
    hm_out.save(hm_out_path)
    in_out = Image.fromarray(np.uint8(in_out))
    in_out_path = os.path.join(current_save_path, '%s_inout.png' % name)
    in_out.save(in_out_path)
    inl_out = Image.fromarray(np.uint8(inl_out))
    inl_out_path = os.path.join(current_save_path, '%s_inlout.png' % name)
    inl_out.save(inl_out_path)
    inlr_out = Image.fromarray(np.uint8(inlr_out))
    inlr_out_path = os.path.join(current_save_path, '%s_inlrout.png' % name)
    inlr_out.save(inlr_out_path)

    renderings_1 = Image.fromarray(np.uint8(renderings[0]))
    renderings_1_path = os.path.join(current_save_path, '%s_render1.png' % name)
    renderings_1.save(renderings_1_path)
    renderings_2 = Image.fromarray(np.uint8(renderings[1]))
    renderings_2_path = os.path.join(current_save_path, '%s_render2.png' % name)
    renderings_2.save(renderings_2_path)
    renderings_3 = Image.fromarray(np.uint8(renderings[2]))
    renderings_3_path = os.path.join(current_save_path, '%s_render3.png' % name)
    renderings_3.save(renderings_3_path)
    renderings_4 = Image.fromarray(np.uint8(renderings[3]))
    renderings_4_path = os.path.join(current_save_path, '%s_render4.png' % name)
    renderings_4.save(renderings_4_path)

    output_path = os.path.join(current_save_path.split('\\')[-1], '%s_out.png' % name)
    img_path = os.path.join(current_save_path.split('\\')[-1], '%s_input.png' % name)
    sm_out_path = os.path.join(current_save_path.split('\\')[-1], '%s_sm_out.png' % name)
    hm_out_path = os.path.join(current_save_path.split('\\')[-1], '%s_hm_out.png' % name)
    in_out_path = os.path.join(current_save_path.split('\\')[-1], '%s_inout.png' % name)
    inl_out_path = os.path.join(current_save_path.split('\\')[-1], '%s_inlout.png' % name)
    inlr_out_path = os.path.join(current_save_path.split('\\')[-1], '%s_inlrout.png' % name)
    renderings_1_path = os.path.join(current_save_path.split('\\')[-1], '%s_render1.png' % name)
    renderings_2_path = os.path.join(current_save_path.split('\\')[-1], '%s_render2.png' % name)
    renderings_3_path = os.path.join(current_save_path.split('\\')[-1], '%s_render3.png' % name)
    renderings_4_path = os.path.join(current_save_path.split('\\')[-1], '%s_render4.png' % name)
    return [img_path, sm_out_path, hm_out_path, in_out_path, inl_out_path, inlr_out_path, output_path,
            renderings_1_path, renderings_2_path, renderings_3_path, renderings_4_path]


def write_html_teaser_real(root_save_path, file_paths, iteration):
    html_path = os.path.join(root_save_path, "real.html")
    if os.path.exists(html_path):
        index = open(html_path, "a")
    else:
        titles = ['Input/Normal_gt/Normal', 'I_n_out/Diffuse_gt/Diffuse', 'I_n_gt/Roughness_gt/Roughness',
                  'Diff_out/Specular_gt/Specular', 'Diff_gt//']
        index = open(html_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>Material Name</th>")
        for title in titles:
            index.write("<th>" + title + "</th>")
        index.write("</tr>\n")

    for file_path in file_paths:
        index.write("<tr>")
        index.write("<td>===========================</td>")
        index.write("<td>===========================</td>")
        index.write("<td>===========================</td>")
        index.write("<td>===========================</td>")
        index.write("<td>===========================</td>")
        index.write("</tr>")
        # 1st row
        index.write("<tr>")
        index.write("<td>%s</td>" % file_path[-1])
        index.write("<td><img src='" + file_path[0] + "'></td>")
        index.write("<td><img src='" + file_path[1] + "'></td>")
        index.write("<td><img src='" + file_path[2] + "'></td>")
        index.write("<td><img src='" + file_path[3] + "'></td>")
        index.write("<td><img src='" + file_path[4] + "'></td>")
        index.write("<td><img src='" + file_path[5] + "'></td>")
        index.write("</tr>")
        # 2nd row
        index.write("<tr>")
        index.write("<td>Iteration: {}</td>".format(iteration))
        index.write("<td><img src='" + file_path[7] + "'></td>")
        index.write("<td><img src='" + file_path[8] + "'></td>")
        index.write("<td><img src='" + file_path[9] + "'></td>")
        index.write("<td><img src='" + file_path[10] + "'></td>")
        index.write("<td><img src='" + file_path[6] + "'></td>")
        index.write("</tr>\n")

    index.write("<tr>")
    index.write("<td>=============</td>")
    index.write("<td>===========================</td>")
    index.write("<td>===========================</td>")
    index.write("<td>===========================</td>")
    index.write("<td>===========================</td>")
    index.write("</tr>\n")


# ================================================ teaser ==============================================================


def write_html(root_save_path, file_paths, iteration):
    html_path = os.path.join(root_save_path, "real.html")
    if os.path.exists(html_path):
        index = open(html_path, "a")
    else:
        titles = ['Input/Normal_gt/Normal', 'I_n_out/Diffuse_gt/Diffuse', 'I_n_gt/Roughness_gt/Roughness',
                  'Diff_out/Specular_gt/Specular', 'Diff_gt//']
        index = open(html_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>Material Name</th>")
        for title in titles:
            index.write("<th>" + title + "</th>")
        index.write("</tr>\n")

    for file_path in file_paths:
        index.write("<tr>")
        index.write("<td>===</td>")
        index.write("<td>===</td>")
        index.write("<td>===</td>")
        index.write("<td>===</td>")
        index.write("<td>===</td>")
        index.write("<td>===</td>")
        index.write("</tr>")
        # 1st row
        index.write("<tr>")
        index.write("<td>%s</td>" % file_path[-1])
        index.write("<td><img src='" + file_path[0] + "'></td>")
        index.write("<td><img src='" + file_path[1] + "'></td>")
        index.write("<td><img src='" + file_path[2] + "'></td>")
        index.write("<td><img src='" + file_path[3] + "'></td>")
        index.write("</tr>")
        # 2nd row
        index.write("<tr>")
        index.write("<td>Iteration: {}</td>".format(iteration))
        index.write("<td><img src='" + file_path[4] + "'></td>")
        index.write("<td><img src='" + file_path[5] + "'></td>")
        index.write("<td><img src='" + file_path[6] + "'></td>")
        index.write("<td><img src='" + file_path[7] + "'></td>")
        index.write("</tr>")
        # 3rd row
        index.write("<tr>")
        index.write("<td>Iteration: {}</td>".format(iteration))
        index.write("<td><img src='" + file_path[8] + "'></td>")
        index.write("<td><img src='" + file_path[9] + "'></td>")
        index.write("<td><img src='" + file_path[10] + "'></td>")
        index.write("<td><img src='" + file_path[11] + "'></td>")
        index.write("<td><img src='" + file_path[12] + "'></td>")
        index.write("</tr>\n")

    index.write("<tr>")
    index.write("<td>=============</td>")
    index.write("<td>===========================</td>")
    index.write("<td>===========================</td>")
    index.write("<td>===========================</td>")
    index.write("<td>===========================</td>")
    index.write("</tr>\n")


def write_html_real(root_save_path, file_paths, name="val.html"):
    html_path = os.path.join(root_save_path, name)
    if os.path.exists(html_path):
        index = open(html_path, "a")
    else:
        titles = ['Input/Normal_gt/Normal', 'I_n_out/Diffuse_gt/Diffuse', 'I_n_gt/Roughness_gt/Roughness',
                  'Diff_out/Specular_gt/Specular', 'Diff_gt//']
        index = open(html_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>Material Name</th>")
        for title in titles:
            index.write("<th>" + title + "</th>")
        index.write("</tr>\n")

    for file_path in file_paths:
        index.write("<tr>")
        index.write("<td>===</td>")
        index.write("<td>===</td>")
        index.write("<td>===</td>")
        index.write("<td>===</td>")
        index.write("<td>===</td>")
        index.write("<td>===</td>")
        index.write("</tr>")
        # 1st row
        index.write("<tr>")
        index.write("<td>%s</td>" % file_path[-1])
        index.write("<td><img src='" + file_path[4] + "'></td>")
        index.write("<td><img src='" + file_path[0] + "'></td>")
        index.write("<td><img src='" + file_path[1] + "'></td>")
        index.write("<td><img src='" + file_path[2] + "'></td>")
        index.write("<td><img src='" + file_path[3] + "'></td>")
        # index.write("<td><img src='" + file_path[5] + "'></td>")
        # index.write("<td><img src='" + file_path[6] + "'></td>")
        index.write("</tr>\n")

    index.write("<tr>")
    index.write("<td>=============</td>")
    index.write("<td>===========================</td>")
    index.write("<td>===========================</td>")
    index.write("<td>===========================</td>")
    index.write("<td>===========================</td>")
    index.write("</tr>\n")


def trunc_normal(tensor, mean=0., std=1., a=-2., b=2.):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


# ===============================================transformation=========================================================
def rotate_resize_normal(normal, angle, scale):
    normal = transform.resize(normal, (math.ceil(normal.shape[0] * scale), math.ceil(normal.shape[1] * scale)))
    # rotate vector field
    angle_pi = angle * math.pi / 180.0
    normal = normal * 2 - 1
    r, g, b = normal[:, :, 0], normal[:, :, 1], normal[:, :, 2]
    rot_r = r * math.cos(angle_pi) + g * math.sin(angle_pi)
    rot_g = g * math.cos(angle_pi) - r * math.sin(angle_pi)
    rot_normal = np.concatenate([rot_r[:, :, np.newaxis], rot_g[:, :, np.newaxis], b[:, :, np.newaxis]], axis=2)
    rot_normal = (rot_normal + 1) / 2
    rot_normal = np.clip(rot_normal, 0, 1)
    # rotate normal map
    rot_normal = transform.rotate(rot_normal, -angle)
    return rot_normal


def rotate_resize_texture(texture, angle, scale):
    texture = transform.resize(texture, (math.ceil(texture.shape[0] * scale), math.ceil(texture.shape[1] * scale)))
    return transform.rotate(texture, -angle)


def get_bounding_square(angle, scale, size):
    angle_pi = angle * math.pi / 180.0
    length = size / scale * (abs(math.cos(angle_pi)) + abs(math.sin(angle_pi)))
    return math.ceil(length)


def bound_after_transformation(size, angle):
    angle_pi = angle * math.pi / 180.0
    valid_size = size / (abs(math.cos(angle_pi)) + abs(math.sin(angle_pi)))
    return math.ceil((size - valid_size) / 2), math.floor((size + valid_size) / 2)


# ======================================================================================================================


def create_folder(path, still_create=False):
    if not os.path.exists(path):
        os.mkdir(path)
    elif still_create:
        if '\\' in path:
            dir_root = path[: path.rfind('\\')]
        else:
            dir_root = '.'
        count = 1
        original_dir_name = path.split('\\')[-1]
        while True:
            dir_name = original_dir_name + '_%d' % count
            path = os.path.join(dir_root, dir_name)
            if os.path.exists(path):
                count += 1
            else:
                os.mkdir(path)
                break
    return path


# ======================================================================================================================


def calculate_rmse(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have same dimensions')

    # multiple images
    if img1.ndim == 4:
        temp = 0
        for i in range(len(img1)):
            temp += calculate_rmse(img1[i], img2[i])
        return temp

    num = (img1 - img2) ** 2
    denom = img2 ** 2 + 1.0e-2
    relative_mse = np.divide(num, denom)
    relative_mse_mean = 0.5 * np.mean(relative_mse)
    return relative_mse_mean


def calculate_root_mse(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have same dimensions')

    # multiple images
    if img1.ndim == 4:
        temp = 0
        for i in range(len(img1)):
            temp += calculate_root_mse(img1[i], img2[i])
        return temp

    num = (img1 - img2) ** 2
    num = np.sqrt(num.mean())
    return num


def calculate_l1(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have same dimensions')

    # multiple images
    if img1.ndim == 4:
        temp = 0
        for i in range(len(img1)):
            temp += calculate_l1(img1[i], img2[i])
        return temp

    num = np.mean(np.abs(img1 - img2))
    return num
