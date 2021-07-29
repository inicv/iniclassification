# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:53

@author: mick.yi

入口类

"""
from skimage import io
import argparse
from torch import nn
from tools import GradCAM, GradCamPlusPlus
from tools.torch_utils import *
from train import parse_args
from mmcv import Config, DictAction
import mmcv

from inicls import build_model, build_optimizer, build_loss, build_scheduler, build_dataset, build_dataloader
def get_net(cfg, weight_path=None):
    """
    根据网络名称获取模型
    :param net_name: 网络名称
    :param weight_path: 与训练权重路径
    :return:
    """

    model = build_model(cfg)
    if cfg.model_path is not None:
        model.load_state_dict(torch.load(cfg.model_path))
    return model


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_input(image):
    image = image.copy()

    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, cfg):
    prefix = os.path.splitext(os.path.basename(cfg.image_path))[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(cfg.output_dir, '{}-{}-{}.jpg'.format(prefix, cfg.model, key)), image)


def main(cfg):
    # 输入
    img = io.imread(cfg.image_path)
    img = np.float32(cv2.resize(img, (224, 224))) / 255


    inputs = prepare_input(img)
    # 输出图像
    image_dict = {}
    # 网络
    net = get_net(cfg)

    # Grad-CAM
    layer_name = get_last_conv_name(net)
    grad_cam = GradCAM(net, layer_name)

    mask = grad_cam(inputs, cfg.class_id)  # cam mask
    image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
    grad_cam.remove_handlers()

    # Grad-CAM++
    grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
    mask_plus_plus = grad_cam_plus_plus(inputs, cfg.class_id)  # cam mask
    image_dict['cam++'], image_dict['heatmap++'] = gen_cam(img, mask_plus_plus)
    grad_cam_plus_plus.remove_handlers()

    # # GuidedBackPropagation
    # gbp = GuidedBackPropagation(net)
    # inputs.grad.zero_()  # 梯度置零
    # grad = gbp(inputs)

    # gb = gen_gb(grad)
    # image_dict['gb'] = norm_image(gb)
    # # 生成Guided Grad-CAM
    # cam_gb = gb * mask[..., np.newaxis]
    # image_dict['cam_gb'] = norm_image(cam_gb)

    save_image(image_dict, cfg)


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.config = args.config
    cfg.tag = args.tag
    main(cfg)

