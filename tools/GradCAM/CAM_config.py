# build_model
model = 'resnet18'
pretrained = False
num_classes = 20

model_path = '/home/muyun99/MyGithub/iniclassification/work_dirs/resnet18_b16x8_voc2012aug_tag_resnet_fp16_voc2012aug/models/best.pth'
image_path = '/home/muyun99/data/dataset/Public-Dataset/VOC2012Aug/JPEGImages/2007_000243.jpg'

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

class_id = 0

output_dir = '/home/muyun99/MyGithub/iniclassification/tools/GradCAM'

