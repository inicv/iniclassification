import os.path
from .base_dataset import BaseDataset
from .builder import DATASETS
from torch.cuda.amp import autocast
from tools.torch_utils import *

@DATASETS.register_module()
class competition_base_dataset(BaseDataset):
    def load_annotations(self):
        self.data_csv_path = os.path.join(self.data_prefix, self.ann_file)
        self.meta_txt = os.path.join(self.data_prefix, 'classmap.txt')
        self.df = pd.read_csv(self.data_csv_path)
        self.imgs = []
        self.gt_labels = []

        for index in range(len(self.df)):
            img = mmcv.imread(self.df['filename'].iloc[index], channel_order='rbg')
            self.imgs.append(img)

            if not self.test_mode:
                label = np.array(self.df['label'].iloc[index])
                self.gt_labels.append(label)

        self._load_meta()

        data_infos = []
        if not self.test_mode:
            for img, gt_label in zip(self.imgs, self.gt_labels):
                gt_label = np.array(gt_label, dtype=np.int64)
                info = {'img': img, 'gt_label': gt_label}
                data_infos.append(info)
        else:
            for img in zip(self.imgs):
                info = {'img': img}
                data_infos.append(info)

        return data_infos


    def _load_meta(self):
        with open(self.meta_txt, 'r') as meta_f:
            lines = meta_f.readlines()
            meta_info = []
            for cls in lines:
                meta_info.append(cls.strip())
            self.CLASSES = meta_info

    def evaluate(self, cfg, model, valid_dataloader):
        model.eval()

        data_len = 0
        total_acc = 0

        with torch.no_grad():
            for step, data in enumerate(valid_dataloader):
                images, labels = data['img'], data['gt_label']
                images, labels = images.cuda(), labels.cuda()
                data_len += len(labels)
                if cfg.fp16:
                    with autocast():
                        logits = model(images)
                else:
                    logits = model(images)

                _acc = accuracy(logits, labels)
                total_acc += _acc[0].item() * len(labels)

        model.train()
        return total_acc / data_len