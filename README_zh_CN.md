### 什么是 iniclassification

这是用来学习和竞赛的一个图像分类框架，目标是基于图像分类任务以及 PyTorch 深度学习框架对图像分类的一些模块进行实验，对做科研的同学提供一个可供参考的分类任务的 pipeline。并尝试集成很多的 trick，希望能够对打算参加竞赛的同学也有一些帮助。

### iniclassification 框架有哪些特性

iniclassification 希望成为一个较为全能的框架，在优化器、学习率调度、网络结构等方面都提供较为丰富的选择。

- **Backbone**
  * [x] [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

- **Attention Module**
  * [ ] [xmu-xiaoma666/External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)

- **Loss**
  * [x] Softmax
  * [x] Cross Entropy Loss
  
- **[Validation metric](https://github.com/Media-Smart/volkscv/tree/master/volkscv/metrics/classification)**
  * [x] Accuracy
  * [ ] Precision 
  * [ ] Recall 
  * [ ] Area Under Curve
  * [ ] mean Average Precision
  * [ ] F1 score
  * [ ] Confusion Matrix

- **Parallel Training**
  * [ ] Data Parallel
  * [ ] Model Parallel

- **Automatic Mixed Precision**
  * [ ] [NVIDIA/apex](https://github.com/NVIDIA/apex)
  
- **Optimizer**
  * [x] [jettify/pytorch-optimizer](https://github.com/jettify/pytorch-optimizer)

- **LR_Scheduler**
  * [x] [torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

- **Data Augmentation**
  * [ ] [albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)

- **Distillation**
  * [ ] Knowledge Distillation
  
- **Bag of Tricks**
  * [x] [LR warmup](https://github.com/ildoonet/pytorch-gradual-warmup-lr)
  * [x] [Model ensemble](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch)
  * [x] [Test Time Augmentation](https://github.com/qubvel/ttach)
  * [ ] [LR finder](https://github.com/davidtvs/pytorch-lr-finder)
  * [ ] [Label smooth](https://github.com/open-mmlab/mmclassification/blob/84a939f858b746fe41a58b78480348ac2b705a98/mmcls/models/losses/label_smooth_loss.py)
  * [ ] [Antialiased CNNs](https://github.com/adobe/antialiased-cnns/)
  
- **Configuration Framework**
  * [ ] [~~facebookresearch/hydra~~](https://github.com/facebookresearch/hydra)
  * [x] [open-mmlab/mmcv](https://github.com/open-mmlab/mmcv)
  
- **AutoML Experiment**
  * [ ] [microsoft/nni](https://github.com/microsoft/nni)
  
### 如何快速上手

详情见 [GETTING_STARTED.md](https://github.com/inicv/iniclassification/tree/main/document/GETTING_STARTED.md).

### 联系作者

```markdown
yundoo99@gmail.com
```

### 致谢

感谢以下仓库的作者提供的优质代码，让我能够做一个缝合怪
- [open-mmlab/mmcv](https://github.com/open-mmlab/mmcv)
- [jettify/pytorch-optimizer](https://github.com/jettify/pytorch-optimizer)
- [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)


感谢以下文章提供的帮助

- http://giantpandacv.com/
- [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)