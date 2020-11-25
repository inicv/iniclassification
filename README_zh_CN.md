### 什么是 iniclassification

这是用来学习和竞赛的一个图像分类框架，目标是基于图像分类任务以及 PyTorch 深度学习框架对图像分类的一些模块进行实验，对做科研的同学提供一个可供参考的分类任务的 pipeline。并尝试集成很多的 trick，希望能够对打算参加竞赛的同学也有一些帮助。

### iniclassification 框架有哪些特性

iniclassification 希望成为一个较为全能的框架，在优化器、学习率调度、网络结构等方面都提供较为丰富的选择。

- **Backbone**
    * [ ] ResNet(IR-SE)
    * [x] ResNeXt
    * [x] DenseNet
    * [ ] MobileFaceNet
    * [ ] MobileNetV3
    * [x] EfficientNet
    * [ ] ProxylessNas
    * [ ] AttentionNet-IRSE
    * [ ] SENet
    * [x] ResNeSt
    * [ ] ReXNet
    * [ ] RegNet
    * [ ] MobileNetV2
    * [ ] MobileNeXt
    * [ ] ShuffleNetV1
    * [ ] ShuffleNetV2
    * [ ] GhostNet

- **Attention Module**
    * [ ] SE
    * [ ] CBAM

- **Loss**
  * [x] Softmax
  * [x] Cross Entropy Loss

- **Parallel Training**
    * [ ] Data Parallel
    * [ ] Model Parallel

- **Automatic Mixed Precision**
  * [ ] Apex
  
- **Optimizer**
    * [x] [jettify/pytorch-optimizer](https://github.com/jettify/pytorch-optimizer)
- **LR_Scheduler**
    * [x] Cosine

- **Data Augmentation**

  * [ ] RandomErasing
  * [ ] Mixup
  * [ ] RandAugment
  * [ ] Cutout
  * [ ] CutMix
  * [ ] Colorjitter 

- **Distillation**

  * [ ] Knowledge Distillation
   
- **Bag of Tricks**

  * [x] LR warmup
  * [x] Model ensemble
  * [ ] LR finder
  * [ ] Label smooth
  
  
- **Configuration Framework**
 
   * [ ] [facebookresearch/hydra](https://github.com/facebookresearch/hydra)
   
- **AutoML Experiment**

   * [ ] [microsoft/nni](https://github.com/microsoft/nni)
   
### 如何快速上手

详情见 [GETTING_STARTED.md](https://github.com/inicv/iniclassification/tree/main/inicls/document/GETTING_STARTED.md).

### 联系作者

```markdown
yundoo99@gmail.com
```

### 致谢

感谢以下仓库的作者提供的帮助

- [jettify/pytorch-optimizer](https://github.com/jettify/pytorch-optimizer)

感谢以下网站的文章提供的帮助

- http://giantpandacv.com/