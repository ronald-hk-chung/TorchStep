# Welcome to Self-Study-Torch
`sstorch` is a light-weight self-study library for PyTorch training tools and utilities

## Installing
To install with pip, use `pip -q install git+https://github.com/ronald-hk-chung/sstorch.git`

## About sstorch
Self-Study-Torch `sstorch` is currently a self-maintained light weight libarary that helps with developing training and evaluating pipline in analysing neural networks in PyTorch.

### Features
1. Simplicity - Less boiler-plate code than original Pytorch training and evaluation loop
2. Flexibility - Comes with `Callback` to ensure flexibility in adding custom functions/analysis while training
3. Built-in `Callback` for typical handling of gradients, hooks, schedulers etc
4. Extra built-in methods to fasten training including fit-one-cycle policy and LR finders

### Sample Learner and training results

Training Loop
```
class SSDTrainer(SSTLearner):
    def __init__(self, optim, train_dataloader, valid_dataloader):
        super().__init__(model=self.get_model(num_classes=3, size=300),
                         optim=optim,
                         loss_fn=self.ssd_loss,
                         metric_fn=self.ssd_metric,
                         train_dataloader=train_dataloader,
                         valid_dataloader=valid_dataloader)
    def get_model(self, num_classes=91, size=300):
        # Load the Torchvision pretrained model.
        model = torchvision.models.detection.ssd300_vgg16(
            weights=SSD300_VGG16_Weights.COCO_V1
        )
        # Retrieve the list of input channels.
        in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
        # List containing number of anchors based on aspect ratios.
        num_anchors = model.anchor_generator.num_anchors_per_location()
        # The classification head.
        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )
        # Image size for transforms.
        model.transform.min_size = (size,)
        model.transform.max_size = size
        return model

    def ssd_loss(self, losses):
        return sum(list(losses.values()))

    def ssd_metric(self, losses):
        return {k: v.cpu().detach().numpy() for k, v in losses.items()}

    def train_step(self):
        input, targets = self.batch
        losses = self.model(input, targets)
        loss = self.loss_fn(losses)
        metric = self.metric_fn(losses)
        return loss, metric

    def valid_step(self):
        input, targets = self.batch
        losses = self.model(input, targets)
        loss = self.loss_fn(losses)
        metric = self.metric_fn(losses)
        return loss, metric

    def set_valid_mode(self):
        self.model.train()
```

Results Display during training:
```
 Epoch: 1 | LR: 1.0E-04 | train_loss: 3.847 | valid_loss: 2.616 
train_metric: {'bbox_regression': 0.841, 'classification': 3.005}
valid_metric: {'bbox_regression': 0.68, 'classification': 1.935}
 Epoch: 2 | LR: 1.0E-04 | train_loss: 2.31 | valid_loss: 2.496 
train_metric: {'bbox_regression': 0.515, 'classification': 1.795}
valid_metric: {'bbox_regression': 0.677, 'classification': 1.819}
 Epoch: 3 | LR: 1.0E-04 | train_loss: 1.895 | valid_loss: 2.582 
train_metric: {'bbox_regression': 0.402, 'classification': 1.494}
valid_metric: {'bbox_regression': 0.74, 'classification': 1.842}
 Epoch: 4 | LR: 1.0E-04 | train_loss: 1.655 | valid_loss: 2.943 
train_metric: {'bbox_regression': 0.353, 'classification': 1.301}
valid_metric: {'bbox_regression': 0.69, 'classification': 2.253}
 Epoch: 5 | LR: 1.0E-04 | train_loss: 1.502 | valid_loss: 2.671 
train_metric: {'bbox_regression': 0.321, 'classification': 1.18}
valid_metric: {'bbox_regression': 0.722, 'classification': 1.949}
```

## Documentation