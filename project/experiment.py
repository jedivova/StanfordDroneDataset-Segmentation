from torch import nn
from catalyst.contrib.nn import FocalLossBinary, DiceLoss, IoULoss, RAdam, Lookahead, OneCycleLRWithWarmup
from catalyst.dl import SupervisedRunner
from model import segmentator
from dataset_UAVid import get_loaders






if __name__=='__main__':
    CLASSES = ['Clutter', 'Building', 'Road', 'Car', 'Tree', 'Vegetation', 'Human']
    model, preprocessing_fn = segmentator(ENCODER = 'timm-resnest14d')

    logdir = "./logs"
    num_epochs = 150
    learning_rate = 1e-3
    base_optimizer = RAdam([
        {'params': model.decoder.parameters(), 'lr': learning_rate},
        {'params': model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.segmentation_head.parameters(), 'lr': learning_rate},
    ], weight_decay=0.0003)
    optimizer = Lookahead(base_optimizer)
    criterion = {
        "dice": DiceLoss(),
        "iou": IoULoss(),
        "focal": FocalLossBinary()
    }
    runner = SupervisedRunner(device='cuda', input_key="image", input_target_key="mask")
    scheduler = OneCycleLRWithWarmup(
        optimizer,
        num_steps=num_epochs,
        lr_range=(0.0016, 0.0000001),
        init_lr = learning_rate,
        warmup_steps=5
    )
    loaders = get_loaders(preprocessing_fn, batch_size=12)

    from catalyst.dl import DiceCallback, IouCallback, CriterionCallback, MetricAggregationCallback, ClasswiseIouCallback
    callbacks = [
        # Each criterion is calculated separately.
        CriterionCallback(input_key="mask", prefix="loss_dice", criterion_key="dice"),
        CriterionCallback(input_key="mask", prefix="loss_iou", criterion_key="iou"),
        CriterionCallback(input_key="mask", prefix="loss_focal", criterion_key="focal"),
        ClasswiseIouCallback(input_key="mask", prefix='clswise_iou',classes=CLASSES),

        # And only then we aggregate everything into one loss.
        MetricAggregationCallback(
            prefix="loss",
            mode="weighted_sum",  # can be "sum", "weighted_sum" or "mean"
            # because we want weighted sum, we need to add scale for each loss
            metrics={"loss_dice": 0.5, "loss_iou": 0.5, "loss_focal": 1},
        ),

        # metrics
        DiceCallback(input_key="mask"),
        # IouCallback(input_key="mask"),

    ]

    model.train()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        # our dataloaders
        loaders=loaders,
        # We can specify the callbacks list for the experiment;
        callbacks=callbacks,
        # path to save logs
        logdir=logdir,
        num_epochs=num_epochs,
        # save our best checkpoint by Dice metric
        main_metric="dice",
        minimize_metric=False,
        fp16=dict(opt_level="O1"),
        verbose=True,
)