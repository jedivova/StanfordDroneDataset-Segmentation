from torch import nn
from catalyst.contrib.nn import FocalLossBinary, DiceLoss, IoULoss, RAdam, Lookahead, OneCycleLRWithWarmup
from catalyst.dl import SupervisedRunner
from torch.optim.lr_scheduler import OneCycleLR
from model import segmentator
from dataset import get_loaders, CLASSES






if __name__=='__main__':
    model, preprocessing_fn = segmentator(ENCODER = 'timm-resnest14d', num_classes=5)

    logdir = "./logs"
    num_epochs = 40
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
    scheduler = OneCycleLR(
        optimizer,
        num_steps=num_epochs,
        max_lr=0.0016,
        steps_per_epoch=10,
        epochs=num_epochs
    )
    loaders = get_loaders(preprocessing_fn, batch_size=12)

    from catalyst.dl import DiceCallback, IouCallback, CriterionCallback, MetricAggregationCallback, ClasswiseIouCallback
    callbacks = [
        # Each criterion is calculated separately.
        CriterionCallback(input_key="mask", prefix="loss_dice", criterion_key="dice"),
        CriterionCallback(input_key="mask", prefix="loss_iou", criterion_key="iou"),
        CriterionCallback(input_key="mask", prefix="loss_focal", criterion_key="focal"),
        ClasswiseIouCallback(input_key="mask", prefix='clswise_iou',classes=CLASSES.keys()),

        # And only then we aggregate everything into one loss.
        MetricAggregationCallback(
            prefix="loss",
            mode="weighted_sum",  # can be "sum", "weighted_sum" or "mean"
            # because we want weighted sum, we need to add scale for each loss
            metrics={"loss_dice": 1, "loss_iou": 1, "loss_focal": 1},
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