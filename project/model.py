import segmentation_models_pytorch as smp



def segmentator(ENCODER = 'timm-efficientnet-b3', ENCODER_WEIGHTS = 'imagenet', num_classes=5):
    '''
    Get the model and preprocesssing_fn
    :param ENCODER: name of encoder
    :param ENCODER_WEIGHTS:
    :return: model, preprocessing_fn
    '''

    model = smp.FPN(encoder_name=ENCODER,
                    encoder_weights=ENCODER_WEIGHTS,
                    classes=num_classes
                    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    return model, preprocessing_fn


def my_segmentator(num_classes=5):
    from Mobile_Unet import MobileUnet
    model = MobileUnet(num_classes=num_classes,
            width_multiplier=1.0,
            mode="small")
    preprocessing_fn = smp.encoders.get_preprocessing_fn('mobilenet_v2', 'imagenet')
    return model, preprocessing_fn