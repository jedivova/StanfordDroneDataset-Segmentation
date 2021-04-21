import segmentation_models_pytorch as smp

def segmentator(ENCODER = 'mobilenet_v2', ENCODER_WEIGHTS = 'imagenet', num_classes=5):
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

