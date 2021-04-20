import segmentation_models_pytorch as smp

def segmentator(ENCODER = 'mobilenet_v2', ENCODER_WEIGHTS = 'imagenet'):
    '''
    Get the model and preprocesssing_fn
    :param ENCODER: name of encoder
    :param ENCODER_WEIGHTS:
    :return: model, preprocessing_fn
    '''

    model = smp.FPN(encoder_name=ENCODER,
                    encoder_weights=ENCODER_WEIGHTS,
                    classes=7
                    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    return model, preprocessing_fn

