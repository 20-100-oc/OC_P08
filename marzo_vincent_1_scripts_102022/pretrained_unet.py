#from types import MethodType
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from .pretrained_encoders import get_vgg_encoder
from .pretrained_encoders import get_resnet50_encoder




IMAGE_ORDERING = 'channels_last'

if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = 3




def _unet(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608, channels=3):

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f4
    
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(32, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING, name="seg_feats"))(o)
    o = (BatchNormalization())(o)
    
    '''
    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)
    '''
    
    o = Conv2D(n_classes, (3, 3), padding='same', activation='softmax', 
               data_format=IMAGE_ORDERING)(o)
    
    output = Resizing(input_height, 
                      input_width, 
                      #interpolation="nearest"
                      )(o)
    
    model = Model(img_input, output)
    return model




def vgg_unet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3):

    model = _unet(n_classes, 
                  get_vgg_encoder,
                  input_height=input_height, 
                  input_width=input_width, 
                  channels=channels, 
                  )
    return model




def resnet50_unet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3):
    model = _unet(n_classes, 
                  get_resnet50_encoder,
                  input_height=input_height, 
                  input_width=input_width, 
                  channels=channels, 
                  )
    return model


