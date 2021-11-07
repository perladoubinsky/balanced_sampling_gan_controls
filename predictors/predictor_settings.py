# python 3.7
"""Contains basic configurations for predictors used in this project.

Please download the public released models and annotations from the following
repositories OR train your own predictor.

places365: https://github.com/CSAILVision/places365

NOTE: Any new predictor should be registered in `PREDICTOR_POOL` before used.
"""

import os.path

# pylint: disable=line-too-long
PREDICTOR_POOL = {    
    'celeba': {
        'weight_path' : 'predictors/pretrain/celeba.pth.tar',
        'resolution': None,  # Image resize will be done automatically.
        'image_channels': 3,
        'channel_order': 'RGB',
        #... 
    },

}
# pylint: enable=line-too-long

# Settings for model running.
USE_CUDA = True

MAX_IMAGES_ON_DEVICE = 64

MAX_IMAGES_ON_RAM = 12800