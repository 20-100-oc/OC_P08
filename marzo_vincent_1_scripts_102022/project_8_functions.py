from .pretrained_unet import vgg_unet, resnet50_unet

import os
import numpy as np
import tensorflow as tf
import albumentations as aug

from PIL import ImageOps
from IPython import display
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img




def get_imgs_path(dir, dataset):
    imgs_path = []
    for root, dirs, files in os.walk(dir + '/' + dataset):
        subset = [root + '/' + file for file in files if file.endswith('.png')]
        imgs_path += subset
    return sorted(imgs_path)




def get_masks_path(dir, dataset):
    masks_path = []
    for root, dirs, files in os.walk(dir + '/' + dataset):
        subset = [root + '/' + file for file in files if file.endswith('labelIds.png')]
        masks_path += subset
    return sorted(masks_path)




class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, imgs_path_list, model_type, img_to_mask, classes, categories, batch_size, img_size, n_augments, img_augments, n_channels, shuffle=False):
        self.model_type = model_type
        self.imgs_path_list = imgs_path_list
        self.img_to_mask = img_to_mask
        self.classes = classes
        self.categories = categories
        self.batch_size = batch_size
        self.batch_size_aug = batch_size*n_augments
        self.img_size = img_size
        self.n_augments = n_augments
        self.img_augments = img_augments
        self.n_channels = n_channels
        self.n_classes = len(classes)
        self.shuffle = shuffle
        
        self.mask_size = self.img_size
        '''
        if self.model_type == 'U-Net_vgg_pretrained':
            # there is one more pooling layer, so prediction is smaller
            self.mask_size = (int(self.img_size[0] / 2), 
                              int(self.img_size[1] / 2))
        '''
        self.on_epoch_end()
        self.transforms = aug.Compose(self.get_transform_list())
    

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.imgs_path_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    
    def get_transform_list(self):
        img_transforms = []
        
        for aug_type in self.img_augments:
            p = self.img_augments[aug_type]
            
            if aug_type == 'h_flip':
                # simple horizontal flip
                transform = aug.HorizontalFlip(p=p)
            
            elif aug_type == 'crop':
                # random crop of random ratio (from 0.5 to 0.9) and rescaled to original size
                height_limits = (int(0.5 * self.img_size[0]), int(0.9 * self.img_size[0]))
                transform = aug.augmentations.crops.transforms.RandomSizedCrop(p=p, 
                                                                          min_max_height=height_limits, 
                                                                          height=self.img_size[0], 
                                                                          width=self.img_size[1], 
                                                                          w2h_ratio=self.img_size[1] / self.img_size[0])
            
            elif aug_type == 'rotate':
                # rotation inside input frame by 5° max, 
                #OLD: borders are padded with black, considered as void
                #NEW: borders are padded with miror pixels
                transform = aug.augmentations.geometric.rotate.SafeRotate(p=p, 
                                                                       limit=5, 
                                                                       #border_mode=0, 
                                                                       #value=(0,0,0), 
                                                                       #mask_value=0, 
                                                                       )
            
            elif aug_type == 'blur':
                # gaussian blur
                transform = aug.augmentations.transforms.GaussianBlur(p=p)
            
            elif aug_type == 'brightness':
                transform = aug.RandomBrightnessContrast(p=p)
            
            img_transforms.append(transform)
        return img_transforms
    
    
    def load_process_img(self, img_path):
        img = img_to_array(load_img(img_path, target_size=self.img_size))
        img /= 255
        return img

    
    def load_process_mask(self, mask_path):
        mask = load_img(mask_path, target_size=self.img_size, color_mode='grayscale')
        mask = img_to_array(mask).squeeze()

        mask_cat = np.zeros((mask.shape[0], mask.shape[1], self.n_classes))
        for n, cat in enumerate(self.classes):
            for i in self.categories[cat]:
                mask_cat[:,:,n] = np.logical_or(mask_cat[:,:,n], mask==i)
        
        return mask_cat
    
    
    def augment_img_mask(self, img, mask):
        imgs_aug, masks_aug = [], []
        
        if self.n_augments == 0:
            imgs_aug, masks_aug = np.asarray(imgs_aug), np.asarray(masks_aug)
            imgs_aug = np.reshape(imgs_aug, (0, *self.img_size, self.n_channels))
            masks_aug = np.reshape(masks_aug, (0, *self.img_size, self.n_classes))
            return imgs_aug, masks_aug
        
        for i in range(self.n_augments):
            res = self.transforms(image=img, mask=mask)
            img_aug, mask_aug = res['image'], res['mask']
            imgs_aug.append(img_aug)
            masks_aug.append(mask_aug)
            
        return np.asarray(imgs_aug), np.asarray(masks_aug)
    

    def __data_generation(self, batch_imgs_path):
        X = np.empty((self.batch_size, *self.img_size, self.n_channels))
        X_aug = np.empty((self.batch_size_aug, *self.img_size, self.n_channels))
        y = np.empty((self.batch_size, *self.img_size, self.n_classes))
        y_aug = np.empty((self.batch_size_aug, *self.img_size, self.n_classes))

        for i, img_path in enumerate(batch_imgs_path):
            img = self.load_process_img(img_path)
            mask = self.load_process_mask(self.img_to_mask[img_path])
            X[i,:,:,:] = img
            y[i,:,:,:] = mask
            
            # make augmented data
            imgs_aug, masks_aug = self.augment_img_mask(img, mask)
            X_aug[i:i+len(imgs_aug),:,:,:] = imgs_aug
            y_aug[i:i+len(imgs_aug),:,:,:] = masks_aug
        
        # concat original and augmented data
        X_final = np.concatenate([X, X_aug], axis=0)
        y_final = np.concatenate([y, y_aug], axis=0)
        
        return X_final, y_final
    
    
    def __len__(self):
        '''
        Definition of this function is required (or there is a NotImplemented error).
        Returns the number of batches per epoch.
        '''
        n_samples = len(self.imgs_path_list)
        return int(np.floor(n_samples / self.batch_size))
    

    def __getitem__(self, index):
        '''Returns a batch (X and y)'''
        start_index = index * self.batch_size
        end_index = start_index + self.batch_size
        batch_indexes = self.indexes[start_index:end_index]

        batch_imgs_path = [self.imgs_path_list[i] for i in batch_indexes]

        X, y = self.__data_generation(batch_imgs_path)
        return X, y




def conv_block_PSPNet(X,filters,block):
    # resiudal block with dilated convolutions
    # add skip connection at last after doing convoluion operation to input X
    
    b = 'block_'+str(block)+'_'
    f1,f2,f3 = filters
    X_skip = X
    # block_a
    X = Convolution2D(filters=f1,kernel_size=(1,1),dilation_rate=(1,1),
                      padding='same',kernel_initializer='he_normal',name=b+'a')(X)
    X = BatchNormalization(name=b+'batch_norm_a')(X)
    X = LeakyReLU(alpha=0.2,name=b+'leakyrelu_a')(X)
    # block_b
    X = Convolution2D(filters=f2,kernel_size=(3,3),dilation_rate=(2,2),
                      padding='same',kernel_initializer='he_normal',name=b+'b')(X)
    X = BatchNormalization(name=b+'batch_norm_b')(X)
    X = LeakyReLU(alpha=0.2,name=b+'leakyrelu_b')(X)
    # block_c
    X = Convolution2D(filters=f3,kernel_size=(1,1),dilation_rate=(1,1),
                      padding='same',kernel_initializer='he_normal',name=b+'c')(X)
    X = BatchNormalization(name=b+'batch_norm_c')(X)
    # skip_conv
    X_skip = Convolution2D(filters=f3,kernel_size=(3,3),padding='same',name=b+'skip_conv')(X_skip)
    X_skip = BatchNormalization(name=b+'batch_norm_skip_conv')(X_skip)
    # block_c + skip_conv
    X = Add(name=b+'add')([X,X_skip])
    X = ReLU(name=b+'relu')(X)
    return X


def base_feature_maps_PSPNet(input_layer):
    # base covolution module to get input image feature maps 
    
    # block_1
    base = conv_block_PSPNet(input_layer,[32,32,64],'1')
    # block_2
    base = conv_block_PSPNet(base,[64,64,128],'2')
    # block_3
    base = conv_block_PSPNet(base,[128,128,256],'3')
    return base


def pyramid_feature_maps_PSPNet(input_layer):
    # pyramid pooling module
    
    base = base_feature_maps_PSPNet(input_layer)
    # red
    red = GlobalAveragePooling2D(name='red_pool')(base)
    red = tf.keras.layers.Reshape((1,1,256))(red)
    red = Convolution2D(filters=64,kernel_size=(1,1),name='red_1_by_1')(red)
    red = UpSampling2D(size=256,interpolation='bilinear',name='red_upsampling')(red)
    # yellow
    yellow = AveragePooling2D(pool_size=(2,2),name='yellow_pool')(base)
    yellow = Convolution2D(filters=64,kernel_size=(1,1),name='yellow_1_by_1')(yellow)
    yellow = UpSampling2D(size=2,interpolation='bilinear',name='yellow_upsampling')(yellow)
    # blue
    blue = AveragePooling2D(pool_size=(4,4),name='blue_pool')(base)
    blue = Convolution2D(filters=64,kernel_size=(1,1),name='blue_1_by_1')(blue)
    blue = UpSampling2D(size=4,interpolation='bilinear',name='blue_upsampling')(blue)
    # green
    green = AveragePooling2D(pool_size=(8,8),name='green_pool')(base)
    green = Convolution2D(filters=64,kernel_size=(1,1),name='green_1_by_1')(green)
    green = UpSampling2D(size=8,interpolation='bilinear',name='green_upsampling')(green)
    # base + red + yellow + blue + green
    return tf.keras.layers.concatenate([base,red,yellow,blue,green])


def last_conv_module_PSPNet(input_layer):
    X = pyramid_feature_maps_PSPNet(input_layer)
    X = Convolution2D(filters=3,kernel_size=3,padding='same',name='last_conv_3_by_3')(X)
    X = BatchNormalization(name='last_conv_3_by_3_batch_norm')(X)
    X = Activation('sigmoid',name='last_conv_relu')(X)
    X = tf.keras.layers.Flatten(name='last_conv_flatten')(X)
    return X




def get_model(model_type, img_size, n_channels, n_classes):
    
    if model_type == 'PSPNet':
        #TODO use above functions to make this available
        #TODO or check divamgupta's implementation 
        pass
    
    
    elif model_type == 'PSPNet_pretrained':
        #TODO
        pass
    
    
    elif model_type == 'U-Net_vgg_pretrained':
        model = vgg_unet(n_classes=n_classes, 
                         input_height=img_size[0], 
                         input_width=img_size[1])
    
    elif model_type == 'U-Net_resnet50_pretrained':
        model = resnet50_unet(n_classes=n_classes, 
                              input_height=img_size[0], 
                              input_width=img_size[1], 
                              )  
    
    
    
    elif model_type == 'U-Net':
        inputs = tf.keras.Input(shape=img_size + (n_channels,))
    
        ### [First half of the network: downsampling inputs] ###
    
        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    
        previous_block_activation = x  # Set aside residual
    
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
    
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
    
            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual
    
        ### [Second half of the network: upsampling inputs] ###
    
        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
    
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
    
            x = layers.UpSampling2D(2)(x)
    
            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual
    
        # Add a per-pixel classification layer
        outputs = layers.Conv2D(n_classes, 3, activation="softmax", padding="same")(x)
    
        # Define the model
        model = tf.keras.Model(inputs, outputs)
    
    return model




def get_mask_cat(mask_path, parameters):
    mask = load_img(mask_path, target_size=parameters['img_size'], color_mode='grayscale')
    mask = img_to_array(mask).squeeze()
    '''
    if parameters['model_type'] == 'U-Net_vgg_pretrained':
        mask = mask[::2,::2]
    '''
    mask_cat = np.zeros((mask.shape[0], mask.shape[1], len(parameters['classes'])))
    for n, cat in enumerate(parameters['classes']):
        for i in parameters['categories'][cat]:
            mask_cat[:,:,n] = np.logical_or(mask_cat[:,:,n], mask==i)
    mask_cat = np.argmax(mask_cat, axis=-1)
    
    return mask_cat




def get_iou(gt, pr, n_classes):
    epsilon = 1e-12   # to prevent dividing by zero in IoU

    class_wise = np.zeros(n_classes)
    n_pixels = np.zeros(n_classes)

    for cl in range(n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection)/(union + epsilon)

        class_wise[cl] = iou
        n_pixels[cl] = np.sum(cl==gt)
    
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    weighted_IoU = np.sum(class_wise * n_pixels_norm)
    mean_IoU = np.mean(class_wise)

    return class_wise, weighted_IoU, mean_IoU




def colorize_mask(mask, classes):
    colors = {
        'void': (0,0,0), 
        'flat': (128,128,128), 
        'construction': (100,66,0), 
        'object': (200,200,0), 
        'nature': (0,200,0), 
        'sky': (0,0,200), 
        'human': (200,0,0), 
        'vehicle': (255,255,255), 
        }
    
    color_mask = np.zeros(mask.shape, dtype='int8')
    for n, class_name in enumerate(classes):
        class_mask = np.where(mask==n, colors[class_name], 0)
        color_mask = color_mask + class_mask
        
    return color_mask




def show_pred(index, IoUs, imgs_list, masks_list, preds, parameters):
    # Display input image
    img = load_img(imgs_list[index], target_size=parameters['img_size'])
    display.display(img)
    print()

    # Display ground-truth mask (with contrasted labels)
    mask_cat = get_mask_cat(masks_list[index], parameters)
    mask_cat = np.expand_dims(mask_cat, axis=-1)
    mask_color = colorize_mask(mask_cat, parameters['classes'])
    display.display(array_to_img(mask_color))
    print()

    # Display prediction
    pred = np.expand_dims(preds[index], axis=-1)
    pred_color = colorize_mask(pred, parameters['classes'])
    display.display(array_to_img(pred_color))
    print()

    # display IoU score
    print('Mean IoU =', np.round(IoUs['mean'][index], 3), '\n')
    print('Weighted IoU =', np.round(IoUs['weighted'][index], 3), '\n')
    print('\nPer class:\n')
    for n in range(len(parameters['classes'])):
        IoU = np.round(IoUs['class_wise'][index,n], 3)
        print(f'({n})', f"{parameters['classes'][n]}".rjust(12), '=', IoU)




