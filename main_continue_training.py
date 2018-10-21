# encoding=utf-8
# Date: 2018-10-18
# Reference from: https://github.com/zhixuhao/unet
# Theory Reference: https://blog.csdn.net/u012931582/article/details/70215756
# Error Correction Reference: https://github.com/zhixuhao/unet/issues/45


from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans

from model import unet
from model import ModelCheckpoint

""" Description: 

    The combination of the values of the array represents the RGB in labeled datas
"""
labeled = [192, 128, 128]  # Note: This RGB array stands for the color pink
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([labeled, Unlabelled])


def adjustData(img, flag_multi_class, num_class, mask_path, gen_i, save_format):
    if (flag_multi_class):
        """ Note: 
            img.shape = <class 'tuple'>: (2, 256, 256, 3);    
            mask.shape = <class 'tuple'>: (2, 256, 256, 3)
        """
        img = img / 255

        mask_pic_path = mask_path + '/' + str(gen_i) + '.' + save_format
        mask_img = io.imread(mask_pic_path, as_gray=False)

        """ Attention:

                        The following scripts 'shape = (1, 256, 256, num_class)' because the batch_size set 1
                    """
        shape = (1, 256, 256, num_class)
        new_mask = np.zeros(shape, dtype=np.uint8)  # Note: (1, 256, 256, 12)

        for class_i in range(num_class):
            a = mask_img == COLOR_DICT[class_i]
            """ Attention:

                The following scripts 'for i in range(1)' because the batch_size set 1
            """
            for i in range(1):
                for j in range(256):
                    for k in range(256):
                        """ Attention:
                            Only [true, true, true] means that this pixel belongs to such class
                        """
                        if a[j][k][0]:
                            if a[j][k][1]:
                                if a[j][k][2]:
                                    new_mask[i][j][k][class_i] = 1

        """ Attention: There is a change, and the original is : new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2], new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))"""
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1], new_mask.shape[2],
                                         new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
        new_mask.shape[0], new_mask.shape[1], new_mask.shape[2]))
    else:
        print("Error: please check the codes")
        exit(1)

    return (img, new_mask)


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, train_data_num,
                   image_color_mode="rgb", mask_color_mode="rgb", image_save_prefix="image",
                   mask_save_prefix="mask",
                   flag_multi_class=True, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    ''' Description:

        Generate image and mask at the same time
        use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
        if you want to visualize the results of generator, set save_to_dir = "your path"

        Notes:
            batch_size = 2; train_path = "data/membrane/train"; image_folder = "image"; mask_folder = "label";
            aug_dict = {'shear_range': 0.05, 'height_shift_range': 0.05, 'horizontal_flip': True, 'fill_mode': 'nearest', 'rotation_range': 0.2, 'width_shift_range': 0.05, 'zoom_range': 0.05}
    '''

    image_datagen = ImageDataGenerator(
        **aug_dict)  # Note: <keras.preprocessing.image.ImageDataGenerator object at 0x000001B604B0E588>
    mask_datagen = ImageDataGenerator(
        **aug_dict)  # Note: <keras.preprocessing.image.ImageDataGenerator object at 0x000001B604B0E588>
    image_generator = image_datagen.flow_from_directory(
        train_path,  # Note: 'data/membrane/train'
        classes=[image_folder],  # Note: 'image'
        class_mode=None,  # Note: None
        color_mode=image_color_mode,  # Note: 'rgb'
        target_size=target_size,  # Note: <class 'tuple'>: (256, 256)
        batch_size=batch_size,  # Note: 2
        save_to_dir=save_to_dir,  # Note: None
        save_prefix=image_save_prefix,  # Note: 'image'
        seed=seed  # Note: 1; this parameter used as the random seed
    )

    mask_generator = mask_datagen.flow_from_directory(
        train_path,  # Note: 'data/membrane/train'
        classes=[mask_folder],  # Note: 'label'
        class_mode=None,  # Note: None
        color_mode=mask_color_mode,  # Note: 'rgb'
        target_size=target_size,  # Note: <class 'tuple'>: (256, 256)
        batch_size=batch_size,  # Note: 2
        save_to_dir=save_to_dir,  # Note: None
        save_prefix=mask_save_prefix,  # Note: 'mask'
        seed=seed,  # Note: 1; this parameter used as the random seed
        save_format='png'
    )

    mask_path = "./" + train_path + '/' + mask_folder
    gen_i = 0
    save_format = 'png'

    train_generator = zip(image_generator, mask_generator)  # Note: <zip object at 0x000001B604B10288>

    for (img, mask) in train_generator:
        img, mask = adjustData(img, flag_multi_class, num_class, mask_path, gen_i, save_format)
        gen_i = (gen_i + 1) % train_data_num
        yield (img, mask)


def testGenerator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=True, as_gray=False):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.jpg" % i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)

        yield img


"""
def geneTrainNpy(image_path, mask_path, flag_multi_class=True, num_class=2, image_prefix="image", mask_prefix="mask",
                 image_as_gray=False, mask_as_gray=False):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), as_gray=mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr
"""


def labelVisualize(num_class, color_dict, img):
    """

    :param num_class:
    :param color_dict:
    :param img: <class 'tuple'>: (256, 256, 2)
    :return:
    """
    img_2 = img[:, :, 0]  # Note: img_2.shape = (256, 256) script executed for the shape constructing of img_out
    img_out = np.zeros(img_2.shape + (3,), dtype=np.uint8)  # Note: (256, 256, 3)
    for class_i in range(num_class):
        for i in range(256):
            for j in range(256):
                if img[i][j][class_i] > 0.5:
                    img_out[i][j][0] = int(color_dict[class_i][0])
                    img_out[i][j][1] = int(color_dict[class_i][1])
                    img_out[i][j][2] = int(color_dict[class_i][2])

    """ Attention: The original codes is : img_out / 255"""
    return img_out


def saveResult(save_path, npyfile, flag_multi_class=True, num_class=2):
    for i, item in enumerate(npyfile):
        if flag_multi_class:
            img = labelVisualize(num_class, COLOR_DICT, item)
            io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)
        else:
            print("Error: please check the code !")
            exit(1)


# Note: {'shear_range': 0.05, 'height_shift_range': 0.05, 'horizontal_flip': True, 'fill_mode': 'nearest', 'rotation_range': 0.2, 'width_shift_range': 0.05, 'zoom_range': 0.05}
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
""" Description: 

    Important Operation Logic Note: 
        You can see the last line of the function "trainGenerator", it says: yield (img, mask) (in a 'for' Loop)
        That means this code structure is used to produce batches of training and labeled datas

    Besides, notice that the variable 'myGene' is utilized by the script below: model.fit_generator(..)
"""
train_data_num = 30
myGene = trainGenerator(1, 'data/membrane/train', 'image', 'label', data_gen_args, train_data_num, save_to_dir=None)

""" Attention: the following codes should be executed while hdf5 not existing

    >>>model = unet()
    while it is existing
    >>>model = unet(pretrained_weights="unet_membrane.hdf5")
"""
model = unet(pretrained_weights="unet_membrane.hdf5")
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1,
                                   save_best_only=True)  # Note: the first parameter of the function: filepath: string, path to save the model file.
model.fit_generator(myGene, steps_per_epoch=500, epochs=1, callbacks=[
    model_checkpoint])  # Note: myGnene: <generator object trainGenerator at 0x000001B67F9FD830>

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene, 30, verbose=1)  # Note: {ndarray} shape = <class 'tuple'>: (2, 256, 256, 12)
saveResult("data/membrane/test", results)