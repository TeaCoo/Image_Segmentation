import segmentation as sg
import train_image as ti
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askdirectory

physic_devices = tf.config.list_physical_devices('GPU')
if len(physic_devices) > 0:
    tf.config.experimental.set_memory_growth(physic_devices[0], True)

train_data_path = os.path.join(os.path.dirname(__file__), 'Training_Image_Data')
acc_data = os.path.join(os.path.dirname(__file__), 'test_image')
test_data_path = os.path.join(os.path.dirname(__file__), 'processing_data')
output_path = os.path.join(os.path.dirname(__file__), 'output')


def get_train_data(path):
    image_list = ti.ImageList(path)
    train_data = []
    train_output = []

    for image in image_list.images:
        sub_image = np.hsplit(image.data, 2)
        train_data.append(sub_image[1]/255)

        air_mask = sub_image[0] == 0
        mixture_mask = sub_image[0] == 127
        water_mask = sub_image[0] == 255

        air = np.expand_dims(np.array(air_mask), axis=-1)
        mixture = np.expand_dims(np.array(mixture_mask), axis=-1)
        water = np.expand_dims(np.array(water_mask), axis=-1)

        output = np.concatenate((air, mixture, water), axis=-1)
        train_output.append(output)

    return np.expand_dims(np.array(train_data), axis=-1), np.array(train_output).astype(float), image_list


def get_test_data(path):
    image_list = ti.ImageList(path)
    test_data = []

    # for i in range(len(image_list.images)):
    #     image_list.images[i].data = image_list.images[i].data[1:len(image_list.images[i].data)-1]

    for index, image in enumerate(image_list.images):
        test_data.append(image.data/255)

    return np.expand_dims(np.array(test_data), axis=-1), image_list


def build_model(load_weight = False):
    load_weight = load_weight
    iteration = 35

    # height = 738, width = 166
    unet = sg.unet((738, 166, 1), 3)
    unet.summary()

    unet.compile(optimizer=Adam(learning_rate=0.001),
                 loss=BinaryCrossentropy(),
                 metrics=['accuracy'])

    if load_weight:
        unet.load_weights(os.path.join(os.path.dirname(__file__), "weights.h5"))
    else:
        train_data, train_output, _ = get_train_data(train_data_path)
        # acc_input, acc_output, _ = get_train_data(acc_data)
        reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5)
        history = unet.fit(train_data, train_output,
                           batch_size=3, epochs=iteration, callbacks=[reduce_lr])
        unet.save_weights(os.path.join(os.path.dirname(__file__), "weights.h5"))
        draw_plot(history)

    root = Tk()
    root.withdraw()
    data_folder = askdirectory()
    output_folder = os.path.join(data_folder, "..", os.path.split(data_folder)[-1]+"_output")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # for each folder in root
    items = os.listdir(data_folder)
    sub_folders = [item for item in items if os.path.isdir(os.path.join(data_folder, item))]
    for item in sub_folders:
        # load data
        print(os.path.join(data_folder, item))
        test_data, image_list = get_test_data(os.path.join(data_folder, item))
        # predict data
        predict_result = unet.predict(test_data, batch_size=4)
        # create a new folder with the same name in output folder
        output_folder_name = os.path.join(output_folder, item)
        if not os.path.exists(output_folder_name):
            os.mkdir(output_folder_name)
        print("Saving...")
        # store output into these folders
        for i in range(len(test_data)):
            save_result(output_folder_name, image_list.images[i].name, test_data[i], predict_result[i])


def show_image(array):
    plt.imshow(array, cmap='gray')
    plt.axis('off')
    plt.show()


def save_result(save_path, file_name, array, mask):
    # 1. save air
    result = mask2image(mask, 0)
    if not os.path.exists(os.path.join(save_path, "air_mask")):
        os.mkdir(os.path.join(save_path, "air_mask"))
    save_image(os.path.join(save_path, "air_mask"), file_name, result)

    # 2. save mixture
    result = mask2image(mask, 1)
    if not os.path.exists(os.path.join(save_path, "mixture_mask")):
        os.mkdir(os.path.join(save_path, "mixture_mask"))
    save_image(os.path.join(save_path, "mixture_mask"), file_name, result)

    # 3. save water
    result = mask2image(mask, 2)
    if not os.path.exists(os.path.join(save_path, "water_mask")):
        os.mkdir(os.path.join(save_path, "water_mask"))
    save_image(os.path.join(save_path, "water_mask"), file_name, result)

    # 4. save both
    result = combine_mask_image(array, mask, mode="both")
    if not os.path.exists(os.path.join(save_path, "combine_mask")):
        os.mkdir(os.path.join(save_path, "combine_mask"))
    save_image(os.path.join(save_path, "combine_mask"), file_name, result)


def save_image(save_path, file_name, result):
    result_uint8 = result.astype(np.uint8)
    image = Image.fromarray(result_uint8)
    image.save(os.path.join(save_path, file_name))


def mask2image(array, channel):
    rgb_image_image = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
    rgb_image_image[:, :, 0] = array[:, :, channel] * 255
    rgb_image_image[:, :, 1] = array[:, :, channel] * 255
    rgb_image_image[:, :, 2] = array[:, :, channel] * 255
    return rgb_image_image


def combine_mask_image(array, mask, mode="both"):
    mask_color = [[200, 50, 50],
                  [50, 200, 50],
                  [50, 50, 200],
                  ]

    array = np.reshape(array, (array.shape[0], array.shape[1]))

    mask_air = mask[..., 0:1]
    mask_mixture = mask[..., 1:2]
    mask_water = mask[..., 2:3]

    rgb_image_image = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
    # rgb_image_mask = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
    result = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)

    rgb_image_image[:, :, 0] = array * 255
    rgb_image_image[:, :, 1] = array * 255
    rgb_image_image[:, :, 2] = array * 255

    rgb_image_mask = []
    for a, b, c in zip(mask_air, mask_mixture, mask_water):
        rgb = a * mask_color[0] + b * mask_color[1] + c * mask_color[2]
        rgb_image_mask.append(rgb)

    rgb_image_mask = np.array(rgb_image_mask)

    if mode == "blender":
        weight = 0.6
        print(rgb_image_mask.shape)
        print(rgb_image_image.shape)
        result = weight * rgb_image_mask + (1 - weight) * rgb_image_image
        result = result.astype(int)
    elif mode == "both":
        result = np.concatenate((rgb_image_mask, rgb_image_image), axis=1)
        result = result.astype(int)
    elif mode == "output":
        result = rgb_image_mask
        result = result.astype(int)

    return result


def draw_plot(history):
    train_loss = history.history['loss']
    # val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, 'b', label='Training loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    build_model(load_weight=False)
