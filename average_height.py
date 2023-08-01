import os
from tkinter import Tk
from tkinter.filedialog import askdirectory
import numpy as np
import train_image as ti
import matplotlib.pyplot as plt
import scipy.io


def compute_average():
    root = Tk()
    root.withdraw()
    data_folder = askdirectory()
    output_folder = os.path.join(data_folder, "..", os.path.split(data_folder)[-1] + "_average")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # for each folder in root
    items = os.listdir(data_folder)
    sub_folders = [item for item in items if os.path.isdir(os.path.join(data_folder, item))]
    for item in sub_folders[::-1]:
        # for each folder

        exp_folder = os.path.join(data_folder, item)
        out_folder = os.path.join(output_folder, item)
        figure_path = os.path.join(out_folder, "figure")
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        if not os.path.exists(figure_path):
            os.mkdir(os.path.join(out_folder, "figure"))

        # get air mask image
        air_folders = os.path.join(exp_folder, "air_mask")
        air_images, air_image_list = get_images(air_folders)

        # get mixture mask image
        mixture_folders = os.path.join(exp_folder, "mixture_mask")
        mixture_images, mixture_image_list = get_images(mixture_folders)

        # get water mask image
        water_folders = os.path.join(exp_folder, "water_mask")
        water_images, water_image_list = get_images(water_folders)

        air_average = np.average(air_images, axis=2)
        mixture_average = np.average(mixture_images, axis=2)
        water_average = np.average(water_images, axis=2)

        print("Saving...")
        # save image
        """
        for i in range(len(air_average)):
            save_figure(figure_path, air_image_list.images[i].name,
                        air_average[i], mixture_average[i], water_average[i])
        """
        # save mat
        save_mat(out_folder, item, air_average, mixture_average, water_average)
        break


def get_images(path):
    image_list = ti.ImageList(path)
    test_data = []

    for index, image in enumerate(image_list.images):
        test_data.append(image.data/255)

    return np.expand_dims(np.array(test_data), axis=-1), image_list


def save_mat(save_path, file_name, air, mix, water):
    result = np.concatenate([air, mix, water], axis=-1)
    scipy.io.savemat(os.path.join(save_path, file_name + ".mat"), {'array': result})


def save_figure(save_path, file_name, air, mix, water):
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 10)
    # draw air
    x = np.arange(0, len(air))
    y = air
    y = np.flip(y, axis=0)
    ax.plot(y, x, color='r')

    # draw mixture
    x = np.arange(0, len(mix))
    y = mix
    y = np.flip(y, axis=0)
    ax.plot(y, x, color='g')

    # draw water
    x = np.arange(0, len(water))
    y = water
    y = np.flip(y, axis=0)
    ax.plot(y, x, color='b')

    fig.savefig(os.path.join(save_path, file_name))
    plt.close()


if __name__ == '__main__':
    compute_average()
