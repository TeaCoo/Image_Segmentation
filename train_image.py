import os

import numpy as np
from PIL import Image


class ImageData:
    def __init__(self, file_path):
        self.name = os.path.basename(file_path)
        self.path = os.path.dirname(file_path)
        self.image = Image.open(file_path)
        self.width, self.height = self.image.size
        self.image = self.image.convert('L')
        self.data = np.reshape(list(self.image.getdata()), (self.height, self.width))


class ImageList:
    def __init__(self, images_path):
        self.images = []
        files = os.scandir(images_path)
        print("Loading image...")
        for file in files:
            self.images.append(ImageData(file.path))
        self.size = len(self.images)
        print("Images count:", self.size)

