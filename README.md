# Image_Segmentation
This project utilizes U-Net to construct a neural network for segmenting the liquid inside the pipes in the images. The liquid inside the pipe is divided into three parts: air, mixture, and water. The model takes a single-channel image of the pipe as input and outputs a three-channel mask image, where each channel corresponds to the three mentioned parts.

![Local Image](asset/demoImage.png)

![Local Image](asset/segentaiondemo.png)

# Model Structure
This model takes an input image of size 738x186x1 and outputs a segmentation image of size 738x186x3. The U-NET architecture consists of an Encoder and a Decoder. The Encoder is composed of multiple convolutional layers (represented by Conv2D), which convolve the image to extract different features. These features are then downsampled through pooling operations, reducing the dimensionality and extracting essential image features. By passing the image through multiple convolutional layers, we can transform the image into multiple low-dimensional feature vectors.

During the decoding process, we use deconvolution operations to restore these low-dimensional feature vectors back to high-dimensional image data. During decoding, each step uses the output before pooling from the corresponding encoding step as the condition for decoding, ensuring a strong correlation between the output and input images.

![Local Image](asset/UNet.png)

## Environment
* Python 3.9.16
* TensorFlow 2.6.0
* PyCharm 2022.3.2
* Anaconda 2.1.1
* CUDA v11.2

## Training Data
All training data is stored in the "Training_Image_Data" directory, which was provided by Arash Rabbani.

## Train Model
Simply run `main.py` to start training the model. At the program entry point, you can set the value of "load_weight" to indicate whether to load pre-trained weights. When "load_weight" is set to `False`, the program will not load pre-trained weights. On the other hand, when "load_weight" is set to `True`, it will load pre-trained weights. For the initial deployment, since there are no weight files available, the default setting should be set to `False`.

```python
if __name__ == '__main__':
    build_model(load_weight=False)
```

After training is completed, the program will automatically generate weight files in the root directory of the project. At the same time, a folder selection dialog will appear, where you need to choose the folder containing the image data you want to process. The generated files after processing will be stored in the newly generated folder in the root directory.

## Compute Average
After running average_height.py, a matrix file analyzing the percentage of each category in each row of pixels will be generated. The file is in MAT format and can be opened using MATLAB. This file is saved in a folder with the same name as the folder containing the data to be processed, with the suffix "_average" appended.

![Local Image](asset/UNet.png)
