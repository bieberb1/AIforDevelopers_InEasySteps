import numpy as np
import os
import pathlib
import PIL
import PIL.Image
import PIL.ImageOps
import PIL.ImageChops
import PIL.ImageEnhance
import tensorflow as tf
import matplotlib.pyplot as plt

def doFolder(inputPath, outputPath):
    if inputPath.is_dir():
        for child in inputPath.iterdir():
            if child.is_file():
                ip = str(child)
                op = str(outputPath / child.name)
                doFile(ip, op)

def doFile(inputFile, outputFile):
    print('Doing file', inputFile, '-->', outputFile)

    with PIL.Image.open(inputFile) as image:
        plt.figure(figsize=(10, 10))

        image = convertMonochrome(image)
        image_array  = tf.keras.utils.img_to_array(image)
        image_tensor = tf.constant(image_array)
        showTensorOnPlot(7, image_tensor, 'raw image')

        image_tensor = cropWidth(cropHeight(image_tensor))
        showTensorOnPlot(8, image_tensor, 'cropped image')

        digits = splitDigits(image_tensor)

        for i in range(len(digits)):
            showTensorOnPlot(i+10, digits[i], 'digit '+str(i))
            digitImage = \
                     tf.keras.preprocessing.image.array_to_img(
                                                                     digits[i])

            digitImage = PIL.ImageOps.pad(digitImage,
                                                   (20,20), color='black')
            digitImage = PIL.ImageOps.expand(digitImage,
                                                   border=4, fill='black')
            saveDigit(outputFile, i, digitImage)
            op = pathlib.Path(outputFile)
        plt.show( )

def convertMonochrome(image):
        image = PIL.ImageOps.grayscale(image)
        showPILOnPlot(1, image, 'grayscale')

        image = PIL.ImageOps.autocontrast(image, cutoff=50)
        showPILOnPlot(2, image, 'autocontrast')

        image = PIL.ImageEnhance.Brightness(image). \
                                                               enhance(1.7)
        showPILOnPlot(3, image, 'brightness')
        image = PIL.ImageEnhance.Contrast(image). \
                                                                enhance(20)
        showPILOnPlot(4, image, 'contrast')

        image = PIL.ImageChops.invert(image)
        showPILOnPlot(5, image, 'invert')
        thresh = 200
        fn = lambda x : 255 if x > thresh else 0
        image = image.convert('L').point(fn, mode='1')

        showPILOnPlot(6, image, '1 bit per pixel')

        return image

def splitDigits(tensor):
    column_totals = tf.math.reduce_sum(tensor, axis=0)
    print('image size =', len(column_totals))
    digits = []
    column = 0
    while column < len(column_totals) - 1:

        digit_start = searchForNonZero(column_totals,
                                                   column, 1, 1)
        digit_end = searchForZero(column_totals,
                                                   digit_start, 1, 5)

        if digit_start == digit_end:
            break

        print('found digit at', digit_start, digit_end)
        digits.append(cropHeight(
                                    tensor[ : , digit_start:digit_end]))

        column = digit_end

    return digits

def cropHeight(tensor):
        row_totals = tf.math.reduce_sum(tensor, axis=1)

        top = searchForNonZero(row_totals, 0, 1, blur=2)
        bottom = searchForNonZero(row_totals,
                                       len(row_totals) - 1, -1, blur=2)
        top = max(0, top - 2)
        bottom = min(len(row_totals), bottom + 2)

        return tensor[top:bottom]

def cropWidth(tensor):
        column_totals = tf.math.reduce_sum(tensor, axis=0)
        left = searchForNonZero(column_totals, 0, 1, blur=2)
        right = searchForNonZero(column_totals,
                                   len(column_totals) - 1, -1, blur=2)

        left = max(0, left - 2)
        right = min(len(column_totals), right + 2)

        return tensor[ : , left:right]

def searchForNonZero(tensor, start, incr, blur):
    return searchForRun(tensor, start, incr, blur, zero=False)

def searchForZero(tensor, start, incr, blur):
    return searchForRun(tensor, start, incr, blur, zero=True)

def searchForRun(tensor, start, incr, blur, zero=False):
    run_length = 0
    run_start = start
    
    i = start
    while i >= 0 and i < len(tensor) and run_length <= blur:

        if (tensor[i] == 0) == zero:
            if run_length == 0:
                run_start = i
            run_length += 1

        else:
            run_length = 0

        i += incr

    return run_start

def showPILOnPlot(index, image, caption):
    if index < 17:
        plt.subplot(4, 4, index)
        plt.imshow(image, cmap='gray')
        plt.title(caption)
        plt.axis('off')

def showTensorOnPlot(index, image, caption):
    if index < 17:
        plt.subplot(4, 4, index)
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        plt.title(caption)
        plt.axis('off')

def saveDigit(outputFile, index, image):
            op = pathlib.Path(outputFile)
            digitName = op.stem +\
                               '_' + str(index) + \
                               op.suffix
            digitPath = op.parent / digitName

            image.save(digitPath)

if __name__ == '__main__':
    parent = pathlib.Path('images')
    inputPath = parent / 'input'
    outputPath = parent / 'output'
    doFolder(inputPath, outputPath)
