import numpy as np
import os
import pathlib
import PIL
import PIL.Image
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf

def predict(model, filename):
    image = tf.keras.utils.load_img(filename,
                                               color_mode='grayscale')

    data = tf.keras.utils.img_to_array(image) / 255.0

    batch = np.array([data])
    prediction = model.predict(batch, verbose=0)

    return int(tf.math.argmax(prediction[0]))

if __name__ == '__main__':
    parent = pathlib.Path('images')
    inputPath = parent / 'output'

    files = [ ]
    if inputPath.is_dir( ):
        for child in inputPath.iterdir( ):
            if child.is_file( ):
                files.append(child)

    modelPath = 'mnist-model.keras'

    model = tf.keras.models.load_model(modelPath)
    
    softmaxModel = tf.keras.Sequential( [
        model,
        tf.keras.layers.Softmax( )
        ] )

    images = dict( )
    for x in files:
        stem = x.stem
        delimiter = stem.rfind('_')
        base = stem[ : delimiter]
        digitPos = int(stem[delimiter + 1 : ])

        if not base in images:
            images[base] = dict( )

        result = predict(softmaxModel, str(x))

        images[base][digitPos] = result

    total = 0
    for fileName, digits in images.items( ):

        i = 0
        number = 0
        while i in digits:
            number = number * 10 + digits[i]
            i += 1
        print (fileName, '=', number)

        total += number

    print ('Total:', total)
