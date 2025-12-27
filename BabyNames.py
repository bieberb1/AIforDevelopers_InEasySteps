import os
import logging
import csv
import random
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
sequenceLength = 25
batchSize = 96
epochs=50

def readNames(firstYear, lastYear,
                     genders='GB', nameSize=99):
    names = set( )
    with open('babies-first-names-all-names-all-years.csv',
                   newline='') as csvfile:
        nameReader = csv.reader(csvfile,
                                           delimiter=',',
                                           quotechar='"')

        for row in nameReader:

            if len(row) > 0 and '0' <= row[0][0] <= '9' and \
                    int(row[0]) >= firstYear and \
                    int(row[0]) <= lastYear and \
                    row[1] in genders and \
                    len(row[2]) < nameSize:
                names.add(row[2].lower( ))

    names = list(names)
    random.shuffle(names)
    return names

class Vocab:
    def __init__(self, names):

        allNamesStr = ' '.join(names)
        self.vocab = sorted(set(allNamesStr))
        print ('vocab:', self.vocab)

        self._char2idx = {c:i for i, c in enumerate(self.vocab)}
        self._idx2char = np.array(self.vocab)
        self.size = len(self.vocab)

    def char2idx(self, c):
        return self._char2idx[c]

    def idx2char(self, i):
        return self._idx2char[i]

def string2array(string, vocab, pad=None):
    if pad != None:
        string = string + ' ' * (pad - len(string))

    return np.array([vocab.char2idx(char) for char in string])

def splitNames(names, mult, div):
    totalNames = len(names)
    testSplit = totalNames * mult // div

    trainingNames = names[:testSplit]
    testingNames = names[testSplit:-1]
    
    return trainingNames, testingNames

def batchUpSeq(array, seqLength, batchSize):
    batch=( [array[i:i+seqLength] \
                      for i in range(len(array)-seqLength) ] )
    over = len(batch) % batchSize
    batch = batch[:-over]
    print('data length: ', len(batch),
            'That is', len(batch) / batchSize, 'batches')
    
    return np.array(batch)

class Names:
    def __init__(self, names):
        self.vocab = Vocab(names)
        self.splitBatches(names)
        self.prepareBatches( )

    def splitBatches(self, names):
        self.trainingNames, self.testingNames = \
                                                splitNames(names, 4, 5)

        allNamesStr = ' '.join(names)
        self.trainingNamesStr = ' '.join(self.trainingNames)
        self.testingNamesStr = ' '.join(self.testingNames)

    def prepareBatches(self):
        trainingNamesXArray = string2array( \
                             self.trainingNamesStr[:-1], self.vocab)

        trainingNamesYArray = string2array( \
                              self.trainingNamesStr[1:], self.vocab)

        testingNamesXArray = string2array( \
                               self.testingNamesStr[:-1], self.vocab)

        testingNamesYArray = string2array( \
                               self.testingNamesStr[1:], self.vocab)

        self.trainingX = batchUpSeq(trainingNamesXArray,
                                            sequenceLength, batchSize)

        self.trainingY = batchUpSeq(trainingNamesYArray, 
                                            sequenceLength, batchSize)

        self.testingX = batchUpSeq(testingNamesXArray,
                                            sequenceLength, batchSize)

        self.testingY = batchUpSeq(testingNamesYArray,
                                           sequenceLength, batchSize)

def lstmModel(embeddingDimension, lstmUnits,
                        loss_fn, vocabSize, batchSize,
                        sequenceLength):

    model = tf.keras.Sequential( [

        tf.keras.layers.Embedding(vocabSize,
                             embeddingDimension),

        tf.keras.layers.LSTM(
                        lstmUnits, 
                        return_sequences=True, 
                        recurrent_initializer='glorot_uniform',
                        recurrent_activation='sigmoid'),

        tf.keras.layers.Dense(vocabSize, activation='relu') ] )

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model

def trainModel(model, data, loss_fn):
    steps = len(data.trainingX) // batchSize

    previousLoss = 9999

    for epoch in range(epochs):
        print ('Epoch', epoch)

        model.fit(data.trainingX, data.trainingY,
            steps_per_epoch=steps,
            batch_size=batchSize,
            epochs=1)

        predictedY = model.predict(data.testingX,
                                               batch_size=batchSize)

        losses = loss_fn(data.testingY, predictedY)
        avgLoss = losses.numpy().mean()

        print('Average test loss:', avgLoss, 'Improvement:',
               previousLoss - avgLoss)

        print(generateNames(model, data.vocab,
                batchSize, 300))

        if avgLoss > previousLoss:
            print('Finished early to avoid over fitting')
            break
        previousLoss = avgLoss

def lstmProject():
    names = readNames(firstYear = 1974, lastYear = 2000,
                                 genders='B')
    dataset = Names(names)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy( \
                                                          from_logits=True)

    model = lstmModel(embeddingDimension = 100,
                        lstmUnits = 1000, loss_fn=loss_fn,
                        vocabSize = dataset.vocab.size,
                        batchSize = batchSize,
                        sequenceLength = sequenceLength)
    trainModel(model, dataset, loss_fn)

    genNames = generateNames(model, dataset.vocab,
                                             batchSize, 1000)
    genNames = genNames.split(' ')[:-1]

    newNames = filter(lambda x: not x in names, genNames)
    oldNames = filter(lambda x: x in names, genNames)

    newNamesStr = ' '.join(newNames)
    oldNamesStr = ' '.join(oldNames)
    print('\nNew names generated:\n', newNamesStr)
    print('\nOld names generated:\n', oldNamesStr)

def generateNames(model, vocab, batchSize, length=1000):
    input_eval = [vocab.char2idx(' ')]
    input_eval = tf.expand_dims(input_eval, 0)
    input_eval = tf.repeat(input_eval, batchSize, axis=0)

    generated = []

    for i in range(length):
        predictions = model(input_eval)

        predictions = predictions[0]

        predicted_id = tf.random.categorical(predictions,
                                     num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        input_eval = tf.repeat(input_eval, batchSize, axis=0)

        generated.append(vocab.idx2char(predicted_id))

    return (''.join(generated))

if __name__ == '__main__':
    lstmProject( )
