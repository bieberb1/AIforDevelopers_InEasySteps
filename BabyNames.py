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

#Create list of names from csv file based on specs and order randomly
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
    random.shuffle(names) #put in random order
    return names

#Create encoding between character and numerical indices (n=30 with apostrophes, hyphens, periods, and spaces)
class Vocab:
    def __init__(self, names):
        #Get a list of all unique characters in names
        allNamesStr = ' '.join(names)
        #Convert to set, convert to sorted list
        self.vocab = sorted(set(allNamesStr))
        print ('vocab:', self.vocab)

        #Use internal attributes to convert to and from numerical indices
        self._char2idx = {c:i for i, c in enumerate(self.vocab)}
        self._idx2char = np.array(self.vocab)
        self.size = len(self.vocab)

    #Converts a character to an index
    def char2idx(self, c):
        return self._char2idx[c]

    #Converts an index to a character
    def idx2char(self, i):
        return self._idx2char[i]

#Convert string to a numpy array
def string2array(string, vocab, pad=None):
    if pad != None:
        string = string + ' ' * (pad - len(string))

    return np.array([vocab.char2idx(char) for char in string])

#Split names into training and testing sets - each batch must contain same number of sequences
def splitNames(names, mult, div):
    totalNames = len(names)
    testSplit = totalNames * mult // div

    trainingNames = names[:testSplit]
    testingNames = names[testSplit:-1]
    
    return trainingNames, testingNames

#Create batches of sequences from array - every batch must be same size
def batchUpSeq(array, seqLength, batchSize):
    batch=( [array[i:i+seqLength] \
                      for i in range(len(array)-seqLength) ] )
    over = len(batch) % batchSize
    batch = batch[:-over]
    print('data length: ', len(batch),
            'That is', len(batch) / batchSize, 'batches')
    
    return np.array(batch)

#Split names into training and testing sets, prepare batches
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

    #Expected output is the input sequence shifted by one character
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

#Define LSTM (long short term memory) model
def lstmModel(embeddingDimension, lstmUnits,
                        loss_fn, vocabSize, batchSize,
                        sequenceLength):

    model = tf.keras.Sequential( [
        #Encode each number as a vector of numbers
        tf.keras.layers.Embedding(vocabSize,
                             embeddingDimension),

        tf.keras.layers.LSTM(
                        lstmUnits, 
                        return_sequences=True, 
                        recurrent_initializer='glorot_uniform',
                        recurrent_activation='sigmoid'),

        #Output will be one-hot encoding - one ouput for each character in vocab
        tf.keras.layers.Dense(vocabSize, activation='relu') ] )

    #Compile the model with Adam optimizer and loss function
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model

#Train model as series of epochs
def trainModel(model, data, loss_fn):
    steps = len(data.trainingX) // batchSize
    #Initiate previousLoss
    previousLoss = float('inf')
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

        #Monitor how training is progressing
        print('Average test loss:', avgLoss, 'Improvement:',
               previousLoss - avgLoss)

        print(generateNames(model, data.vocab,
                batchSize, 300))

        #Halt if getting worse
        if avgLoss > previousLoss:
            print('Finished early to avoid over fitting')
            break
        previousLoss = avgLoss

def lstmProject():
    #Read names and convert to batched data
    names = readNames(firstYear = 2022, lastYear = 2023,
                                 genders='B')
    dataset = Names(names)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy( \
                                                          from_logits=True)

    #Create neural network 
    model = lstmModel(embeddingDimension = 100,
                        lstmUnits = 1000, loss_fn=loss_fn,
                        vocabSize = dataset.vocab.size,
                        batchSize = batchSize,
                        sequenceLength = sequenceLength)
    #Train neural network
    trainModel(model, dataset, loss_fn)

    #Generate new names
    genNames = generateNames(model, dataset.vocab,
                                             batchSize, 1000)
    #Last name will be truncated - just remove it
    genNames = genNames.split(' ')[:-1]

    #Report new names and accidental copies of old ones separately
    newNames = filter(lambda x: not x in names, genNames)
    oldNames = filter(lambda x: x in names, genNames)

    newNamesStr = ' '.join(newNames)
    oldNamesStr = ' '.join(oldNames)
    print('\nNew names generated:\n', newNamesStr)
    print('\nOld names generated:\n', oldNamesStr)

#Convert vectors back to characters to form names
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
