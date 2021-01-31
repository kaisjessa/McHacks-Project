import argparse
import os
import subprocess

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import ctc_utils
import cv2
import numpy as np
import crepe
import math
from scipy.io import wavfile
#import ctc_predict.py

sheetMusic = input('Enter the path to the sheet music: ')
audioFile = input('Enter the path to your audio file: ')

sr, audio = wavfile.read(audioFile)
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=False, step_size=100, verbose=0)

#print(list(frequency))
#print(list(confidence))

def frequency_to_note(freq):
	if(freq <= 0):
		return 0
	return(round(12*math.log(freq/440, 2)))

fqs = []
l = len(frequency)
for i in range(l):
	if(confidence[i] > 0.85):
		fqs.append(frequency_to_note(frequency[i]))

#print(fqs)

scale = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

notes = []

for fq in fqs:
	notes.append(scale[fq % 12] + str(int((fq+9)/12) + 4))



def remove_duplicates(arr):
	arr2 = []
	for i in range(len(arr)-1):
		if(arr[i] != arr[i+1]):
			arr2.append(arr[i])
	if(arr2[-1] != arr[-1]):
		arr2.append(arr[-1])
	return(arr2)


notes = remove_duplicates(notes)

voc_file = "Data/vocabulary_semantic.txt"
image = sheetMusic
model = "Models/semantic_model.meta"

tf.reset_default_graph()
sess = tf.InteractiveSession()

# Read the dictionary
dict_file = open(voc_file,'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
dict_file.close()

# Restore weights
saver = tf.train.import_meta_graph(model)
saver.restore(sess,model[:-5])

graph = tf.get_default_graph()

input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

image = cv2.imread(image,0)
image = ctc_utils.resize(image, HEIGHT)
image = ctc_utils.normalize(image)
image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]

prediction = sess.run(decoded,
                      feed_dict={
                          input: image,
                          seq_len: seq_lengths,
                          rnn_keep_prob: 1.0,
                      })

str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
sheetnotes = []
for w in str_predictions[0]:
    if int2word[w].startswith("note-"):
         sheetnotes.append(int2word[w])

print("You played ")
print(notes)
print("The written notes were ")
print(sheetnotes)
wrongNotes=[]

i = 0
for note in notes:

    if (i+1) > len(sheetnotes) or note not in sheetnotes[i]:
        wrongNotes.append("Issue with note " + str(i+1))
    i += 1
print(wrongNotes)

