import os

sheetMusic = input('Enter the path to the sheet music: ')
audioFile = input('Enter the path to your audio file: ')

os.system("python ctc_predict.py -image " + sheetMusic + " -model Models/semantic_model.meta -vocabulary Data/vocabulary_semantic.txt -audio " + audioFile)
