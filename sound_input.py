import crepe
from scipy.io import wavfile

sr, audio = wavfile.read('/path/to/audiofile.wav')
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)