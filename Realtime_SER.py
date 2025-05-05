from emotion_recognition import EmotionRecognizer
import pyaudio
import os
import wave
from sys import byteorder
from array import array
from struct import pack
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
import time
from utils import get_best_estimators
from collections import Counter

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100
CHANNELS = 1
RECORD_SECONDS = 5
TEMP_WAV = "temp.wav"

SILENCE = 30

def preprocess_audio(filename):
    # Resample audio to 44.1 kHz
    sample_rate, audio_data = wavfile.read(filename)
    resampled_audio = resample(audio_data, int(audio_data.shape[0] * RATE / sample_rate))
    wavfile.write(TEMP_WAV, RATE, resampled_audio.astype(np.int16))

def is_silent(snd_data):
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    snd_data = _trim(snd_data)

    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

def record_audio(filename, duration):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)

    print("Recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK_SIZE * duration)):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)

    print("Finished recording!")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wavefile = wave.open(filename, 'wb')
    wavefile.setnchannels(CHANNELS)
    wavefile.setsampwidth(audio.get_sample_size(FORMAT))
    wavefile.setframerate(RATE)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()


def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}

def find_final_emotion(emotion_list):
    emotions_count = Counter(emotion_list)
    
    if len(emotions_count) == 1:
        return emotion_list[0]

    sorted_emotions = sorted(emotions_count.items(), key=lambda x: (-x[1], x[0]))
    
    final_emotion = sorted_emotions[0][0]
    
    if sorted_emotions[0][1] == sorted_emotions[1][1]:
        if 'happy' in emotions_count and 'neutral' in emotions_count:
            final_emotion = 'happy' if emotions_count['happy'] >= emotions_count['neutral'] else 'neutral'
        if len(sorted_emotions) > 2 and sorted_emotions[0][1] == sorted_emotions[2][1]:
            final_emotion = sorted_emotions[2][0]

    return final_emotion

if __name__ == "__main__":
    estimators = get_best_estimators(True)
    estimators_str, estimator_dict = get_estimators_name(estimators)
    print("Loading estimators: {}".format(estimators_str))

    features = ["mfcc", "chroma", "mel"]
    detector1 = EmotionRecognizer(estimator_dict["SVC"], emotions=["sad","neutral","happy","angry"], features=features, verbose=0)
    detector2 = EmotionRecognizer(estimator_dict["RandomForestClassifier"], emotions=["sad","neutral","happy","angry"], features=features, verbose=0)
    detector3 = EmotionRecognizer(estimator_dict["GradientBoostingClassifier"], emotions=["sad","neutral","happy","angry"], features=features, verbose=0)
    detector4 = EmotionRecognizer(estimator_dict["KNeighborsClassifier"], emotions=["sad","neutral","happy","angry"], features=features, verbose=0)
    detector5 = EmotionRecognizer(estimator_dict["MLPClassifier"], emotions=["sad","neutral","happy","angry"], features=features, verbose=0)
    detector6 = EmotionRecognizer(estimator_dict["BaggingClassifier"], emotions=["sad","neutral","happy","angry"], features=features, verbose=0)
    
    detector1.train()
    print("SVC Ready")
    detector2.train()
    print("RandomForestClassifier Ready")
    detector3.train()
    print("GradientBoostingClassifier Ready")
    detector4.train()
    print("KNeighborsClassifier Ready")
    detector5.train()
    print("MLPClassifier Ready")
    detector6.train()
    print("BaggingClassifier Ready")

    print("Test accuracy score SVC : {:.3f}%".format(detector4.test_score()*100))
    print("Test accuracy score RandomForestClassifier : {:.3f}%".format(detector2.test_score()*100))
    print("Test accuracy score GradientBoostingClassifier : {:.3f}%".format(detector1.test_score()*100))
    print("Test accuracy score KNeighborsClassifier : {:.3f}%".format(detector6.test_score()*100))
    print("Test accuracy score MLPClassifier : {:.3f}%".format(detector3.test_score()*100))
    print("Test accuracy score BaggingClassifier : {:.3f}%".format(detector5.test_score()*100))

    print("Please talk")
    
    while True:
        try:
            record_to_file("test.wav")
            results = [detector1.predict("test.wav"), detector2.predict("test.wav"), detector3.predict("test.wav"), detector4.predict("test.wav"), detector5.predict("test.wav"), detector6.predict("test.wav")]
            result = find_final_emotion(results)
            print(result, results)
        except KeyboardInterrupt:
            break