import numpy as np
import scipy.io.wavfile as wav

def txt2list(filename):
    lines_list = []
    with open(filename, "r", encoding="utf-8") as txt:
        for line in txt:
            lines_list.append(line.rstrip("\n"))
    return lines_list

def get_random_noise(noise_files, size):
    noise_idx = np.random.choice(len(noise_files))
    _, noise_wav = wav.read(noise_files[noise_idx])
    offset = np.random.randint(len(noise_wav) - size)
    noise_wav = noise_wav[offset : offset + size].astype(float)
    return noise_wav

def split_wav(waveform, frame_size, split_hop_length):
    splitted_wav = []
    offset = 0
    while offset + frame_size < len(waveform):
        splitted_wav.append(waveform[offset : offset + frame_size])
        offset += split_hop_length
    return splitted_wav

def generate_random_silence_files(nb_files, noise_files, size, prefix, sr=16000):
    for i in range(nb_files):
        silence_wav = get_random_noise(noise_files, size)
        wav.write(prefix + "_" + str(i) + ".wav", sr, silence_wav)

