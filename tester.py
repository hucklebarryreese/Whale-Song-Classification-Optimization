from scipy.io import wavfile
from scipy.ndimage import gaussian_filter
import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import pickle as pkl
import pylab
import conf
from util import map_range

### spectrogram is a function found in visualize.py, we use it in sgraph
### to generate the figure snippits
def spectrogram(stft, window_size, overlap, fs,
                y='linear', freq_subset: tuple = None, c_bar=None):

    hop_len = window_size * (1 - overlap)

    display.specshow(stft, y_axis=y,
                     sr=fs, hop_length=hop_len)

    if c_bar is str:
        plt.colorbar(format="%.2f "+"{}".format(c_bar))

    if freq_subset:
        hz_per_bin = (fs / 2) / (1 + window_size / 2)
        locs, labels = plt.yticks()
        c = hz_per_bin*math.floor(freq_subset[0]/hz_per_bin)
        d = hz_per_bin*math.ceil(freq_subset[1]/hz_per_bin)
        new_labels = ["%.2f" % map_range(locs[i], locs[0], locs[-1], c, d) for i in range(len(locs))]
        plt.yticks(locs, new_labels)

    return plt.gca()

### sgraph generates snippites of various spectrograms by using the start/end time
### of associated .txt files
def sgraph(start_time, end_time, naming_convention, model_path=conf.model_path, stft_path=conf.stft_path,
         target_file=conf.target_file, window_size=conf.window_size, overlap=conf.overlap,
         fs=conf.sample_rate, subset=conf.subset, words_per_doc=conf.words_per_doc):

    ## all the calculationns and transformations needed to prepare stft
    ## data structures for graphing
    theta = pd.read_csv(model_path + "theta.csv", header=None).values

    secs_per_frame = window_size * (1 - overlap) / fs

    secs_per_doc = secs_per_frame * words_per_doc
    start_doc = math.floor(start_time / secs_per_doc)
    end_doc = math.ceil(end_time / secs_per_doc)

    real_start_t = secs_per_doc * start_doc
    real_end_t = secs_per_doc * end_doc

    # subset of theta
    theta_sub = theta[start_doc: end_doc]

    start_frame = int(real_start_t * (1 / secs_per_frame))
    end_frame = int(real_end_t * (1 / secs_per_frame))

    stft = pkl.load(open(stft_path+target_file+'.pkl', "rb"))
    stft_sub = stft[start_frame:end_frame]
    stft_sub = np.matrix.transpose(stft_sub.values)

    ## generate the figure, eliminate the axis/border from it, and save it into
    ## a directory based on description name
    ## TODO: generate directories based on description and store there
    fig = plt.figure()
    pylab.ylim([0,7900]) 
    width = 3
    height = 3
    fig.set_size_inches(width, height)  
    plt.axis('off')
    spectrogram(stft_sub, window_size, overlap, fs,
                freq_subset=subset)
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0) 
    extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig("sslices/" + naming_convention + ".png", bbox_inches=extent)
