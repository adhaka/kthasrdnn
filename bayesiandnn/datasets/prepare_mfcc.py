
# single point package to create mfccs
#  for both timit and tidigit


import cPickle, gzip
import os
import numpy as np 

from scipy import fftpack, signal
from scipy.signal import lfilter, hamming
from itertools import chain
import helper




def enframe(samples, winlen, winshift, padded=False):
    """Slices the input samples into overlapping windows.
    Args:
        winlen: window lenght in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """

    shift = (winlen - winshift)
    N_samples = samples.size
    N_frames = 1 + (N_samples - winlen) / shift

    if padded:
        # Attach as much zeros to cover other window
        sample = np.append(samples, np.zeros(winlen))
        output = np.zeros((N_frames + 1, winlen))

        for index in xrange(N_frames + 1):
            output[index] = sample[index*shift:index*shift + winlen]

    else:
        output = np.zeros((N_frames, winlen))

        for index in xrange(N_frames):
            output[index] = samples[index*shift:index*shift + winlen]

    return output



def preemp(input, p=0.97):
    """Pre-emphasis filter.
    Args:
        input: array of speech samples
        p: preemhasis factor (defaults to the value specified in the exercise)
    Output:
        output: array of filtered speech samples
    """
    b = np.array((1, -p))
    a = 1

    return lfilter(b, a, input, axis = 1)




def mfcc(samples, winlen, winshift, nfft, nceps, samplingrate):
    """Computes Mel Frequency Cepstrum Coefficients.
    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        nfft: length of the Fast Fourier Transform (power of 2, grater than winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
    Note: for convenienve, you can define defaults for the input arguments that fit the exercise
    Returns:
        ceps: N x nceps array with one MFCC feature vector per row
        mspec: N x M array of outputs of the Mel filterbank (of size M)
        spec: N x nfft array with squared absolute fast Fourier transform
    """

    enframes = enframe(samples, winlen, winshift)
    # preemp_signal = map(lambda x: preemp(x, 0.97), enframes)
    preemp_signal = preemp(enframes, p=0.97)
    hamWindow = hamming(winlen, False)
    ham_signal = helper.combineHam(preemp_signal, hamWindow)

    if not nfft:
        nfft = 512

    spec, logspec_fft = fft(ham_signal, nfft);

    bank1 = tools.trfbank(samplingrate, nfft);
    mspec = helper.melSpec(spec, bank1)
    spec_dct = helper.cosineTransform(mspec)
    ceps = spec_dct[:, :nceps]

    return (spec, mspec, ceps)

 