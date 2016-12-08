
import sys
import math 
import numpy as np 
import scipy 
import matplotlib.pyplot as plt 
import wave 
import proto as pr



def combineHam(signal, hamWindow):
    # output =[]
    # for sig in signal:
    #     output.append(hamWindow * sig)

	return signal * hamWindow


def main():
	if len(sys.argv) < 2:
		raise Exception("no filename given")

	filename = sys.argv[1]
	speech = wave.open('/home/akashd/masters-thesis/bayesiandnn/bayesiandnn/results_plot/' + filename)


	# n_frames = speech.getnframes()
	# sampling_rate = speech.getframerate()
	# window_width = speech.getsampwidth()
	# print n_frames, sampling_rate, window_width

	sampling_rate = 16000
	filename = sys.argv[1]
	outfile = 'speech1'
	speechdata = open(filename, 'rb').read()
	header_info = speechdata[:1024]
	speech_frames_raw = speechdata[1024:]
	frames_data = np.frombuffer(speech_frames_raw, dtype='i2')
	# frames_data = speech_frames_samples[:, np.newaxis]


	n_frames = frames_data.shape[0]
	# frames_data = speech.readframes(n_frames)

	# frames_data = np.load("speech1.npz")
	print frames_data
	# print  type(frames_data)

	fs = 1. /sampling_rate

	winlen_time = 20
	shift_time = 10

	winlen = int(winlen_time * 0.001 / fs)
	shift = int(shift_time * 0.001 / fs)

	enframes = pr.enframe(frames_data, winlen, shift)
	filterSignals = pr.preemp(enframes, 0.97)
	print filterSignals.shape

	''' hamming window '''
	hamWindow = pr.hamming(filterSignals.shape[1], False)
	signal_hamming = combineHam(filterSignals, hamWindow)

	nfft = 512;
	spec_fft, logspec_fft = pr.fft(signal_hamming, nfft);

	logspec_fft = logspec_fft[:,:256]
	logspec_fft = logspec_fft.T
	# logspec_fft = np.flipud(logspec_fft)


	print logspec_fft.shape

	fig3 = plt.figure()
	plt.imshow(spec_fft)
	fig3.savefig('fftspectrum.png')
	plt.close()

	fig4 = plt.figure()
	plt.imshow(logspec_fft)

	plt.xlabel('Frames')
	plt.ylabel('Frequency Bands')
	plt.gca().invert_yaxis()
	# plt.show()
	# fig4.savefig('logfftspectrum-new.pdf')
	fig4.savefig('logfftspectrum-new.png')








if __name__ == '__main__':
	print 'hi'
	main()