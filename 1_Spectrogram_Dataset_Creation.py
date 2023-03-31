import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io

# Define parameters for spectrogram computation
# sampling frequency of 42,000
fs = 42000
nperseg = 505 # segments for the image
noverlap = 504 # over lap per image used
segment_size = 512 # setting size of image

# Specify input and output directories
# Read the UODS_VAFDC folder README file for where to download the datasets
input_dir = 'UODS_VAFDC/1_Healthy/'
# change this name from 1_Healthy to 2_Inner_Race_Faults and the other fault types as you run this code
output_dir = 'UODS_spectrogram_datasets/1_Healthy/'

# Loop through all .mat files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.mat'):
        # Load variables from .mat file
        mat_contents = scipy.io.loadmat(os.path.join(input_dir, filename))
        # Load accelerometer data
        data = mat_contents[list(mat_contents.keys())[3]][:, 0]
        # Compute and save spectrogram for the first 200 segments of the data
        for i in range(0, segment_size*400, segment_size):
            segment = data[i:i+segment_size]
            f, t, Sxx = signal.stft(segment, nperseg=1024)
            
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(np.fliplr(abs(Sxx).T).T, cmap='viridis', aspect='auto', extent=[t.min(), t.max(), f.min(), f.max()])
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.axis('off')
            # dividing image to 512 for i / 512
            output_filename = os.path.join(output_dir, os.path.splitext(filename)[0] + '_{}.png'.format(int(i/512)))
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

# once the code finishes saving the datasets for one class it will print Complete!
print('Complete!')
