"""
This script performs Signal-to-Noise Ratio (SNR) calculations on Magnetic Resonance Imaging (MRI) images using various analysis methods based on the NEMA MS 1-2008 (R2014) standard.

The script includes the following functionalities:
1. 2D Fast Fourier Transform (FFT): Applies the FFT to 2D image data to convert the data into the frequency domain.
2. Zero Filling: Zero-fills the k-space data to improve spatial resolution.
3. Magnitude Image Calculation: Computes the magnitude image from complex data.
4. Extraction of the Central 75% of the Signal Image: Extracts the central portion corresponding to 75% of the total signal image area.
5. SNR Calculation Using the NEMA Method:
    - The calculation is done using two approaches: one based on background noise and another based on signal and noise images.
    - The SNR is calculated using regions of interest (MROI), separated by intensity thresholds.
6. Rayleigh Noise Correction: The function accounts for the fact that the magnitude image does not have a Gaussian distribution and applies a noise correction (dividing by 0.66).

Input Files:
- data_background.npy: Background data for SNR calculation based on background noise.
- data_signal.npy: Signal data for SNR calculation based on the signal image.
- data_noise.npy: Noise data for SNR calculation based on the noise image.

Exceptions:
- The script checks for the presence of required files and prints error messages if the files are not found.

Requirements:
- Required libraries: numpy, matplotlib

Author: Marolan, J.  
Date: October 2024
"""

import numpy as np
import matplotlib.pyplot as plt

def extract_central_75_percent(signal_mask, data_signal):
    """
    Extracts the central 75% of the signal from the data_signal using the signal_mask.
    
    This function identifies the central region of the signal mask that covers 
    approximately 75% of the total area, and uses this region to extract the corresponding 
    signal from the data_signal. The extracted signal and the new mask are returned.

    Parameters:
        signal_mask (np.ndarray - 2D): A boolean array where True represents the regions corresponding to the signal
    in the data_signal, and False represents non-signal (or noise) regions.
    
    data_signal ( np.ndarray  - 2D): The array containing the actual signal data, typically an image or 2D matrix.
    
    Returns:
        np.ndarray - 2D: The portion of data_signal corresponding to the central 75% region of the signal_mask.
    
        np.ndarray - 2D: A boolean array representing the mask for the central 75% region of the signal_mask.
        True values correspond to the central region, False values correspond to the rest.
    """
    # Get the shape of the signal mask
    mask_shape = signal_mask.shape
    
    # Determine the center of the mask
    center_y, center_x = mask_shape[0] // 2, mask_shape[1] // 2
    
    # Calculate the bounds of the 75% region
    y_size_75 = int(mask_shape[0] * 0.75)
    x_size_75 = int(mask_shape[1] * 0.75)
    
    # Compute the starting and ending indices for the central region
    start_y = max(0, center_y - y_size_75 // 2)
    end_y = min(mask_shape[0], center_y + y_size_75 // 2)
    start_x = max(0, center_x - x_size_75 // 2)
    end_x = min(mask_shape[1], center_x + x_size_75 // 2)
    
    # Create a new mask that represents the central 75%
    central_mask = np.zeros_like(signal_mask, dtype=bool)
    central_mask[start_y:end_y, start_x:end_x] = signal_mask[start_y:end_y, start_x:end_x]
    
    # Extract the central 75% signal from the data_signal using the new mask
    central_signal = data_signal * central_mask
    
    return central_signal, central_mask

def signal_and_noise_image(data_signal, data_noise, threshold): 
    """
    Calculates the SNR value for an image dataset according to the Method 2 procedure for evaluating image noise from NEMA MS 1-2008 (R2014).
    
    NEMA MS 1-2008 (R2014): 
        Signal: To calculate the signal value, the mean pixel value within the MROI is considered. 
        The MROI is calculated using a segmentation method that separates the image of the object from the noise according to a threshold and covers  75% of the signal.

        Noise: A noise scan image is acquired with the phantom in its original position and with no RF excitation.
    
    Parameters: 
        data_signal (np.ndarray - 2D): A 2D array representing the signal image, where the signal's intensity is to be 
        analyzed relative to the noise.

        data_noise (np.ndarray - 2D): A 2D array representing the noise image, used to calculate the noise component 
        in regions corresponding to the signal.

        threshold (float): A threshold value used to differentiate between signal and noise in the `data_signal`. Values greater than this threshold are considered signal, 
        and values less than or equal  to it are considered noise.
    
    Returns: 
        float:  The Signal-to-Noise Ratio (SNR) calculated as the ratio of the root mean square (RMS)  of the signal to the RMS of 
        the noise.

        float:  The SNR expressed in decibels (dB), calculated as 20 * log10(SNR).

    """
    # Create masks to identify signal (image) and noise regions in the signal image 
    signal_mask = data_signal > threshold
    noise_mask = data_signal <=  threshold

    # Extract the central 75% of the signal
    central_signal, central_mask = extract_central_75_percent(signal_mask, data_signal)
    
    # Calculate the signal and noise
    rms_signal = np.mean(central_signal[central_mask])
    rms_noise = np.std(data_noise[central_mask])/ 0.66

    snr = rms_signal / rms_noise

    # Calculate the SNR in dB
    snr_db = 20 * np.log10(snr)

    return  snr, snr_db

def background_noise(data, threshold, num_noise):
    """
    
    Calculates the SNR value for an image dataset according to the Method 4 procedure for evaluating image noise from NEMA MS 1-2008 (R2014).
    Note: 
    - The data must be zero-filled first to ensure that at least 1000 points can be considered in the noise MROI.
    - The MROI should cover at leas 75% of the image area.

    NEMA MS 1-2008 (R2014): 
        - Signal: To calculate the signal value, the mean pixel value within the MROI is considered. 
        The MROI is calculated using a segmentation method that separates the image of the object from the noise according to a threshold.
        
        - Noise: Draw a noise MROI in a background region of the image, well removed from the phantom and any visible artifacts, containing a 
        minimum of 1000 points. 
        In this case, we considered 4 squares from each corner of the FOV (each one with at least 250 points). To calculate the noise, we use 
        the standard deviation. Since a magnitude image is being evaluated, the image noise will not be Gaussian distributed but rectified to a 
        Rayleigh distribution. Therefore, a correction needs to be made, and the noise = SD / 0.66.
    
    Args:
        data (ndarray): The magnitude image data array.
        threshold (float): The threshold value to separate signal and noise.
        num_noise (int): The number of pixels to consider in each corner to calculate the noise.

    Returns:
        signal_mask (ndarray): Signal mask.
        noise_mask (ndarray): Noise mask.
        signal (ndarray): Signal image.
        noise (ndarray): Noise image.
        snr (float): Signal-to-noise ratio.
        snr_db (float): Signal-to-noise ratio in decibels.
    """
    # Create masks to identify signal (image) and noise regions
    signal_mask = data > threshold
    noise_mask = data <=  threshold

    noise = data * noise_mask

    central_signal, central_mask = extract_central_75_percent(signal_mask, data)
    
    # Extract submatrices from each corner to calculate noise
    submatrix1 = noise[:num_noise, :num_noise]        # Top-left corner
    submatrix2 = noise[:num_noise, -num_noise:]       # Top-right corner
    submatrix3 = noise[-num_noise:, :num_noise]       # Bottom-left corner
    submatrix4 = noise[-num_noise:, -num_noise:]      # Bottom-right corner

    # Concatenate horizontally the top and bottom submatrices
    top = np.hstack((submatrix1, submatrix2))
    bottom = np.hstack((submatrix3, submatrix4))

    concatenated_noise = np.vstack((top, bottom))

    # Calculate the signal and noise
    rms_signal = np.mean(central_signal[central_mask])
    rms_noise = np.std(concatenated_noise) / 0.66
    
    snr = rms_signal / rms_noise

    # Calculate the SNR in dB
    snr_db = 20 * np.log10(snr)

    return  snr, snr_db

def apply_FFT_2D(data):
    """
    Apply 2D Fast Fourier Transform (FFT) to input data.

    This function performs the following steps:
    1. Shifts the input data to center low-frequency components
    2. Performs FFT along the first axis (axis 0)
    3. Performs FFT along the second axis (axis 1)
    4. Shifts the transformed data back to the original position

    Args:
        data (array_like): Input data to which the FFT will be applied. Should be a 2D array.

    Returns:
        array_like: Transformed data after applying 2D FFT.

    """

    # Shift the data to center low-frequency components
    shifted_data = np.fft.fftshift(data, axes=(0, 1))

    # Perform FFT along the first axis (axis 0)
    transf_data = np.fft.fft(shifted_data, axis=0)

    # Perform FFT along the second axis (axis 1)
    transf_data = np.fft.fft(transf_data, axis=1)

    # Shift the transformed data back to the original position
    transf_data = np.fft.ifftshift(transf_data, axes=(0, 1))

    return transf_data

def zero_fill_2Ddata(offsetData, dim):
    """
    Zero-fills the k-space to reconstruct the data with increased resolution and interpolation with correct aspect ratio.
    
    Parameters:
        offsetData (ndarray): Input data in k-space.
        dim (tuple): Tuple representing the desired dimention of the zero fill data ..

    Returns:
        ndarray: Zero-filled data.
    """
    zeroFill = 2
    offsetData_dim = offsetData.shape  

    dim0Padding =  max(0, int((dim[0] - offsetData_dim[0]) / zeroFill))
    dim1Padding =  max(0, int((dim[1] - offsetData_dim[1]) / zeroFill))

    # Pad the data with zeros
    zeroFillData = np.pad(offsetData, [(dim0Padding, dim0Padding), (dim1Padding, dim1Padding)], mode='constant')

    return zeroFillData

def calculate_magnitude_image(data):
    """
    Calculates the magnitude image from complex-valued data.

    This function takes complex-valued image data and calculates the magnitude image 
    by computing the absolute value of the data and normalizing it by the maximum value 
    in the input data array.

    Args:
        data (numpy.ndarray): Complex-valued image data.

    Returns:
        numpy.ndarray: Magnitude image computed from the input data.
    """
    # Compute the magnitude image
    mag_data = np.abs(data)

    # Normalize the magnitude data
    mag_data /= np.max(mag_data)

    return mag_data

# Load data for background noise calculation
try:
    #read the data 
    data_background = np.load('data_background.npy')

    #zero fill 
    zero_fill_data = zero_fill_2Ddata(data_background, [1024,1024])

    # calculating the FFT 
    zero_fill_fft = apply_FFT_2D(zero_fill_data)

    #calculate the magnitude image 
    zero_fill_mag = calculate_magnitude_image(zero_fill_fft)

    #calculate the SNR 
    snr, snr_db = background_noise(zero_fill_mag, 0.1, 250)

    print(f"snr (background_noise): {snr}")
    print(f"snr_db (background_noise): {snr_db}")
except FileNotFoundError:
    print("Error: 'data_background.npy' file not found.")

# Load data for signal and noise image calculation
try:
    #read data
    data_signal = np.load('data_signal.npy')
    data_noise = np.load('data_noise.npy')

    #zero fill 
    zero_fill_data_signal= zero_fill_2Ddata(data_signal, [1024,1024])
    zero_fill_data_noise= zero_fill_2Ddata(data_noise, [1024,1024])

    # calculating the FFT 
    fft_signal = apply_FFT_2D(zero_fill_data_signal)
    fft_noise = apply_FFT_2D(zero_fill_data_noise)

    #calculate the magnitude image 
    imag_singal = calculate_magnitude_image(fft_signal)
    imag_noise = calculate_magnitude_image(fft_noise)

    #calculate the SNR 
    snr, snr_db = signal_and_noise_image(imag_singal, imag_noise, 0.1) 

    print(f"snr (ignal_and_noise_imag): {snr}")
    print(f"snr_db (ignal_and_noise_imag): {snr_db}")
except FileNotFoundError:
    print("Error: Signal or noise data file not found.")


