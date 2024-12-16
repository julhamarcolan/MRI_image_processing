"""
This script processes 2D k-space data, typically used in MRI image reconstruction, by applying zero-filling to improve the resolution and 
performing a 2D Fast Fourier Transform (FFT) to shift the data from k-space to image space. 

The steps include:
1. Zero-filling the data for interpolation and increased resolution.
2. Applying a 2D FFT to convert the k-space data into image space.
3. Calculating and normalizing the magnitude of the resulting image for visualization.

The output is a grayscale magnitude image saved as a PNG file.

Author: Marolan, J.  
Date: October 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------
# Functions
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

def calculate_phase_image(data):
    """
    Calculates the phase image from a complex data array.
    
    Args:
        data (array): Complex data array representing the image.
        
    Returns:
        array: Array containing the phases of the complex components of each element of the input array.
    """
    
    phase_data = np.angle(data)

    return phase_data



try:
    #read the data 
    
    data = np.load('data.npy')

    #zero fill 
    #zero_fill_data = zero_fill_2Ddata(data_background, [1024,1024])

    # calculating the FFT 
    #zero_fill_fft = apply_FFT_2D(zero_fill_data)
    data_fft = apply_FFT_2D(data)


    #calculate the magnitude image 
    data_mag = calculate_magnitude_image(data_fft)

    plt.figure(figsize=(10,6))
    plt.imshow(np.abs(data_mag), cmap='gray')
    plt.title("Plane")
    plt.savefig("file_name.png")
    plt.show()


except FileNotFoundError:
    print("Error: 'data.npy' file not found.")
