from numpy.fft import *
import numpy as np
from pathGenerator import *
import matplotlib.pyplot as plt

def rectify(data):
    #returns data minus the line that connects the two endpoints, gives the formula for the line
    #return value is (newData, slope, b)
    b = data[0]
    slope = (data[len(data) - 1] - data[0])/(len(data) - 1)
    newData = []
    for i in range(len(data)):
        newValue = data[i] - (b + slope*i)
        newData.append(newValue)
    return (newData, slope, b)

def findFourierRepresentation(augmented_data, k=1.0, L=20):
    #gives first L coefficients of the Fourier series for augmented_data, with precision 1 / k (i.e. if k = 4, we can determine relative intensities for 0.25 Hz, 0.5 Hz, 0.75 Hz, ..., 0.25*L Hz)
    #it is assumed that augmented_data[0] = 0, so the nth coefficient always refers to the sin(n*2*pi*x) term
    #returns a dictionary mapping select frequencies to their relative intensities, adds future flexibility if we only want to see intensities for select frequencies.

    # Number of sample points
    N = len(augmented_data)
    # sample spacing
    T = k / N #if in the format k / N, the precision to which we can see a frequency will be (1 / k) and the maximum frequency will be N // 2k

    #we can play around with these if we wanted to experiment with more precise frequency values (i.e. 0.2 Hz)

    augmented_data_f = fft(augmented_data) #complex valued, needs some post processing for better clarity
    frequency_values = 2.0/N * np.abs(augmented_data_f[0:N//2]) #when T = (k / N) the jth value here corresponds to a (j / k) hertz sin wave
    for i in range(len(frequency_values)):
        frequency_values[i] = np.sign(np.real(augmented_data_f[i])) * frequency_values[i] #enforces the sign of the sin wave
        frequency_values[i] = frequency_values[i] * -1
        #print(frequency_values[i])

    #we can select which frequencies to output
    frequency_dictionary = {}
    for j in range(L + 1):
        frequency_dictionary[(j / k)] = frequency_values[j]
        print(frequency_values[j])

    return frequency_dictionary

def generateApproximation(frequency_dictionary, slope, b, numPoints):
    #returns the desired Fourier-based approximating function based on the number of points in the original data, plus the linspace to plot it against
    #sin waves are evaluated on np.linspace(0.0, 1.0, numPoints)
    #final values of approximation should be plotted against np.linspace(0.0, numPoints - 1, numPoints)

    t =  np.linspace(0.0, 1.0, numPoints)
    approximation = 0*t
    for frequency in frequency_dictionary:
        approximation += frequency_dictionary[frequency] * np.sin(frequency * 2.0*np.pi*t)
    for i in range(numPoints):
        approximation[i] += (b + slope*i)

    return approximation

def display1D(approximationFunction, actualData):
    #plots input data for one dimension vs its approximation function
    time_vals = np.linspace(0.0, 1, len(actualData))
    plt.plot(time_vals, actualData)
    plt.plot(time_vals, approximationFunction)
    plt.grid()
    plt.show()




if __name__ == "__main__":
    read_in_trajectories = read_trajectories_from_file("sampleTrajectory_0[].txt")
    x_vals = [point[0] for point in read_in_trajectories[0]]
    #y_vals = [point[1] for point in read_in_trajectories[0]]
    augmented_data = rectify(x_vals)
    frequencies = findFourierRepresentation(augmented_data[0])
    approximation = generateApproximation(frequencies, augmented_data[1], augmented_data[2], len(x_vals))
    display1D(approximation, x_vals)
    for key in frequencies:
        frequencies[key] = -1.0 * frequencies[key]
    approximation = generateApproximation(frequencies, augmented_data[1], augmented_data[2], len(x_vals))
    display1D(approximation, x_vals)
