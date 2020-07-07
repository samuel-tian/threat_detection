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

def findFourierRepresentation(augmented_data, k=4.0, L=10):
    #gives first L coefficients of the Fourier series for augmented_data, with precision 1 / k
    #it is assumed that augmented_data[0] = 0, so the nth coefficient always refers to the sin(n*2*pi*x) term
    #returns a dictionary mapping select frequencies to their relative intensities, adds future flexibility if we only want to see intensities for select frequencies.

    # Number of sample points
    N = len(augmented_data)
    # sample spacing
    T = k / N #if in the format k / N, the precision to which we can see a frequency will be (1 / k) and the maximum frequency will be N // 2k

    #we can play around with these if we wanted to experiment with more precise frequency values (i.e. 0.2 Hz)

    augmented_data_f = fft(augmented_data) #complex valued, needs some post processing for better clarity
    frequency_values = 2.0/N * np.abs(yf[0:N//2]) #when T = (k / N) the jth value here corresponds to a (j / k) hertz sin wave

    #we can select which frequencies to output
    frequency_dictionary = {}
    for j in range(1, L + 1):
        frequency_dictionary[(j / k)] = frequency_values[j]

    return frequency_dictionary



"""
# Number of sample points
N = 1600 #can play with this value, the bigger it is, the steeper and higher the spikes but the spikes' location does not change
# sample spacing
T = 1.0 / 1600.0 #can play with this value, The maximum frequency you can detect is half the denominator


x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2) #don't play around with these values

plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()
print(2.0/N * np.abs(yf[0:N//2]))
print(len(2.0/N * np.abs(yf[0:N//2])))
"""

read_in_trajectories = read_trajectories_from_file("sampleTrajectory_0[].txt")
#x_vals = [point[0] for point in read_in_trajectories[0]]
y_vals = [point[1] for point in read_in_trajectories[0]]
augmented_data = rectify(y_vals)
augmented_vals = augmented_data[0]
time_vals = np.linspace(0.0, 1, len(y_vals))
plt.plot(time_vals, y_vals)
plt.plot(time_vals, augmented_vals)
plt.grid()
plt.show()
print(len(y_vals))
#display_trajectory(read_in_trajectories[0])
