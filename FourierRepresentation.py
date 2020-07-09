from numpy.fft import *
import numpy as np
import pathGenerator
import matplotlib.pyplot as plt

def Nmaxelements(list1, N):
    final_list = []

    for i in range(0, N):
        max1 = 0

        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j];

        list1.remove(max1);
        final_list.append(max1)

    return final_list

def preProcessAugmentedData(augmentedData):
    #takes augmented position values, essentially doubles it and compresses it so that the frequency is higher
    newData = []
    for i in range(len(augmentedData) - 1, -1, -1):
        newData.append(-1*augmentedData[i])
    for i in range(len(augmentedData)):
        newData.append(augmentedData[i])

    return newData

def postProcessFrequencyDictionary(frequency_dictionary):
    #must be run after the preProcessAugmentedData, basically divides all the frequencies by 2
    new_dict = {}
    for key in frequency_dictionary:
        new_dict[(key/2)] = frequency_dictionary[key]
    return new_dict

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


def findFourierRepresentation(augmented_data, k=1.0, L=40):
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
        #frequency_values[i] = frequency_values[i] * -1
        #print(frequency_values[i])

    #we can select which frequencies to output
    frequency_dictionary = {}
    for j in range(L + 1):
        frequency_dictionary[(j / k)] = frequency_values[j]
        #print(frequency_values[j])

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

def determineOptimal(frequency_dictionary, slope, b, numPoints, actualData, N=10):
    #calculates the approximation that minimizes the sum of the N, biggest pointwise differences between the original data
    approximation_1 = generateApproximation(frequency_dictionary, slope, b, numPoints)
    for key in frequency_dictionary:
        frequency_dictionary[key] = -1.0 * frequency_dictionary[key]
    approximation_2 = generateApproximation(frequency_dictionary, slope, b, numPoints)
    approximation_1_difference = []
    approximation_2_difference = []
    for i in range(len(actualData)):
        approximation_1_difference.append(np.abs(approximation_1[i] - actualData[i]))
        approximation_2_difference.append(np.abs(approximation_2[i] - actualData[i]))

    max_elts_1 = Nmaxelements(approximation_1_difference, N)
    max_elts_2 = Nmaxelements(approximation_2_difference, N)

    if sum(max_elts_1) > sum(max_elts_2):
        return approximation_2
    else:
        return approximation_1








if __name__ == "__main__":
    
    read_in_trajectories = pathGenerator.read_trajectories_from_file("sampleTrajectory_0[].txt")
    x_vals = [point[0] for point in read_in_trajectories[0]]
    y_vals = [point[1] for point in read_in_trajectories[0]]
    augmented_data_x = rectify(x_vals)
    augmented_data_y  = rectify(y_vals)

    frequencies_x = postProcessFrequencyDictionary(findFourierRepresentation(preProcessAugmentedData(augmented_data_x[0])))
    frequencies_y = postProcessFrequencyDictionary(findFourierRepresentation(preProcessAugmentedData(augmented_data_y[0])))

    #frequencies_x = findFourierRepresentation(augmented_data_x[0])
    #frequencies_y = findFourierRepresentation(augmented_data_y[0])


    approximation_x = determineOptimal(frequencies_x, augmented_data_x[1], augmented_data_x[2], len(x_vals), x_vals)
    approximation_y = determineOptimal(frequencies_y, augmented_data_y[1], augmented_data_y[2], len(y_vals), y_vals)

    display1D(approximation_x, x_vals)
    display1D(approximation_y, y_vals)

    parameterized_trajectory = []
    for i in range(len(approximation_x)):
        parameterized_trajectory.append((approximation_x[i], approximation_y[i]))



    plt.plot(x_vals, y_vals, 'o', color='black')
    plt.plot(approximation_x, approximation_y, 'o', color="orange")
    plt.show()


    #plt.plot(np.linspace(0, 1, len(preProcessAugmentedData(augmented_data_y[0]))),  preProcessAugmentedData(augmented_data_y[0])  )
    #print(findFourierRepresentation(preProcessAugmentedData(augmented_data_y[0])))
    #plt.show()



    #parameterized_trajectory = (approximation_x, approximation_y)
    #pathGenerator.display_trajectory(parameterized_trajectory)
