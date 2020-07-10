from numpy.fft import *
import numpy as np
import pathGenerator
import matplotlib.pyplot as plt
import math

def sigmoid(x):
    #take real number x to the output interval (0, 1)
  return 1 / (1 + math.exp(-x))

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
        #frequency_values[i] = frequency_values[i] * -1
        #print(frequency_values[i])

    #we can select which frequencies to output
    frequency_dictionary = {}
    for j in range(1, L + 1):
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
    negative_frequency_dictionary = {}
    for key in frequency_dictionary:
        negative_frequency_dictionary[key] = -1.0 * frequency_dictionary[key]
    approximation_2 = generateApproximation(negative_frequency_dictionary, slope, b, numPoints)
    approximation_1_difference = []
    approximation_2_difference = []
    for i in range(len(actualData)):
        approximation_1_difference.append(np.abs(approximation_1[i] - actualData[i]))
        approximation_2_difference.append(np.abs(approximation_2[i] - actualData[i]))

    max_elts_1 = Nmaxelements(approximation_1_difference, N)
    max_elts_2 = Nmaxelements(approximation_2_difference, N)

    if sum(max_elts_1) > sum(max_elts_2):
        return negative_frequency_dictionary
    else:
        return frequency_dictionary



def processTrajectory(trajectory):
    #returns a tuple of tuples ( (frequency_dictionary_x, slope_x, b_x, numPoints_x), (frequency_dictionary_y, slope_y, b_y, numPoints_y) ) needed for generating the approximation
    x_vals = [point[0] for point in trajectory]
    y_vals = [point[1] for point in trajectory]
    augmented_data_x = rectify(x_vals)
    augmented_data_y  = rectify(y_vals)

    frequencies_x = postProcessFrequencyDictionary(findFourierRepresentation(preProcessAugmentedData(augmented_data_x[0])))
    frequencies_y = postProcessFrequencyDictionary(findFourierRepresentation(preProcessAugmentedData(augmented_data_y[0])))

    frequencies_x = determineOptimal(frequencies_x, augmented_data_x[1], augmented_data_x[2], len(x_vals), x_vals)
    frequencies_y = determineOptimal(frequencies_y, augmented_data_y[1], augmented_data_y[2], len(y_vals), y_vals)

    #display1D(approximation_x, x_vals)
    #display1D(approximation_y, y_vals)

    return ( (frequencies_x, augmented_data_x[1], augmented_data_x[2], len(x_vals)) , (frequencies_y, augmented_data_y[1], augmented_data_y[2], len(y_vals)) )

def display_approximation(approximation_parameters):
    approximation_x = generateApproximation(approximation_parameters[0][0], approximation_parameters[0][1], approximation_parameters[0][2], approximation_parameters[0][3])

    approximation_y = generateApproximation(approximation_parameters[1][0], approximation_parameters[1][1], approximation_parameters[1][2], approximation_parameters[1][3])

    plt.plot(approximation_x, approximation_y, 'o', color="orange")
    plt.show()

def display_approximation_plus_original_trajectory(approximation_parameters, originalTrajectory):
    x = []
    y = []
    for point in originalTrajectory:
        if point[0] != "invisible":
            x.append(point[0])
            y.append(point[1])

    approximation_x = generateApproximation(approximation_parameters[0][0], approximation_parameters[0][1], approximation_parameters[0][2], approximation_parameters[0][3])

    approximation_y = generateApproximation(approximation_parameters[1][0], approximation_parameters[1][1], approximation_parameters[1][2], approximation_parameters[1][3])

    plt.plot(x, y, 'o', color='black')
    plt.plot(approximation_x, approximation_y, 'o', color="orange")
    plt.show()


def write_approximations_to_file(list_of_approximation_parameters, trajectoryType):
    name = trajectoryType + ".txt"
    outputFile = open(name, "w")
    numKeys = len(list_of_approximation_parameters[0][0][0].keys())
    parameterization_length = numKeys*2 + 5
    num_parameterizations = len(list_of_approximation_parameters)

    outputFile.write(str(num_parameterizations) + " " + str(parameterization_length) + "\n")

    for approximation_parameters in list_of_approximation_parameters:
        numPoints = approximation_parameters[0][3]
        outputFile.write(str(sigmoid(numPoints)) + "\n")
        outputFile.write(str(sigmoid(approximation_parameters[0][1])) + "\n")
        outputFile.write(str(sigmoid(approximation_parameters[1][1])) + "\n")
        outputFile.write(str(sigmoid(approximation_parameters[0][2])) + "\n")
        outputFile.write(str(sigmoid(approximation_parameters[1][2])) + "\n")

        for key in approximation_parameters[0][0]:
            outputFile.write(str(sigmoid(approximation_parameters[0][0][key])) + "\n")
            outputFile.write(str(sigmoid(approximation_parameters[1][0][key])) + "\n")

    outputFile.close()



if __name__ == "__main__":

    read_in_trajectories = pathGenerator.read_trajectories_from_file("sampleTrajectory_0[].txt")

    approximation_parameters = processTrajectory(read_in_trajectories[0])
    #processTrajectory is the main function for converting the trajectory to a Fourier plus linear parameterization
    #approximation_parameters is how we store / move around the approximation of the given trajectory

    display_approximation_plus_original_trajectory(approximation_parameters, read_in_trajectories[0])
    #use display_approximation for just displaying the approximation generated from approximation_parameters

    list_of_approximation_parameters = []
    list_of_approximation_parameters.append(approximation_parameters)

    write_approximations_to_file(list_of_approximation_parameters, "sample_approximation")
