from FourierRepresentation import *
import numpy as np
import math
import matplotlib.pyplot as plt

def processSegmentedTrajectory(trajectory, numSegments, frequencies_per_segment):
    #returns numSegments long list of tuples of tuples ( (frequency_dictionary_x, slope_x, b_x, numPoints_x), (frequency_dictionary_y, slope_y, b_y, numPoints_y) ) needed for generating the approximation
    x_vals = [point[0] for point in trajectory]
    y_vals = [point[1] for point in trajectory]

    segment_parameters = []

    for i in range(numSegments):
        step_value = len(x_vals) // numSegments
        beginning_index = step_value * i
        ending_index = beginning_index + step_value

        x_vals_slice = x_vals[beginning_index: ending_index]
        y_vals_slice = y_vals[beginning_index: ending_index]

        augmented_data_x = rectify(x_vals)
        augmented_data_y  = rectify(y_vals)

        frequencies_x = postProcessFrequencyDictionary(findFourierRepresentation(preProcessAugmentedData(augmented_data_x[0]), k=1.0, L=frequencies_per_segment))
        frequencies_y = postProcessFrequencyDictionary(findFourierRepresentation(preProcessAugmentedData(augmented_data_y[0]), k=1.0, L=frequencies_per_segment))

        frequencies_x = determineOptimal(frequencies_x, augmented_data_x[1], augmented_data_x[2], len(x_vals), x_vals)
        frequencies_y = determineOptimal(frequencies_y, augmented_data_y[1], augmented_data_y[2], len(y_vals), y_vals)

        segment_parameters.append( ( ( (frequencies_x, augmented_data_x[1], augmented_data_x[2], len(x_vals)) , (frequencies_y, augmented_data_y[1], augmented_data_y[2], len(y_vals)) ), (sum(x_vals_slice)/len(x_vals_slice), sum(y_vals_slice)/len(y_vals_slice)) ) )

    return segment_parameters

def write_approximations_to_file(list_of_approximation_parameters_plus_centroid, trajectoryType, numSegmentsPerTrajectory):
    list_of_approximation_parameters = []
    for tuple in list_of_approximation_parameters_plus_centroid:
        list_of_approximation_parameters.append(tuple[0])

    name = trajectoryType + ".txt"
    outputFile = open(name, "w")
    numKeys = len(list_of_approximation_parameters[0][0][0].keys())
    parameterization_length = numKeys*2 + 7
    num_parameterizations = len(list_of_approximation_parameters) // numSegmentsPerTrajectory

    outputFile.write(str(num_parameterizations) + " " + str(numSegmentsPerTrajectory) + " " + str(parameterization_length) + "\n")

    for approximation_parameters_plus_centroid in list_of_approximation_parameters_plus_centroid:
        approximation_parameters = approximation_parameters_plus_centroid[0]
        x_centroid = approximation_parameters_plus_centroid[1][0]
        y_centroid = approximation_parameters_plus_centroid[1][1]

        numPoints = approximation_parameters[0][3]

        outputFile.write(str(x_centroid) + "\n")
        outputFile.write(str(y_centroid) + "\n")
        outputFile.write(str(numPoints) + "\n")
        outputFile.write(str(approximation_parameters[0][1]) + "\n")
        outputFile.write(str(approximation_parameters[1][1]) + "\n")
        outputFile.write(str(approximation_parameters[0][2]) + "\n")
        outputFile.write(str(approximation_parameters[1][2]) + "\n")

        for key in approximation_parameters[0][0]:
            outputFile.write(str(approximation_parameters[0][0][key]) + "\n")
            outputFile.write(str(approximation_parameters[1][0][key]) + "\n")

    outputFile.close()

if __name__ == "__main__":

    read_in_trajectories = pathGenerator.read_trajectories_from_file("random_path_400(20, 15, 15, 10, 10, 10, 10, 10, 0, 0).txt")

    #approximation_parameters = processTrajectory(read_in_trajectories[0])
    #processTrajectory is the main function for converting the trajectory to a Fourier plus linear parameterization
    #approximation_parameters is how we store / move around the approximation of the given trajectory

    #display_approximation_plus_original_trajectory(approximation_parameters, read_in_trajectories[0])
    #use display_approximation for just displaying the approximation generated from approximation_parameters
    numSegments = 14
    numFreqsPerSubTrajectory = 10

    list_of_approximation_parameters_plus_centroid = []

    count = 1

    for trajectory in read_in_trajectories:
        print(count)
        count += 1
        segment_approximation_parameters_plus_centroid = processSegmentedTrajectory(trajectory, numSegments, numFreqsPerSubTrajectory)
        for approximation in segment_approximation_parameters_plus_centroid:
            list_of_approximation_parameters_plus_centroid.append(approximation)
    write_approximations_to_file(list_of_approximation_parameters_plus_centroid, "random_path_400(20, 15, 15, 10, 10, 10, 10, 10, 0, 0)_sample_segmented_approximation_14_10", numSegmentsPerTrajectory=numSegments)
