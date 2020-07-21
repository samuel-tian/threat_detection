import FourierRepresentation
import matplotlib.pyplot as plt

def readBucketizedTrajectories(filename):
    bucketFile = open(filename)
    firstFileLine = bucketFile.readline()
    numBuckets = int(firstFileLine)

    everything = []


    for i in range(numBuckets):
        firstBucketLine = bucketFile.readline()
        numTrajectories = int(firstBucketLine)
        numTrajectoriesRead = 0
        bucket_trajectories = []

        while numTrajectoriesRead < numTrajectories:

            trajectory_parameters = []
            numSegments = int(bucketFile.readline())
            numSegmentsRead = 0

            while numSegmentsRead < numSegments:
                segment_parameters = read_in_segment(bucketFile.readline(), numSegments)
                trajectory_parameters.append(segment_parameters)
                numSegmentsRead += 1

            numTrajectoriesRead += 1
            bucket_trajectories.append(trajectory_parameters)
        everything.append(bucket_trajectories)

    return everything

def read_in_segment(segment_string, numSegments):
    segment_tuple = eval(segment_string)

    x_centroid = segment_tuple[0]
    y_centroid = segment_tuple[1]
    numPointsPerSegment = segment_tuple[2] // numSegments
    slope_x = segment_tuple[3]
    slope_y = segment_tuple[4]
    intercept_x = segment_tuple[5]
    intercept_y = segment_tuple[6]

    frequency_dictionary_x = {}
    frequency_dictionary_y = {}

    for i in range(7, len(segment_tuple)):
        frequency_value = ((i-7) // 2)*0.5 + .5
        if (frequency_value - 7) % 2 == 0:
            frequency_dictionary_x[frequency_value] = segment_tuple[i]
        else:
            frequency_dictionary_y[frequency_value] = segment_tuple[i]

    return ((frequency_dictionary_x, slope_x, intercept_x, numPointsPerSegment), (frequency_dictionary_y, slope_y, intercept_y, numPointsPerSegment))

def displayBucketizedTrajectories(processedData):
    bucketCount = 1
    for bucket in processedData:
        print(bucketCount)
        displayBucket(bucket)
        bucketCount += 1

def displayBucket(bucketTrajectories):
    for segmented_trajectory in bucketTrajectories:
        (x, y) = generateSegmentedApproximation(segmented_trajectory)
        plt.plot(x, y, 'o', color='black')
    plt.show()

def generateSegmentedApproximation(segmented_trajectory):
    approximation_x = []
    approximation_y = []
    for segment in segmented_trajectory:
        segment_approximation_x = FourierRepresentation.generateApproximation(segment[0][0], segment[0][1], segment[0][2], segment[0][3])
        segment_approximation_y = FourierRepresentation.generateApproximation(segment[1][0], segment[1][1], segment[1][2], segment[1][3])
        for value in segment_approximation_x:
            approximation_x.append(value)
        for value in segment_approximation_y:
            approximation_y.append(value)
        
    return (approximation_x, approximation_y)

if __name__ == "__main__":
    file_path = "./classifiers/data/bucketized/circling_seg_bucketized.dat"
    #print(readBucketizedTrajectories(file_path))
    displayBucketizedTrajectories(readBucketizedTrajectories(file_path))
