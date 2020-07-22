import pathGenerator
import bucketVisualization
import SegmentedFourierRepresentation
import matplotlib.pyplot as plt

#orange is approximation, black is actual

read_in_trajectories = pathGenerator.read_trajectories_from_file("circling_400(20, 15, 15, 10, 10, 10, 10, 10, 0, 0).txt")
trajectory = read_in_trajectories[5] # change this number to look at different trajectories 


#pathGenerator.display_trajectory(trajectory)
x = []
y = []
for point in trajectory:
    if point[0] != "invisible":
        x.append(point[0])
        y.append(point[1])
plt.plot(x, y, 'o', color='black')



segmented_approximation_parameters_plus_centroid = SegmentedFourierRepresentation.processSegmentedTrajectory(trajectory, 14, 10)
segmented_approximation_parameters = []
for segment in segmented_approximation_parameters_plus_centroid:
    segmented_approximation_parameters.append(segment[0])

(x, y) = bucketVisualization.generateSegmentedApproximation(segmented_approximation_parameters)
plt.plot(x, y, 'o', color='orange')
plt.show()
