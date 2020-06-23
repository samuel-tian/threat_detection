#author: Raghav Samavedam
import random, math
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt

def sort_by_distance_to_end(current_position, displacement_vectors, ending_position):
    #sorts list of moves in increasing order of the distance to the ending position of the position gained from applying the move to the current position
    move_magnitude = []
    for move in displacement_vectors:
        position = (current_position[0] + move[0], current_position[1] + move[1])
        distance_vector = (ending_position[0] - position[0], ending_position[1] - position[1])
        distance_to_end = math.sqrt(distance_vector[0]**2 + distance_vector[1]**2)
        move_magnitude.append((move, distance_to_end))
    sorted_moves_plus_mag = sorted(move_magnitude, key=itemgetter(1))
    return [x[0] for x in sorted_moves_plus_mag]

def sort_by_distance_to_vector(target_vector, displacement_vectors):
    #sorts list of moves in increasing order of the distance to the target vector which characterizes circular motion about the origin
    target_vector = ((1/math.sqrt(target_vector[0]**2 + target_vector[1]**2))*target_vector[0], (1/math.sqrt(target_vector[0]**2 + target_vector[1]**2))*target_vector[1])
    move_magnitude = []
    for move in displacement_vectors:
        if math.sqrt(move[0]**2 + move[1]**2) == 0:
            normalized_displacement = move
        else:
            normalized_displacement = ((1/math.sqrt(move[0]**2 + move[1]**2))*move[0], (1/math.sqrt(move[0]**2 + move[1]**2))*move[1])
        distance_vector = (target_vector[0] - normalized_displacement[0], target_vector[1] - normalized_displacement[1])
        distance = math.sqrt(distance_vector[0]**2 + distance_vector[1]**2)
        move_magnitude.append((move, distance))
    sorted_moves_plus_mag = sorted(move_magnitude, key=itemgetter(1))
    return [x[0] for x in sorted_moves_plus_mag]

def generate_chase_points(scale, prob_distribution):
    #method objective:
    #return a list of points of a "chase boat" moving from one side of the grid towards the "victim boat" with a given erraticity

    #parameter descriptions:
    #grid across which the chase is happening will have dimension (2*scale) x (2*scale), scale must be > 0
    #the "victim" boat will always have position (0, 0) in this grid
    #prob_distribution is a list of ten non-negative integers, the first 9 sum up to 100, the last entry is less than 100
    #the 1st entry in prob_distribution refers to the probability the chasing boat moves to the grid position that is closest to the victim boat, the 2nd entry refers to the probability the chasing boat moves towards the 2nd closest grid position that is the 2nd closest to the victim boat and so on...
    #the last entry of prob_distribution refers to to the probability the chasing boat disappears from view

    #returned values:
    #list of points of length "scale"

    starting_side = random.choice(["up", "down", "left", "right"])
    if starting_side == "up":
        starting_pos = (random.choice([x for x in range(-1*scale, scale + 1)]), scale)
    if starting_side == "down":
        starting_pos = (random.choice([x for x in range(-1*scale, scale + 1)]), -1*scale)
    if starting_side == "left":
        starting_pos = (-1*scale, random.choice([x for x in range(-1*scale, scale + 1)]))
    if starting_side == "right":
        starting_pos = (scale, random.choice([x for x in range(-1*scale, scale + 1)]))

    list_of_positions = []
    list_of_positions.append(starting_pos)

    current_pos = [starting_pos[0], starting_pos[1]]

    possible_moves = [(0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1)] #displacement vectors

    #we now create the trajectory
    trajectory = [starting_pos]
    for i in range(scale):
        sorted_moves = sort_by_distance_to_end(current_pos, possible_moves, (0, 0))

        weighted_moves = [] #we randomly select our next move from here
        for i in range(len(sorted_moves)):
            for j in range(prob_distribution[i]):
                weighted_moves.append(sorted_moves[i])
        move = random.choice(weighted_moves)
        current_pos[0] += move[0]
        current_pos[1] += move[1]

        #now determine if the "chasing boat" is visible
        rand_number = random.random() #random number between 0 and 1
        if rand_number > (prob_distribution[9]/100):
            trajectory.append((current_pos[0], current_pos[1]))
        else:
            trajectory.append(("invisible", "invisible"))

    return trajectory

def generate_random_path_points(scale, prob_distribution):
    #method objective:
    #return a list of points of a boat moving from side to another of the grid with given erraticity

    #parameter descriptions:
    #grid across which the chase is happening will have dimension (2*scale) x (2*scale), scale must be > 0
    #the observing boat will always have position (0, 0) in this grid
    #prob_distribution is a list of ten non-negative integers, the first 9 sum up to 100, the last entry is less than 100
    #the 1st entry in prob_distribution refers to the probability the chasing boat moves to the grid position that is closest to its target destination, the 2nd entry refers to the probability the chasing boat moves towards the 2nd closest grid position that is the 2nd closest to its target destination and so on...
    #the last entry of prob_distribution refers to to the probability the moving boatdisappears from view

    #returned values:
    #list of points of length 2*"scale"

    starting_side = random.choice(["up", "down", "left", "right"])
    if starting_side == "up":
        starting_pos = (random.choice([x for x in range(-1*scale, scale + 1)]), scale)
    if starting_side == "down":
        starting_pos = (random.choice([x for x in range(-1*scale, scale + 1)]), -1*scale)
    if starting_side == "left":
        starting_pos = (-1*scale, random.choice([x for x in range(-1*scale, scale + 1)]))
    if starting_side == "right":
        starting_pos = (scale, random.choice([x for x in range(-1*scale, scale + 1)]))

    ending_side = random.choice(["up", "down", "left", "right"])
    if ending_side == "up":
        ending_pos = (random.choice([x for x in range(-1*scale, scale + 1)]), scale)
    if ending_side == "down":
        ending_pos = (random.choice([x for x in range(-1*scale, scale + 1)]), -1*scale)
    if ending_side == "left":
        ending_pos = (-1*scale, random.choice([x for x in range(-1*scale, scale + 1)]))
    if ending_side == "right":
        ending_pos = (scale, random.choice([x for x in range(-1*scale, scale + 1)]))

    list_of_positions = []
    list_of_positions.append(starting_pos)

    current_pos = [starting_pos[0], starting_pos[1]]

    possible_moves = [(0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1)] #displacement vectors

    #we now create the trajectory
    trajectory = [starting_pos]
    for i in range(2*scale):
        sorted_moves = sort_by_distance_to_end(current_pos, possible_moves, ending_pos)

        weighted_moves = [] #we randomly select our next move from here
        for i in range(len(sorted_moves)):
            for j in range(prob_distribution[i]):
                weighted_moves.append(sorted_moves[i])
        move = random.choice(weighted_moves)
        current_pos[0] += move[0]
        current_pos[1] += move[1]

        #now determine if the "chasing boat" is visible
        rand_number = random.random() #random number between 0 and 1
        if rand_number > (prob_distribution[9]/100):
            trajectory.append((current_pos[0], current_pos[1]))
        else:
            trajectory.append(("invisible", "invisible"))

    return trajectory

def generate_circling_points(scale, prob_distribution):
    #method objective:
    #return a list of points of a "circling boat" moving around the "victim boat" with a given erraticity

    #parameter descriptions:
    #grid across which the chase is happening will have dimension (2*scale) x (2*scale), scale must be > 0
    #the "victim" boat will always have position (0, 0) in this grid
    #prob_distribution is a list of ten non-negative integers, the first 9 sum up to 100, the last entry is less than 100
    #the 1st entry in prob_distribution refers to the probability the chasing boat moves to the grid position that is closest to the tangent vector to the victim boat, the 2nd entry refers to the probability the chasing boat moves towards the 2nd closest grid position that is the 2nd closest to the tangent vector to the victim boat and so on...
    #the last entry of prob_distribution refers to to the probability the chasing boat disappears from view

    #returned values:
    #list of points of length 2*"scale"
    rotation_direction = random.choice(["CW", "CCW"])
    starting_pos = (0, 0)
    while starting_pos == (0, 0):
        starting_pos = (random.choice([x for x in range(-1*scale, scale + 1)]), random.choice([x for x in range(-1*scale, scale + 1)]))

    list_of_positions = []
    list_of_positions.append(starting_pos)

    current_pos = [starting_pos[0], starting_pos[1]]

    possible_moves = [(0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1)] #displacement vectors

    #we now create the trajectory
    trajectory = [starting_pos]
    for i in range(8*scale):
        if rotation_direction == "CW":
            target_vector = (current_pos[1], -1*current_pos[0])
        else:
            target_vector = (-1*current_pos[1], current_pos[0])
        sorted_moves = sort_by_distance_to_vector(target_vector, possible_moves)

        weighted_moves = [] #we randomly select our next move from here
        for i in range(len(sorted_moves)):
            for j in range(prob_distribution[i]):
                weighted_moves.append(sorted_moves[i])
        move = random.choice(weighted_moves)
        current_pos[0] += move[0]
        current_pos[1] += move[1]

        #now determine if the "chasing boat" is visible
        rand_number = random.random() #random number between 0 and 1
        if rand_number > (prob_distribution[9]/100):
            trajectory.append((current_pos[0], current_pos[1]))
        else:
            trajectory.append(("invisible", "invisible"))

    return trajectory

def write_trajectories_to_file(trajectories, type="sampleTrajectory", numLines=0, prob_dist=[]):
    name = type + "_" + str(lines) + str(prob_dist) + ".txt"
    outputFile = open(name, "w")
    #let me know if you need to change the formatting of this csv
    for trajectory in trajectories:
        line = ""
        for point in trajectory:
            line = line + "(" + point[0] + ", " + point[1] + "), "
        line = line[0:len(line)-2] + "\n"
        outputFile.write(line)
    outputFile.close()

def display_trajectory(trajectory):
    x = []
    y = []
    for point in trajectory:
        if point[0] != "invisible":
            x.append(point[0])
            y.append(point[1])
    plt.plot(x, y, 'o', color='black')
    plt.show()




if __name__ == "__main__":
    trajectory = generate_circling_points(10, (60, 15, 15, 10, 0, 0, 0, 0, 0, 0)) #example
    display_trajectory(trajectory)
