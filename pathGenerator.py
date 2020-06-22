#author: Raghav Samavedam
import random, math
from operator import itemgetter

def sort_by_magnitude(current_position, displacement_vectors):
    #sorts list of moves in increasing order of the magnitude of the position gained from applying the move to the current position
    move_magnitude = []
    for move in displacement_vectors:
        position = (current_position[0] + move[0], current_position[1] + move[1])
        position_magnitude = math.sqrt(position[0]**2 + position[1]**2)
        move_magnitude.append((move, position_magnitude))
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
    #list of points of length "scale" + 1

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
        sorted_moves = sort_by_magnitude(current_pos, possible_moves)

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


if __name__ == "__main__":
    points = generate_chase_points(10, (100, 0, 0, 0, 0, 0, 0, 0, 0, 0)) #example 
    print(points)
