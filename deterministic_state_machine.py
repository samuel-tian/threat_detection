#author: Raghav Samavedam
import math
from operator import itemgetter
from pathGenerator import *

def determine_chasing_1(trajectory):
    #uses three state process to determine if vessel is chasing
    #outlined on scratch paper
    current_state = 1
    for i in range(1, len(trajectory)):
        current_pos = trajectory[i]
        previous_pos = trajectory[i-1]
        possible_moves = [(0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1)] #displacement vectors
        move_magnitude = []
        for move in possible_moves:
            position = (current_pos[0] + move[0], current_pos[1] + move[1])
            distance = math.sqrt(position[0]**2 + position[1]**2)
            move_magnitude.append((move, distance))
        sorted_moves_plus_mag = sorted(move_magnitude, key=itemgetter(1))
        sorted_moves = [x[0] for x in sorted_moves_plus_mag]

        actual_move = (current_pos[0] - previous_pos[0], current_pos[1] - previous_pos[1])
        actual_move_rank = sorted_moves.index(actual_move)

        #implement the deterministic model

        if actual_move_rank <= 1:
            current_state += 1

        if actual_move_rank >= 8 and current_state > 1:
            current_state -= 1

        if current_state == 55:
            #we are in the accepting state, boat is determined to be chasing
            return ("chasing", i)

    return ("not chasing")


if __name__ == "__main__":
    successes = 0
    totalSteps = 0
    for i in range(1000):
        trajectory = generate_chase_points(100, (60, 15, 15, 10, 0, 0, 0, 0, 0, 0))
        result = determine_chasing_1(trajectory)
        if result[0] == "chasing":
            successes += 1
            totalSteps += result[1]
    success_rate = successes / 1000
    average_number_of_steps_required = 69 #totalSteps / successes
    print("Success rate: " + str(success_rate*100) + "%")
    print("Average number of steps required to determine that the boat was chasing: " + str(average_number_of_steps_required))
