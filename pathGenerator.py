#author: Raghav Samavedam
import random

def magnitude()

def generate_chase_points(scale, prob_distribution):
    #method objective:
    #return a list of points of a "chase boat" moving from one side of the grid towards the "victim boat" with a given erraticity

    #parameter descriptions:
    #grid across which the chase is happening will have dimension (2*scale) x (2*scale), scale must be > 0
    #the "victim" boat will always have position (0, 0) in this grid
    #prob_distribution is a list of ten non-negative integers that sum up to 100
    #the 1st entry in prob_distribution refers to the probability the chasing boat moves to the grid position that is closest to the victim boat, the 2nd entry refers to the probability the chasing boat moves towards the 2nd closest grid position that is the 2nd closest to the victim boat and so on...
    #the last entry of prob_distribution refers to to the probability the chasing boat disappears

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

    current_pos = starting_pos
    
