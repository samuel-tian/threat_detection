#objective is to generate points that start from edge of grid and move towards center

def generate_chase_points(scale, prob_distribution):
    #grid across which the chase is happening will have dimension (2*scale) x (2*scale)
    #the "victim" boat will always have position (0, 0) in this grid
    #prob_distribution is a list of ten non-negative integers that sum up to 100
    #the 1st entry in prob_distribution refers to the probability the chasing boat moves to the grid position that is closest to the victim boat, the 2nd entry refers to the probability the chasing boat moves towards the 2nd closest grid position that is the 2nd closest to the victim boat and so on...
    #the last entry of prob_distribution refers to to the probability the chasing boat disappears

    
