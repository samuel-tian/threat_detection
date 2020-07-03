import numpy as np
from HMMClasses.HMMSingleLogScaling import HMMScaling
from HMMClasses.HMMDecimalClass import HMMDecimal

def read_trajectories_from_file(filename):
    trajectoryFile = open(filename)
    firstLine = trajectoryFile.readline()
    numTrajectories = int(firstLine)
    numTrajectoriesRead = 0
    trajectories = []
    while numTrajectoriesRead < numTrajectories:
        trajectories.append([])
        numPointsRead = 0
        numPointsPerTrajectory = int(trajectoryFile.readline())
        while numPointsRead < numPointsPerTrajectory:
            line = trajectoryFile.readline()
            point = ( int(line[0:line.index(" ")]) + 128, int(line[line.index(" ") + 1:]) + 128 )
            trajectories[numTrajectoriesRead].append(point)
            numPointsRead += 1
        numTrajectoriesRead += 1

    return trajectories

if __name__ == "__main__":
	n = 5
	k = 256
	emit_prob = np.empty(dtype=float, shape=(n, k*k))
	for i in range(n):
		cur = 1
		for j in range(k*k-1):
			x = randint(1, 9)
			emit_prob[i][j] = cur * x / 10
			cur -= cur * x / 10
		emit_prob[i][k*k-1] = cur
	following_HMM = HMMScaling(emit_prob)
	normal_HMM = HMMScaling(emit_prob)
	print(following_HMM)

	trajectory_list = read_trajectories_from_file("sampleTrajectory_0[].txt")
	print(trajectory_list)
	for i in range(len(trajectory_list)):
		trajectory = trajectory_list[i]
		np_trajectory = np.empty(dtype=int, shape=len(trajectory))
		for j in range(len(trajectory)):
			np_trajectory[j] = trajectory[j][0] * k + trajectory[j][1]
		if (i == 0):
			following_HMM.baum_welch(np_trajectory)
			print("trained following")
		elif (i == 1):
			normal_HMM.baum_welch(np_trajectory)
			print("trained normal")
			print(normal_HMM.evaluate(np_trajectory))
		else:
			following_eval = following_HMM.evaluate(np_trajectory)
			normal_eval = normal_HMM.evaluate(np_trajectory)
			if following_eval > normal_eval:
				print("following", following_eval, normal_eval)
			else:
				print("normal", following_eval, normal_eval)
	print(following_HMM)
	trajectory_list = read_trajectories_from_file("sampleTrajectory_1[].txt")
	for i in range(len(trajectory_list)):
		trajectory = trajectory_list[i]
		trajectory = trajectory_list[i]
		np_trajectory = np.empty(dtype=int, shape=len(trajectory))
		for j in range(len(trajectory)):
			np_trajectory[j] = trajectory[j][0] * k + trajectory[j][1]
		following_eval = following_HMM.evaluate(np_trajectory)
		normal_eval = normal_HMM.evaluate(np_trajectory)
		if following_eval > normal_eval:
			print("following", following_eval, normal_eval)
		else:
			print("normal", following_eval, normal_eval)
