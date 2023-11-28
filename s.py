import numpy as np
import matplotlib.pyplot as plt
import random
import math
import libs.genetic_all as ga  # Assuming you have a module named 'ga' with genetic algorithm functions

# Define your constants
lpop = 50
lstring = 20
numgen = 10000
M = 18  # Assuming this is the number of cities

# Initialize B
B = np.array([
    [0, 0], [77, 68], [12, 75], [32, 17], [51, 64],
    [20, 19], [72, 87], [80, 37], [35, 82], [2, 15],
    [18, 90], [33, 50], [85, 52], [97, 27], [37, 67],
    [20, 82], [49, 0], [62, 14], [7, 60], [10, 10]  # Assuming 20 cities
])

# Initialize other variables
FitMin = float('inf')
PopMin = None
evolution = np.zeros(numgen + 1)

def basicGA(Pop, Fit):
    SubPop1, _ = ga.selbest(Pop, Fit, [1, 1, 1], 0)
    SubPop2, _ = ga.seltourn(Pop, Fit, 47)

    ga.swappart(SubPop1, 0.11)
    ga.swapgen(SubPop2, 0.17)

    Pop = np.vstack((SubPop1, SubPop2))
    return Pop

def fitness(Pop, B):
    Fit = np.zeros((lpop), float)

    for i in range(lpop):
        for j in range(lstring - 1):
            x = int(Pop[i, j])
            n = int(Pop[i, j + 1])

            if 0 <= x < len(B) and 0 <= n < len(B):
                dis = math.sqrt(((B[x][0] - B[n][0]) ** 2) + ((B[x][1] - B[n][1]) ** 2))
                Fit[i] += dis
            else:
                print(f"Index out of bounds - x={x}, n={n}")

        if 0 <= int(Pop[i][1]) < len(B) and 0 <= int(Pop[i][17]) < len(B) and 0 <= 19 < len(B):
            Fit[i] += (math.dist(B[0], B[int(Pop[i][1])]) + math.dist(B[int(Pop[i][17])], B[19]))
        else:
            print(f"Index out of bounds - Pop[i][1]={Pop[i][1]}, Pop[i][17]={Pop[i][17]}, 19={19}")

    return Fit

def genrpop_perm():
    Pop = np.zeros((lpop, lstring), int)
    for i in range(lpop):
        shuffled_array = np.random.permutation(range(1, M + 1))[:lstring]
        Pop[i, :len(shuffled_array)] = shuffled_array

    return Pop

# Main loop
for gen in range(1, numgen + 1):
    Pop = genrpop_perm()
    Fit = fitness(Pop, B)
    evolution[gen] = min(Fit)

    if FitMin > Fit[0]:
        FitMin = Fit[0]
        PopMin = Pop[0]

    if FitMin < 500:
        break

    Pop = basicGA(Pop, Fit)

plt.plot(range(numgen + 1), evolution, label='Convergence')
plt.xlabel('Generation')
plt.ylabel('Minimum Fitness Value')
plt.title('Genetic Algorithm Convergence')
plt.legend()
plt.show()

temparray = np.zeros((lstring + 2, 2))
for i in range(lstring):
    p_act = PopMin[i]
    temparray[i + 1] = B[p_act]
temparray[0] = B[0]
temparray[19] = B[19]

x_coords, y_coords = temparray[:, 0], temparray[:, 1]

plt.plot(x_coords, y_coords, marker="o", linestyle="-")
for i, (xi, yi) in enumerate(temparray):
    plt.text(xi, yi, f"{i + 1}")

plt.title("TSP")
plt.xlabel("X súradnica")
plt.ylabel("Y súradnica")
plt.show()