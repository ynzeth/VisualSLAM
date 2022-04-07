import numpy as np
import matplotlib.pyplot as plt

with open('similarityMatrix.npy', 'rb') as f:
    similarities = np.load(f)

with open('matchings.npy', 'rb') as f:
    matchings = np.load(f).T

similarities = similarities.T
N_database = similarities.shape[0]
N_query = similarities.shape[1]

maxmatchings = np.argmax(similarities, axis=1)
naiveOptimal = np.stack((range(0,N_database)[0::1], maxmatchings[0::1]))

x, y = np.meshgrid(np.linspace(0, N_query, N_query), np.linspace(0, N_database, N_database))
z = similarities[:-1, :-1]
maxVal = np.amax(similarities)
minVal = np.amin(similarities)

plt.pcolormesh(x, y, z, cmap='Blues', vmin=minVal, vmax=maxVal)
plt.scatter(naiveOptimal[1], naiveOptimal[0],50, alpha=0.4, color='black')
plt.scatter(matchings[1],matchings[0], 5, alpha=0.5, color='red')

plt.show()