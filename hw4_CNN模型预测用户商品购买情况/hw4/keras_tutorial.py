import numpy as np

rand = np.random.randint(1,10,(8,4))
print(rand)
rand = rand.reshape((-1, 8, 2))
print(rand.shape)