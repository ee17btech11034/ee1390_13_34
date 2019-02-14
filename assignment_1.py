import numpy as np
from matplotlib import pyplot as plt

origin = np.array([0, 0])

c1 = np.array([1, 3])
c2 = np.array([4, -1])
radius2 = 3
radius1_min = 2
radius1_max = 8

circle1 = plt.Circle(c2, radius2, color='g', fill=False)
circle2 = plt.Circle(c1, radius1_min, color='r', fill=False)
circle3 = plt.Circle(c1, radius1_max, color='c', fill=False)

ax = plt.gca()
plt.axis('equal')

ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)

plt.plot(c1[0], c1[1], 'o')
plt.plot(c2[0], c2[1], 'o')
plt.text(c1[0], c1[1], 'C1 (1,3)')
plt.text(c2[0], c2[1], 'C2 (4,-1)')

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid(True, which='both')
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.show()
