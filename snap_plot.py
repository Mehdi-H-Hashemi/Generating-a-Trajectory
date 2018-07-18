import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
#
# Enter Waypoint coordinates here. Each row is coordinates of one waypoint.
#
waypoints = np.matrix([[0,    0,   0],
              [1,    1,   1],
              [2,    0,   2],
              [3,    -1,  1],
              [4,    0,   0],
              [5,    2,    0],
              [6,    1,    1] ])
waypoints = np.matrix.transpose(waypoints)
# print(waypoints)
# S = np.matrix([[0, 1, 2, 3, 4, 5]]);  
#end = S[0,:].size
#T = S[0,1:end] - S[0,0:end-1]
n = waypoints[0,:].size - 1
m = waypoints[:,0].size
#
# Initialize
#
end = waypoints[0,:].size
# print(end)
d = waypoints[:,1:end] - waypoints[:,0:end-1]
d0 = 1 * np.sqrt(np.power(d[0,:],2) + np.power(d[1,:],2) + np.power(d[2,:],2))
#print(d0)
traj_time = np.concatenate(([[0]], np.cumsum(d0)), axis=1)
#print(traj_time)
(alpha, alpha_b) = snap(waypoints, traj_time)
m = traj_time[0,:].size
delta_t = 0.01
end = traj_time[0,:].size - 1
i_max = np.floor(traj_time[0,end]/delta_t)
#print(i_max)

for i in range(int(i_max)):
    t = i*delta_t
    index = np.asarray(np.where(traj_time >= t)).T
    t_index = index[0,1]
    #print(t_index)
    if t_index > 0:
        t = t - traj_time[0, t_index - 1]
        #print(t)
    if t == 0:
        position = waypoints[:,0]
        #print(position)
    else:
        ii = t_index-1
        #print(ii)
        pos = np.zeros((3,1))
        scale = t/d0[0,t_index-1]
        #print(scale)
        for j in range(8):
            alph = np.asmatrix(alpha[8*(ii)+j,:]*scale**(j))
            #print(alph)
            pos = pos + alph.T
            #print(pos)       
        position = np.concatenate((position, pos),axis = 1)

x = np.array(position[0,:])[0]
y = np.array(position[1,:])[0]
z = np.array(position[2,:])[0]

fig = plt.figure(figsize=(8,8))
ax = fig.gca(projection='3d')
#ax.plot(x, y, z, label='trajectory')
ax.scatter(x, y, z, label='Drone trajectory')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend()


plt.show()




