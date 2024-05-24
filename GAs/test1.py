from master_project.GAs.simulation import *
import h5py

"""
full_path = "/home/samani/Documents/sim"


sp1 = np.zeros((20, 20))
sp2 = np.zeros((20, 20))
sp1_cells = np.zeros((20, 20))
sp1_cells[:, 0] = 1
sp2_cells = np.zeros((20, 20))
sp2_cells[:, 5] = 1
params = np.array([[.5, .5, 4, 4, .1, .1]])
dt = 0.01
sim_start = 1
sim_stop = 20
epochs = 500
target_shape = (20, 20)
result = simulation(sp1, sp2, sp1_cells, sp2_cells, params, dt, sim_start, sim_stop, epochs, target_shape)





full_file_path = os.path.join(full_path, "sim.h5")

with h5py.File(full_file_path, "w") as file:
    file.create_dataset("sp2", data=result)


file = h5py.File("/home/samani/Documents/sim/sim.h5", "r")
data = np.array(file["sp2"])
print(data)
print(data[:, 4])
print(data[:, 17])

plt.figure(figsize=(10, 10))
plt.imshow(data, cmap="hot", interpolation="nearest")
plt.title("Target", fontsize=20)
plt.colorbar(shrink=0.9)
plt.axis("off")
plt.show()
"""





file = h5py.File("/home/samani/Documents/sim/sim.h5", "r")
data = np.array(file["sp2"])