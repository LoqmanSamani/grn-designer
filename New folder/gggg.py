import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import h5py
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec





path = "data.h5"
data = h5py.File(path, "a")




fig, ax = plt.subplots(figsize=(10, 10))

# First heatmap with 'inferno' colormap and an alpha value
sns.heatmap(data["ind"][3], cmap="plasma", ax=ax, cbar=False)

#sns.heatmap(data["target"][1], cmap="inferno", alpha=0.3, ax=ax, cbar=False)
plt.axis('off')
plt.show()
