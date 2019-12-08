#%%
import numpy as np

a = np.array(range(10))
b = np.array_split(a,3)
c = np.concatenate(b)

# %%
