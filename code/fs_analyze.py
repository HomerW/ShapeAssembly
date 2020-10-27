import pickle
import matplotlib.pyplot as plt

with open("program_fit/fit/fs_data", "rb") as file:
    fs_data = pickle.loads(file.read())

# print(sorted(fs_data.items(), key=lambda x: x[1]))

# print(fs_data)
#
fig, ax = plt.subplots()
ax.hist(fs_data.values())
ax.set_title("pre-optimization f-scores, 234 val. shapes")
ax.set_ylabel("count")
ax.set_xlabel("f-score")
plt.savefig("pre_234_shapes.png")
