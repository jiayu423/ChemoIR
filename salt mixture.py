from utils.ProcessData import *

run1_spec = loadSpec('Data/sodium salt mixture/run1/', [[400, 1200]], [5, 13])

run1_temp = loadTemperature('Data/sodium salt mixture/temp_run1.xlsx')
selected_temp = []
for i in range(run1_spec.shape[0]):

    dt = np.abs(run1_temp[:, 0] - run1_spec[i, -1])
    ind = np.where(dt==dt.min())[0][0]
    print(run1_spec[i, -1], run1_temp[ind, 0], ind, run1_temp[ind, 1])
    selected_temp.append(run1_temp[ind, 1])

run1_all = np.hstack((run1_spec, np.array(selected_temp).reshape(-1, 1)))
plt.scatter([i for i in range(run1_spec.shape[0])], run1_all[:, -1])
plt.show()