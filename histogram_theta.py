import matplotlib.pyplot as plt

with open('data_new/experiment-2/all_classes/theta_list.txt', 'r') as f:
    data = f.read()

theta_list = [int(i) for i in data.split()]

plt.hist(theta_list, density=False, bins=61)

plt.xlabel('Theta [degrees]')
plt.ylabel('Count')

# plt.title('Histogram of Theta for Classes: CAR/VAN')
# plt.title('Histogram of Theta for Classes: CYCLIST')
plt.title('Histogram of Theta for Classes: ALL')

plt.savefig('data_new/experiment-2/all_classes/theta_distribution-exp2-all_classes.png')

plt.show()
