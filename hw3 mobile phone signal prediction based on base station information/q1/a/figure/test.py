import matplotlib.pyplot as plt


axis = [i for i in range(1,7,1)]
axis_w = ['a', 'b', 'c', 'd', 'e', 'f']
value = [3,2,5,1,6,8]
value2 = [1,2,3,4,5,6]
width = 0.2
print(axis)
plt.bar(axis, value, label='1', width=width)
for i in range(len(axis)):
    axis[i] += width
plt.bar(axis, value2, label='2', tick_label=axis_w, width =width)
plt.show()

# import matplotlib.pyplot as plt
#
# name_list = ['Monday', 'Tuesday', 'Friday', 'Sunday']
# num_list = [1.5, 0.6, 7.8, 6]
# num_list1 = [1, 2, 3, 1]
# x = list(range(len(num_list)))
# total_width, n = 0.8, 2
# width = total_width / n
#
# plt.bar(x, num_list, width=width, label='boy', fc='y')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, num_list1, width=width, label='girl', tick_label=name_list, fc='r')
# plt.legend()
# plt.show()