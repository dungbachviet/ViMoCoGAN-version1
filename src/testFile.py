import numpy as np
import random

#
# a = np.array([[[1, 2], [3, 4]], [[1, 1], [1, 1]]])
# b = np.array([[[5, 6], [7, 8]], [[1, 1], [1, 1]]])
# c = np.concatenate([a, b], axis=1)
#
#
# print(a)
# print(b)
# print(c)
# print(c.shape)

#
# print(np.linspace(1, 70, 16, dtype=int))
# index = np.linspace(1, 70, 16, dtype=int)
# for i in range(1,71):
#     if i in index :
#         print(i)


# print( (2 == 2))
#
# test = np.array([0,1,2,3,4,5,6,7,8,9,10,11], dtype=float)
# test = (test/4).astype(int)
#
#
# print(test)


# view_list = []
# for i in range(5):
#     view_list.extend(range(12))
# print(view_list)
#
# factor = 2
# for i in range(factor):
#     view_list.extend(view_list)
# print(view_list)
# import os
# os.makedirs("./abc/111/222/333")


# view_list = []
# for i in range(2):
#     view_list.extend([i] * 12)
# print(view_list)
#
# factor = 2
# for i in range(factor):
#     view_list.extend(view_list)
# print(view_list)
# print(len(view_list))


# classes_to_generate = np.array([0, 1, 2, 3], dtype=int)
# print(classes_to_generate)
# classes_to_generate += 1
# print(classes_to_generate % 4)






num_frames = 16
chosen_frames = list(np.linspace(1, num_frames, 16, dtype=np.int))
print(chosen_frames)


interval = int((num_frames-16)/(15*2))
print("interval = ", interval)
if (interval > 0):
    for i in range(16):
        if (i == 0 or i == 15): continue
        chosen_frames[i] += random.randint(-interval, interval)

    print("chosen_frames (update) = ", chosen_frames)


















