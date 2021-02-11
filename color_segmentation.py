#!/usr/bin/env python
# coding: utf-8

# In[103]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# def calculate_distance_l2(img1, img2):
#     result = (img1 - img2) ** 2
#     if img1.shape[-1] == 3: # RGB image
#         result = np.sum(result, axis=-1)
#     return np.sqrt(result)

# img = cv2.cvtColor(cv2.imread('ishihara.png'), cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(cv2.imread('ishihara.png'), cv2.COLOR_BGR2HSV)

print(img.shape)
plt.imshow(img)
plt.show()

chans = cv2.split(img)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []

# loop over the image channels
for (chan, color) in zip(chans, colors):
    # create a histogram for the current channel and
    # concatenate the resulting histograms for each
    # channel
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    features.extend(hist)
    # plot the histogram
    plt.plot(hist, color = color)
    plt.xlim([0, 256])
plt.show()

# Reference background color = white (255, 255, 255) in RGB
white_bg = (255, 255, 255)

distance = (img - white_bg)**2
if img.shape[-1] == 3: # RGB image
    distance = np.sqrt(np.sum(distance, axis=-1))

print(distance.shape)
plt.imshow(distance, cmap='gray', vmin=0, vmax=255)
plt.show()


# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
# Z = np.float32(img.reshape((-1,3)))
# Z = np.tile(distance, (3, 1, 3))

# Reshape just to obtain 640000 (800*800)
Z = np.float32(distance.reshape((-1, 1)))

# print(Z.shape)

ret,labels,centers = cv2.kmeans(Z, K, bestLabels=None, criteria=criteria, attempts=10, 
                            flags=cv2.KMEANS_RANDOM_CENTERS)

# labels = labels.reshape((img.shape[:-1]))
labels = labels.reshape((distance.shape))
reduced = np.uint8(centers)[labels]

# print(labels)

results = list()

for i, c in enumerate(centers):
    mask = cv2.inRange(labels, i, i)
#     mask = np.dstack([mask] * 3)
    
    # The mask should be 1D now
    mask = np.dstack([mask])
    
    ex_reduced = cv2.bitwise_and(reduced, mask)
    
    results.append(ex_reduced)

for i, im in enumerate(results):
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.show()

# print(center.shape, center)

# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((distance.shape))

# plt.imshow(res2)
# plt.show()





# n_clusters = 2

# km = KMeans(n_clusters=n_clusters, n_init=40, max_iter=500)
# y_km = kmeans.fit_predict(distance)

# reshaped = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

# km = KMeans(n_clusters=n_clusters, n_init=40, max_iter=500).fit(reshaped)

# km = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300).fit(distance)

# # print('labels_ = ', km.labels_)

# #clustering = np.reshape(np.array(km.labels_, dtype=np.uint8), (img.shape[0], img.shape[1]))
# clustering = np.tile(km.labels_, (800, 1))
# # print(clustering, clustering.shape)

# sorted_labels = sorted([n for n in range(n_clusters)], key=lambda x: -np.sum(clustering == x))

# km_image = np.zeros(img.shape[:2], dtype=np.uint8)
# km_image = np.copy(img)

# for i, label in enumerate(sorted_labels):
#     km_image[clustering == label] = int((255) / (n_clusters - 1)) * i
#     plt.imshow(km_image)
#     plt.show()

# plt.imshow(km_image)
# plt.show()

# otsu_th = cv2.threshold(result,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# print(otsu_th.shape)
# plt.imshow(otsu_th)
# plt.show()

