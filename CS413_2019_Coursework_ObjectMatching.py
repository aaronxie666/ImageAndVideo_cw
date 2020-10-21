import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from scipy.cluster.vq import *


#set the file rul
# trainData_dir = './coursework-data/objects-train/'
# testData_dir = './coursework-data/objects-test/'
#own made train&test data 
trainData_dir = './coursework-data/changMadeData/img_train/'
testData_dir = './coursework-data/changMadeData/video_test/'
#get all training images in the objects-train directory
trainFileNames = glob.glob(trainData_dir+'*.png')
print(len(trainFileNames))

#stroe all decriptors for buding a dictionary(the bag of words)
des_list = []
kp_list = []
descriptors_all = []

for i in range(len(trainFileNames)):
	#load a training image
	img = cv.imread(trainFileNames[i])
	#set the training image to gray scale
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	#initialize the sift object
	# sift = cv.xfeatures2d.SIFT_create()
	surf = cv.xfeatures2d.SURF_create(400)

	#get the sift descripter and key points
	# kp, des = sift.detectAndCompute(gray, None)
	kp, des = surf.detectAndCompute(gray, None)
	print(len(kp), len(des))
	des_list.append((trainFileNames[i], des))
	kp_list.append((trainFileNames[i], kp))
	#sum all individule descriptors for later clustering
	for d in des:
		descriptors_all.append(d)

#k-means clustering
#set k is 200
k = 200
#set batch_size
batch_size = np.size(os.listdir(trainData_dir))*3
print(batch_size)
#perform k-means
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(descriptors_all)
# kmeans = KMeans(k)
# kmeans.fit(descriptors_all)

#set the verbose false to avoid too many printed info
kmeans.verbose = False

#histogram list containing all training images
histo_list = []

#label names of all training images
Y = []

for i in range(len(trainFileNames)):
	#initialize the histogram
	histo = np.zeros(k)
	nkp = np.size(des_list[i][1])
	print(nkp)

	for d in des_list[i][1]:
		idx = kmeans.predict([d])
		#1/nkp:normalize the histogarm(nkp can be removed due to the less images)
		histo[idx] += 1/nkp
	# plt.bar(list(range(100)), histo)
	# plt.show()
	histo_list.append(histo)

	address = des_list[i][0]
	name = address.rsplit("/", 1)[-1]
	Y.append(name)

X = np.array(histo_list)
# Y = []
# #may inproved by differentiating same/diff things
# for i in range(len(trainFileNames)):
# 	address = des_list[i][0]
# 	name = address.rsplit("/", 1)[-1]
# 	Y.append(name)



# mlp = MLPClassifier(verbose=True, max_iter=100000)
# mlp.fit(X,Y)

videoCap = cv.VideoCapture(testData_dir+"objects-test-2_own.mov")

#feature matching initialize for labeling the region
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

frame_count = 0
res = []

surf = cv.xfeatures2d.SURF_create(400)
# kp_frame_firstframe = []
# des_frame_firstframe = np.empty((0))

# ret, first_frame = videoCap.read()
# first_frame_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# kp_frame_firstframe, des_frame_firstframe = surf.detectAndCompute(first_frame_gray, None)
# kp_frame_firstframe_pt = [int(p.pt) for p in kp_frame_firstframe]
# print(len(kp_frame_firstframe), len(kp_frame_firstframe_pt))
# print(kp_frame_firstframe_pt[:20])
for i in range(100):
	ret,frame = videoCap.read()

while (1):
	ret,frame = videoCap.read()
	if ret==True:
		frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		
		# kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
		kp_frame, des_frame = surf.detectAndCompute(frame_gray, None)
		print(len(kp_frame))

		if len(kp_frame) >= 10:
			histo_frame = np.zeros(k)
			nkp_frame = np.size(kp_frame)
			for d in des_frame:
				idx_frame = kmeans.predict([d])
				histo_frame[idx_frame] += 1/nkp_frame
			distance_list = []
			#perform NN to find the most similar histogram
			for i in range(len(trainFileNames)):
				distance = mean_squared_error(histo_frame, X[i])
				distance_list.append(distance)
			res = np.argsort(distance_list)[:3]
			print(Y[res[0]], Y[res[1]], Y[res[2]])

			#set the training image of the shortest distance from the frame as the template
			template = cv.imread(trainData_dir+Y[res[0]], cv.IMREAD_GRAYSCALE)
			kp_list_template = kp_list[res[0]][1]
			flann_matches = flann.knnMatch(des_list[res[0]][1], des_frame, k=2)
			good_points = []
			for m,n in flann_matches:
				if m.distance < 0.6*n.distance:
					good_points.append(m)
			
			#set the training image of the second shortest distance from the frame as the template
			template_1 = cv.imread(trainData_dir+Y[res[1]], cv.IMREAD_GRAYSCALE)
			flann_matches_1 = flann.knnMatch(des_list[res[1]][1], des_frame, k=2)
			good_points_1 = []
			for m,n in flann_matches_1:
				if m.distance < 0.6*n.distance:
					good_points_1.append(m)
				
			#set the training image of the third shortest distance from the frame as the template
			template_2 = cv.imread(trainData_dir+Y[res[2]], cv.IMREAD_GRAYSCALE)
			flann_matches_2 = flann.knnMatch(des_list[res[2]][1], des_frame, k=2)
			good_points_2 = []
			for m,n in flann_matches:
				if m.distance < 0.6*n.distance:
					good_points_2.append(m)
			
			#compare the number of good_points of each template
			good_points_list = [len(good_points), len(good_points_1), len(good_points_2)]
			greatest_index = good_points_list.index(max(good_points_list))

			#set the template with most good points as the final template
			if greatest_index == 1:
				good_points = good_points_1
				kp_list_template = kp_list[res[1]][1]

			elif greatest_index == 2:
				good_points = good_points_2
				kp_list_template = kp_list[res[2]][1]

			print(good_points_list, greatest_index, len(good_points))

			#make sure the number of good points is enough to make sure a region
			if len(good_points) >= 10:
				query_pts = np.float32([kp_list_template[m.queryIdx].pt for m in good_points]).reshape(-1,1,2)
				train_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_points]).reshape(-1,1,2)
				matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
				if matrix is not None:
					matches_mask = mask.ravel().tolist()

					h,w = template.shape
					pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
					dst = cv.perspectiveTransform(pts, matrix)
					homography = cv.polylines(frame, [np.int32(dst)], True, (255,0,0), 3)
					cv.imshow('Video', homography)
				else:
					cv.imshow('Video', frame)
			else:
				cv.imshow('Video', frame)
			
		else:
			cv.imshow('Video', frame)
			
		if cv.waitKey(1)&0xFF ==ord('q'):
			break
		
	else:
		break
	
	frame_count += 1


videoCap.release()
cv.destroyAllWindows()







