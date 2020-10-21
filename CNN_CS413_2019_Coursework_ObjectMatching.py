import keras
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as pt
import matplotlib.image as mpimg
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout, Softmax, Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop, Adam
import glob
from PIL import Image
import cv2 as cv

trainData_dir = '/Users/chang/Desktop/Master/ImageAndVideo/CW/coursework-data/objects-train/'
testData_dir = '/Users/chang/Desktop/Master/ImageAndVideo/CW/coursework-data/objects-test/'
BATCH_SIZE = 32
MAX_EPOCH = 50

#get all training images in the objects-train directory
trainFileNames = glob.glob(trainData_dir+'*.png')
print(len(trainFileNames))

num_classes = len(trainFileNames)
labels = []
img_train = []
label_train = []
for i in range(len(trainFileNames)):
	img = Image.open(trainFileNames[i]).convert('L')
	img = np.array(img)
	img = cv.cvtColor(cv.resize(img, (224, 224)), cv.COLOR_GRAY2BGR)
	img_train.append(img)
	name = trainFileNames[i].rsplit("/", 1)[-1]
	labels.append(name)
	label_train.append(i)

#transfer the training data to the format that vgg16 cna accept(unit-8)
img_train = np.concatenate([arr[np.newaxis] for arr in img_train]).astype('float32')
label_train = to_categorical(label_train, num_classes=num_classes)

#rebuild the VGG16 networks to fit the training set & output labels
inputs = Input(shape=[224, 224, 3])




#TRANING MODEL(CAN USE THE TRAINED MODEL DERECTLY BY COMMENT THE LINES)
# ---------------------------------------------------------------------------------------
# adjust the vgg16 model to fit the new training data
model_vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
for layer in model_vgg16.layers:
	layer.trainable = False
fla = Flatten()(model_vgg16.output)				#save the vgg16 layers and weithts
#add addtional layers for the training data
fc6 = Dense(4096, activation='relu')(fla)
drop6 = Dropout(rate = 0.5)(fc6)
fc7 = Dense(4096, activation='relu')(drop6)
drop7 = Dropout(rate = 0.5)(fc7)
fc8 = Dense(4096, activation='relu')(drop7)
drop8 = Dropout(rate = 0.5)(fc8)
fc9 = Dense(num_classes, activation='softmax')(drop8)
#build the model
model = Model(inputs=inputs, outputs=fc9)
optimizer = Adam(lr=0.00001)
model.summary()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(img_train, label_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCH)
model.save(filepath='./retrained_vgg16.h5')
# -----------------------------------------------------------------------------------------



#load trained model for time consuming
model = load_model('retrained_vgg16.h5')
#test an imge first
img = load_img(trainData_dir+"Quorn.png", target_size =(224,224))
img_data = img_to_array(img)
print(img_data.shape)
img_data = img_data.reshape((1,)+img_data.shape)
print(img_data.shape)
img_data = preprocess_input(img_data)
#make prediction of the test img
prediction = model.predict(img_data)
#print the top 3 predictions
print(np.argsort(prediction)[0][:-4:-1])
#print the best prediction
prediction = labels[np.argmax(prediction)]
print(prediction)

#match with the video
des_list = []
kp_list = []
descriptors_all = []

for i in range(len(trainFileNames)):
	#load a training image
	img = cv.imread(trainFileNames[i])
	#set the training image to gray scale
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	surf = cv.xfeatures2d.SURF_create(400)
	kp, des = surf.detectAndCompute(gray, None)
	print(len(kp), len(des))
	des_list.append((trainFileNames[i], des))
	kp_list.append((trainFileNames[i], kp))
	for d in des:
		descriptors_all.append(d)

videoCap = cv.VideoCapture(testData_dir+"objects-test-2.mov")

#feature matching initialize for labeling the region
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

frame_count = 0
res = []

surf = cv.xfeatures2d.SURF_create(400)

for i in range(100):
	ret,frame = videoCap.read()

while (1):
	ret,frame = videoCap.read()
	if ret==True:
		frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		kp_frame, des_frame = surf.detectAndCompute(frame_gray, None)
		print(len(kp_frame))
		
		#reshape the frame size for the prediction
		frame_pre = cv.resize(frame, (224, 224))
		frame_pre = img_to_array(frame_pre)
		frame_pre = frame_pre.reshape((1,)+frame_pre.shape)
		frame_pre = preprocess_input(frame_pre)
		#make prediction
		prediction = model.predict(frame_pre)
		#get the top 3 best prediction
		res = np.argsort(prediction)[0][:-4:-1]
		print(len(res), res)
		#get the best prediction
		prediction = labels[np.argmax(prediction)]
		print(prediction)
		print(labels[res[0]], labels[res[1]], labels[res[2]])

		#make sure there are enough key points in the fram
		if len(kp_frame) >= 10:
			#for this method just use the best prediction as the template
			template = cv.imread(trainData_dir+labels[res[0]], cv.IMREAD_GRAYSCALE)
			kp_list_template = kp_list[res[0]][1]
			flann_matches = flann.knnMatch(des_list[res[0]][1], des_frame, k=2)
			good_points = []
			for m,n in flann_matches:
				if m.distance < 0.6*n.distance:
					good_points.append(m)

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




