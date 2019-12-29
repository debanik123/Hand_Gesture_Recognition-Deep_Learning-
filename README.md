# Hand_Gesture_Recognition-Deep_Learning-
I used Image processing and Deep learning to detect Hand Gesture.
This project has two parts first one is image processing and second one is deep learning. 
#Let's talk about image processing 

	importing image processing and numerical python library.
	initialize webcam. 
	make a infinite loop. 
	read the video from webcam.
	then create a black frame with the same size of original frame.
	create a region of interest of the original frame.
	convert the colour from rgb to gray in-order to convert it into three channel to one channel.
	Apply gaussian blur on it with (7,7) kernel size for removing noise in it.
	Apply Canny Edge detection algorithm for getting edge
	pass the edge through finding contour for getting number of contours 
	maximize the contour based on there area.
	Approximate the contour. 
	Find the Convex hull from max contour.
	Calculate the areas of the convex hull and max contour.
	Calculate the area ratio 
	Pass the approx. value through the contour 
	calculate the convex hull defect.
	the find the Euclidean distance of them and put text one, two , three, four and five.
	finally release the web cam and destroy all windows.
	You will find the code in my github account.   

#Second part is about deep learning 
	I made own neural network model using keras and its awesome function

                layer	                Kernel_size	              activation
Conv(32)	                        (3,3)	            Linear
BatchNormalization(axis=-1)		
LeakyReLU(alpha=0.1)		
Maxpool()	Pool_size = (2,2)	
Dropout(0.3)		
Conv(64)	                        (3,3)	            Linear
BatchNormalization(axis=-1)		
LeakyReLU(alpha=0.1)		
Maxpool()	Pool_size = (2,2)	
Dropout(0.3)		
Conv(128)	                        (3,3)	            Linear
BatchNormalization(axis=-1)		
LeakyReLU(alpha=0.1)		
Maxpool()	Pool_size = (2,2)	
Dropout(0.4)		
Flatten()		
Dense(120)		            Linear
BatchNormalization(axis=-1)		
LeakyReLU(alpha=0.1)		
Dropout(0.3)		
Dense(60)		            Linear
BatchNormalization(axis=-1)		
LeakyReLU(alpha=0.1)		
Dropout(0.2)		
Dense(30)		            Linear
BatchNormalization(axis=-1)		
LeakyReLU(alpha=0.1)		
Dropout(0.1)		
Dense(10)		            Softmax

	The I compile the model with
loss	optimizer	 metrices
Categorical_cros_entropy	Adam	accuracy

	I took hand gesture data set from Kaggle “LeapGestureRecog”
	Lode this data set using os and its function and resize and and visualize them using matplot lib .
	split them using sklearn lib.
	I will fit the x_training and y_training  data into the neural network with 15 epochs along with 128 verbose and validate x_test and y_test data with it. 
	I plot the loss and accuracy graph and save the mdel
	I predict the x_test data along with the model and also compare the predicted and actual model out put.
	It almost gives 99.87% accuracy
	Finally, I apply the model on the real life data set and visualize the result. It seems like a good result. 
	I draw confusion matrix and gets some important results. The matrix proved it is a good model.
	If we use it in real time we may have faced some major problem that is low light situation and another is real time is not stable so our model could de unable to solve this, though we have a good tool open cv could often solved this problem. 
