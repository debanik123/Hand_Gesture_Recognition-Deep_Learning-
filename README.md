# Hand_Gesture_Recognition-Deep_Learning-
I used Image processing and Deep learning to detect Hand Gesture.
This project has two parts first one is image processing and second one is deep learning. Lets talk about image processing 

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

