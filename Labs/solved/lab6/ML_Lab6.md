---
title: Object Detection with SSD
subtitle: Machine Learning, Lab 6
documentclass: scrartcl
fontsize: 12pt
---

# Objective

Students should understand the principle of and be able to use pretrained neural networks
for object detection, in particular the Single Shot Detector.

# Exercises

0. Download the laboratory files from [https://drive.google.com/open?id=1B98JshSA9cWrbYXCqO0InJgQGADcjHGP](https://drive.google.com/open?id=1B98JshSA9cWrbYXCqO0InJgQGADcjHGP)

1. Run the pretrained SSD detector (`SSD_Emulation_Script_2`) on the provided images (e.g. `Pedestrian.jpg` and the others) and observe the detected boxes.

    (Note: the model is available [here](https://github.com/fanyang16120029/SSD-Single-Shot-Detector-in-Matlab)).

1. Modify script `SSD_Emulation_Script_2` to make the pedestrian boxes in blue color.

2. Change the non-max suppression threshold parameter of the SSD detector and observe the changes.
	
3. Write a Matlab script which run the SSD detector on the video sequence `SecurityCamera.mp4`, and provide the following information for every frame of the video

    - how many cars are detected at every frame?
	- how many pedestrians are detected at every frame?
	- are there new cars appearing in the scene?
	- how many cars are moving?
	- if there is a moving object, is it moving left-to-right or right-to-left?
 
     Steps:
	 
      - write a Matlab script to read every frame from the video (look for documentation on `videoReader()` function)
	  - apply the SSD detector on every frame
	  - extract the information from the detected boxes
	  - new cars: if a car appears on the same location as in the previous frame, it is not new. Otherwise, it is a new car.
	  - for moving / non-moving objects, compare list of boxes during one second (last 25 frames). 
	  If an object appears in the same spot for more than 20 frames out of 25, then it is parked.
	  If an object's position keeps changing, it is moving.
	  - for direction detection: if the center of a box keeps moving in one direction, the object is moving.
