# Re_Identification_FYP
It is a reposity for my Final Year Project about Re_Identification.

## Introducation
Pedestrian re-identification, referred to as ReID for short in this paper, is a technology that uses computer vision technology to determine whether there is a specific pedestrian in an image or video sequence. ReID has been applied to many fields especially in the area of Intelligent Video Surveillance and Intelligent Security to reduce the burden on the public security system.The fact is that in surveillance video, due to the camera resolution and shooting Angle, it is usually impossible to get very high quality human face pictures. Besides, the method of finding targeted people should be cross-camera,that is, the same pedestrian pictures under different cameras should be retrieved. As a matter of fact, ReID becomes an important alternative technology when face recognition fails and cross-camera is needed. 
ReID has been working in academia for years, but it's only in the last few years, with the development of deep learning, that it has made a very big breakthrough. The main objective of my project is identifying the targeted people in any given video with only one image being inputted(One Sample), and it would be primarily based on the Deep Learning technology.

## How to Simply Use
1. Change the path in defaults.py in reid/config.   (C.TEST.WEIGHT and C.MODEL.PRETRAIN_PATH are crucial.)
2. Download pre-trained weights of YOLOv5.  (yolov5s.pt for example)
3. Download pre-trained weights of ReID.  (Here is the link: https://pan.baidu.com/s/16kyogSsGwL2VgMkNSn9-zg  , with code: f0g9) Thanks to the autor in https://github.com/songwsx/person_search_demo! 
4. Start.
