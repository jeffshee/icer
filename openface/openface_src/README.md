
## Source Code: https://github.com/TadasBaltrusaitis/OpenFace

# OpenFace 2.2.0: a facial behavior analysis toolkit
Over the past few years, there has been an increased interest in automatic facial behavior analysis
and understanding. We present OpenFace – a tool intended for computer vision and machine learning
researchers, affective computing community and people interested in building interactive
applications based on facial behavior analysis. OpenFace is the ﬁrst toolkit capable of facial
landmark detection, head pose estimation, facial action unit recognition, and eye-gaze estimation
with available source code for both running and training the models. The computer vision algorithms
which represent the core of OpenFace demonstrate state-of-the-art results in all of the above
mentioned tasks. Furthermore, our tool is capable of real-time performance and is able to run from a
simple webcam without any specialist hardware.

OpenFace was originally developed by Tadas Baltrušaitis in collaboration with CMU MultiComp Lab led by Prof. Louis-Philippe Morency. Some of the original algorithms were created while at Rainbow Group, Cambridge University. The OpenFace library is still actively developed at the CMU MultiComp Lab in collaboration with Tadas Baltršaitis. Special thanks to researcher who helped developing, implementing and testing the algorithms present in OpenFace: Amir Zadeh and Yao Chong Lim on work on the CE-CLM model and Erroll Wood for the gaze estimation work.

## WIKI

**For instructions of how to install/compile/use the project please see [WIKI](https://github.com/TadasBaltrusaitis/OpenFace/wiki)**


#### Overall system

**OpenFace 2.0: Facial Behavior Analysis Toolkit**
Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency,
_IEEE International Conference on Automatic Face and Gesture Recognition_, 2018

#### Facial landmark detection and tracking

**Convolutional experts constrained local model for facial landmark detection**
A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency.
_Computer Vision and Pattern Recognition Workshops_, 2017

**Constrained Local Neural Fields for robust facial landmark detection in the wild**
Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency.
in IEEE Int. _Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge_, 2013.

#### Eye gaze tracking

**Rendering of Eyes for Eye-Shape Registration and Gaze Estimation**
Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling
in _IEEE International Conference on Computer Vision (ICCV)_, 2015

#### Facial Action Unit detection

**Cross-dataset learning and person-specific normalisation for automatic Action Unit detection**
Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson
in _Facial Expression Recognition and Analysis Challenge_,
_IEEE International Conference on Automatic Face and Gesture Recognition_, 2015

# Commercial license

For inquiries about the commercial licensing of the OpenFace toolkit please visit https://cmu.flintbox.com/#technologies/5c5e7fee-6a24-467b-bb5f-eb2f72119e59

# Final remarks

I did my best to make sure that the code runs out of the box but there are always issues and I would be grateful for your understanding that this is research code and a research project. If you encounter any problems/bugs/issues please contact me on github or by emailing me at tadyla@gmail.com for any bug reports/questions/suggestions. I prefer questions and bug reports on github as that provides visibility to others who might be encountering same issues or who have the same questions.

# Copyright

Copyright can be found in the Copyright.txt

You have to respect dlib, OpenBLAS, and OpenCV licenses.

Furthermore you have to respect the licenses of the datasets used for model training - https://github.com/TadasBaltrusaitis/OpenFace/wiki/Datasets