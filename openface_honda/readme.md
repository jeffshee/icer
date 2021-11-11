# OpenFace
paper: https://www.cl.cam.ac.uk/research/rainbow/projects/openface/wacv2016.pdf \
code: https://github.com/TadasBaltrusaitis/OpenFace


## Setup
```
docker run -it --rm --name openface algebr/openface:latest
```
https://github.com/TadasBaltrusaitis/OpenFace/wiki/Docker


## Output Format (CSV)
https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format


## Execution
https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments

Ex.
If output landmarks of the video which containes multiple faces.\
The results output to processed directory (default).
```
FaceLandmarkVidMulti -f "/my videos/video.avi"
```