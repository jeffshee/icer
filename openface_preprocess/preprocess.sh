#!/bin/sh -x

export DATA_MOUNT=$(pwd)/data
mkdir -p $DATA_MOUNT/processed # this is just to ensure this is writable by user
docker-compose up -d openface && sync # sync is to wait till service starts

# # For many faces
# build/bin/FaceLandmarkVidMulti -f "/docker_video_dir/video.avi"
# # For single face
# build/bin/FeatureExtraction -f "/docker_video_dir/video.avi"

# For many faces
docker exec -it openface FaceLandmarkVidMulti -f $DATA_MOUNT/test.mp4 -out_dir $DATA_MOUNT/processed
# # For single face
# build/bin/FeatureExtraction -f $DATA_MOUNT/test.mp4 -out_dir $DATA_MOUNT/processed
docker exec -it openface chown -R $(id -u):$(id -u) $DATA_MOUNT # chown to current user
docker-compose down # stop service if you wish
