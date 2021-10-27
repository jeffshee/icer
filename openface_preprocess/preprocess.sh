#!/bin/sh -x
# 実行時に"sh -x ./preprocess.sh"とすることで実行状況がわかりやすくなる

export DATA_MOUNT=$(pwd)/data
mkdir -p $DATA_MOUNT/processed # this is just to ensure this is writable by user
docker-compose up -d openface && sync # sync is to wait till service starts

for file in `\find $DATA_MOUNT -name '*.avi'`; do
    mkdir -p $DATA_MOUNT/processed/$(basename $file .avi)
    # For single face
    # docker exec -it openface FeatureExtraction -f $file -out_dir $DATA_MOUNT/processed/$(basename $file .avi)
    # For many faces
    docker exec -it openface FaceLandmarkVidMulti -f $file -out_dir $DATA_MOUNT/processed/$(basename $file .avi)
done

for file in `\find $DATA_MOUNT -name '*.mp4'`; do
    mkdir -p $DATA_MOUNT/processed/$(basename $file .mp4)
    # For single face
    # docker exec -it openface FeatureExtraction -f $file -out_dir $DATA_MOUNT/processed/$(basename $file .mp4)
    # For many faces
    docker exec -it openface FaceLandmarkVidMulti -f $file -out_dir $DATA_MOUNT/processed/$(basename $file .mp4)
done

docker exec -it openface chown -R $(id -u):$(id -u) $DATA_MOUNT # chown to current user
docker-compose down # stop service if you wish
