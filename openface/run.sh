#!/bin/sh -x
# 実行時に"sh -x ./preprocess.sh"とすることで実行状況がわかりやすくなる

export DATA_MOUNT=$1
echo $1
# export DATA_MOUNT="/home/icer/Project/openface_dir2/2020-01-23_Take01_copycopy_3_3_5people/split_video_1/reid/"

mkdir -p $DATA_MOUNT/processed # this is just to ensure this is writable by user
cd openface/openface_src
docker-compose up -d openface && sync # sync is to wait till service starts

for file in `\find $DATA_MOUNT -name 'reid*.mp4'`; do
    # For single face
    docker exec -it openface FeatureExtraction -f $file -out_dir $DATA_MOUNT/processed
done

docker exec -it openface chown -R $(id -u):$(id -u) $DATA_MOUNT # chown to current user
docker-compose down # stop service if you wish
