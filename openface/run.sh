#!/bin/sh -x

export DATA_MOUNT=$1

mkdir -p $DATA_MOUNT/processed # this is just to ensure this is writable by user
docker-compose up -d openface && sync # sync is to wait till service starts

for file in `\find $DATA_MOUNT -name 'reid*.mp4'`; do
    # For single face
    docker exec -it openface FeatureExtraction -f $file -out_dir $DATA_MOUNT/processed

    # delete unneccesarry files
    FILENAME=$(basename $file .mp4)
    docker exec -it openface rm -rf $DATA_MOUNT/processed/${FILENAME}_aligned/
    docker exec -it openface rm -rf $DATA_MOUNT/processed/${FILENAME}.hog
    docker exec -it openface rm -rf $DATA_MOUNT/processed/${FILENAME}_of_details.txt
    docker exec -it openface rm -rf $DATA_MOUNT/processed/${FILENAME}.avi
done

docker exec -it openface chown -R $(id -u):$(id -u) $DATA_MOUNT # chown to current user
docker-compose down # stop service if you wish
