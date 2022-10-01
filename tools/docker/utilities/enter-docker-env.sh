#!/bin/bash
# remove this line once we fix pelago-build
rm /usr/local/cuda && ln -s /usr/local/cuda-11.3 /usr/local/cuda
# This script adds a new user called 'user' with the same UID/GID as the user which started the docker container.
# This is so that actions within the docker container can write to bind mounts, e.g cmake configure / build
USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

echo "Starting with UID: $USER_ID, GID: $GROUP_ID"
useradd -u $USER_ID -o -m user
groupmod -g $GROUP_ID user
export HOME=/home/user
/usr/sbin/gosu user /bin/bash