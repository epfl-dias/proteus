#!/bin/bash
# This script adds a new user called 'temp_user' with the same UID/GID as the user which started the docker container.
# This is so that actions within the docker container can write to bind mounts, e.g cmake configure / build
USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

echo "Starting with UID: $USER_ID, GID: $GROUP_ID"
useradd -u $USER_ID -o -m temp_user
groupmod -g $GROUP_ID temp_user
export HOME=/home/temp_user
/usr/sbin/gosu temp_user /bin/bash