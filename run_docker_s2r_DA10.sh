#!/bin/bash

gpu=$1
docker run --rm -it --name "DA_10" --gpus "device=$gpu" --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/.Xauthority --hostname $(hostname) -v /mount/huch/Salinas/dataset_DA_10:/home/data/s2r -v /data/salinas_huch/sim2real_trainings/train_DA_10:/home/output openpcdet:sim2real bash

