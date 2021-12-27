#!/bin/bash
DATE=`date -d now`
FUNC=main
SAVE="save/${FUNC}"
CONFIG='configs/single2single.yaml'
gpu_id=0
CUDA_VISIBLE_DEVICES=${gpu_id}; python train.py --net-savedir ${SAVE} --config ${CONFIG} --func ${FUNC} 




