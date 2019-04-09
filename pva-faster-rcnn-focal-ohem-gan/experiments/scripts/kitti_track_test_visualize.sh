#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_alt_opt.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is only pascal_voc for now
#
# Example:
# python tools/test_pvanet_v3.py         --gpu 0         --testing-path ./data/KITTI/track/testing/image_02/0028         --result-path results/try         --net ./output/pvanet_gan_focal_kitti/kitti_detect_trainval/pvanet_stage2_adv_iter_100000.caffemodel         --def ./models/pvanet_gan_focal_kitti/stage0_standard/test.prototxt

# ./experiments/scripts/faster_rcnn_alt_opt.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
TESTPATH=$2
RESULTPATH=$3
CAFFEMODEL=$4
PROTOTXT=$5

for i in $(seq -f "%04g" 0 28)
do
    python ./tools/test_pvanet_v3.py --gpu ${GPU_ID} \
        --testing-path "${TESTPATH}/$i" \
        --result-path "${RESULTPATH}/$i" \
        --net "${CAFFEMODEL}" \
        --def "${PROTOTXT}"
    
    ffmpeg -r 15 -f image2 -s 1920x1080 -i "${RESULTPATH}/$i/%06d.png" -vf scale="1200:-2" -vcodec libx264 -crf 25  -pix_fmt yuv420p "${RESULTPATH}/$i.mp4"
done
# ./experiments/scripts/test_kitti_all.sh 1 /mnt/ext_disk1_2TB/vatic_datasets/kitti/track/testing/image_02 ./results/kitti/pva_focal ./output/pvanet_focal/pvanet_focal_0712_iter_150000.caffemodel ./models/pvanet/example_train/test.prototxt
