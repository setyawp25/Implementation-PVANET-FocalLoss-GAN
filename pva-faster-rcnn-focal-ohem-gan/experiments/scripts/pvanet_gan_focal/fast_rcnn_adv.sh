#!/bin/bash
# Usage:
# ./experiments/scripts/fast_rcnn_ohem.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/fast_rcnn_ohem.sh 0 VGG16 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3
PRETRAINED_MODEL=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc_07)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=100000
    ;;
  pascal_voc_12)
    TRAIN_IMDB="voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=100000
    ;;
  coco)
    echo "Support coming soon. Stay tuned!"
    exit
    # TRAIN_IMDB="coco_2014_train"
    # TEST_IMDB="coco_2014_minival"
    # PT_DIR="coco"
    # ITERS=280000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_stage2_adv.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


#time ./tools/train_net.py --gpu ${GPU_ID} \
#  --solver models/${NET}/stage2_adv/solver.prototxt \
#  --weights  ${PRETRAINED_MODEL}  \
#  --imdb ${TRAIN_IMDB} \
#  --iters ${ITERS} \
#  --cfg ./models/${NET}/cfgs/${NET}.yml \
#  ${EXTRA_ARGS}

#set +x
#NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
#set -x
#--net ${NET_FINAL} \
time
./tools/test_net.py --gpu ${GPU_ID} \
  --def ./models/${NET}/stage0_standard/test.prototxt \
  --net ./output/pvanet_gan_focal/voc_2012_trainval/pvanet_stage2_adv_iter_100000.caffemodel \
  --imdb ${TEST_IMDB} \
  --cfg ./models/${NET}/cfgs/${NET}.yml \
  ${EXTRA_ARGS}
