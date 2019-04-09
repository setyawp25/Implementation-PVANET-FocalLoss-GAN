set -x
set -e

GPU_ID=$1

./experiments/scripts/pvanet_gan_focal/fast_rcnn_std.sh ${GPU_ID} pvanet_gan_focal pascal_voc
./experiments/scripts/pvanet_gan_focal/fast_rcnn_adv_pretrain.sh ${GPU_ID} pvanet_gan_focal pascal_voc
./experiments/scripts/pvanet_gan_focal/fast_rcnn_adv.sh ${GPU_ID} pvanet_gan_focal pascal_voc_07 ./output/pvanet_gan_focal/voc_2007_trainval/pvanet_stage1_pretrain_iter_30000.caffemodel
./experiments/scripts/pvanet_gan_focal/fast_rcnn_adv.sh ${GPU_ID} pvanet_gan_focal pascal_voc_12 ./output/pvanet_gan_focal/voc_2007_trainval/pvanet_stage2_adv_iter_100000.caffemodel
