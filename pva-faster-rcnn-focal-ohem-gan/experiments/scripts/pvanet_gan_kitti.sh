set -x
set -e

GPU_ID=$1

#./experiments/scripts/pvanet_gan_kitti/fast_rcnn_std.sh ${GPU_ID} pvanet_gan_kitti kitti
#./experiments/scripts/pvanet_gan_kitti/fast_rcnn_adv_pretrain.sh ${GPU_ID} pvanet_gan_kitti kitti
./experiments/scripts/pvanet_gan_kitti/fast_rcnn_adv.sh ${GPU_ID} pvanet_gan_kitti kitti
