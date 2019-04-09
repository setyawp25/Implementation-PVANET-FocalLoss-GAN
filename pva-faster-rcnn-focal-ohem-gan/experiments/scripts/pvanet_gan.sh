./experiments/scripts/pvanet_gan/fast_rcnn_std.sh 0 pvanet_gan pascal_voc
./experiments/scripts/pvanet_gan/fast_rcnn_adv_pretrain.sh 0 pvanet_gan pascal_voc
./experiments/scripts/pvanet_gan/fast_rcnn_adv.sh 0 pvanet_gan pascal_voc_07 ./output/pvanet_gan/voc_2007_trainval/pvanet_stage1_pretrain_iter_30000.caffemodel
./experiments/scripts/pvanet_gan/fast_rcnn_adv.sh 0 pvanet_gan pascal_voc_12 ./output/pvanet_gan/voc_2012_trainval/pvanet_stage2_adv_iter_100000.caffemodel
