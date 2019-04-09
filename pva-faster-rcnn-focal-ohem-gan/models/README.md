## Model Zoo

### PVANET trained with OHEM method

Training command:
```
./tools/train_net.py \
    --gpu 1 \
    --solver ./models/pvanet_ohem/faster_rcnn_end2end/solver.prototxt \
    --weights ./models/pvanet/pva9.1/PVA9.1_ImgNet_COCO_VOC0712plus.caffemodel \
    --imdb voc_2007_trainval \
    --iters 10 \
    --cfg ./models/pvanet_ohem/cfgs/faster_rcnn_end2end.yml
```
```
./tools/train_net.py \
    --gpu 0 \
    --solver ./models/pvanet/example_train/solver.prototxt \
    --weights ./models/pvanet/pva9.1/PVA9.1_ImgNet_COCO_VOC0712plus.caffemodel \
    --imdb voc_2007_trainval \
    --iters 10 \
    --cfg ./models/pvanet/cfgs/train.yml
```
```
./tools/train_faster_rcnn_alt_opt.py --gpu 0 \
  --net_name pvanet_ohem \
  --weights ./models/pvanet/pva9.1/PVA9.1_ImgNet_COCO_VOC0712plus.caffemodel \
  --imdb voc_2007_trainval \
  --cfg ./models/pvanet_ohem/cfgs/faster_rcnn_alt_opt.yml
```
```
./tools/train_faster_rcnn_alt_opt.py --gpu 0 \
  --net_name pvanet_ohem \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb voc_2007_trainval \
  --cfg ./models/pvanet_ohem/cfgs/faster_rcnn_alt_opt.yml
```
```
./tools/train_net.py --gpu 0 \
  --solver ./models/pvanet_ohem/faster_rcnn_end2end/solver.prototxt \
  --weights ./output/pvanet_faster_rcnn_ohem_end2end/voc_2007_trainval/pvanet_frcnn_ohem_end2end_iter_150000.caffemodel \
  --imdb voc_2012_trainval \
  --iters 30000 \
  --cfg ./models/pvanet_ohem/cfgs/faster_rcnn_end2end.yml
```
```
./tools/test_net.py --gpu 0 \
  --def ./models/pvanet_ohem_gan/stage0_standard/test.prototxt \
  --net ./output/pvanet_ohem_gan/voc_2007_trainval/pvanet_stage2_adv_parellel_iter_100000.caffemodel \
  --imdb voc_2007_test \
  --cfg ./models/pvanet_ohem_gan/cfgs/pvanet_ohem_gan.yml
```
```
./tools/test_net.py --gpu 0 \
  --def ./models/pvanet_ohem_gan/stage0_standard/test.prototxt \
  --net ./output/pvanet_ohem_gan/pvanet_stage1_pretrain_iter_50000.caffemodel \
  --imdb voc_2007_test \
  --cfg ./models/pvanet_gan_focal/cfgs/pvanet_gan_focal.yml
```
`py-faster-rcnn` commit: 68eec95

test-dev2015 results
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.242
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.235
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.077
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.264
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.371
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.238
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.346
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.120
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.385
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.544
```

test-standard2015 results
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.242
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.234
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.072
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.264
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.369
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.238
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.341
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.115
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.389
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.544
```
