python python_utils/do_net_surgery.py \
  --out_net_def models/pvanet_ohem_gan/stage2_adv_parellel/train.prototxt \
  --net_surgery_json models/pvanet_ohem_gan/stage2_adv_parellel/init_weights2.json \
  --out_net_file output/pvanet_ohem_gan/voc_2007_trainval/pvanet_stage0_stage1_integration_initialmodel.caffemodel
