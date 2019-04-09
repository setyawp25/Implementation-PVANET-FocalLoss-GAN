# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.kitti_voc import kitti_voc
from datasets.udacity_kitti_voc import udacity_kitti_voc
import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up kittivoc_<datatype>_<split>
# Alice: 2017.9.30 >>
for datatype in ['track', 'detect']:
    for split in ['train', 'val', 'trainval']:
        name = 'kitti_{}_{}'.format(datatype, split)
        #kitti_voc: __init__(self, image_set=['train', 'test'], datatype=['track','detect'], devkit_path=None):
        __sets[name] = (lambda split=split, datatype=datatype: kitti_voc(split, datatype))
# Alice: 2017.9.30 <<

# Alice: 2017.10.16 >>
# Set up udacitykitvoc_<datatype>_<split>
for datatype in ['autti', 'crowdai']:
    for split in ['train', 'val', 'trainval']:
        name = 'udacitykitvoc_{}_{}'.format(datatype, split)
        #def __init__(self, image_set, datatype, devkit_path=None)
        __sets[name] = (lambda split=split, datatype=datatype: udacity_kitti_voc(split, datatype))

# Alice: 2017.10.16 <<

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
