# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from fast_rcnn.nms_wrapper import nms

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    if cfg.TRAIN.HAS_RPN:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        #blobs['im_info'] = np.array(
        #    [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        #    dtype=np.float32)
        blobs['im_info'] = np.array(
            [np.hstack((im_blob.shape[2], im_blob.shape[3], im_scales[0]))],
            dtype=np.float32)
    else: # not using RPN
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        #all_overlaps = []

        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes)

            # Add to RoIs blob
            rois = _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
            #all_overlaps = np.hstack((all_overlaps, overlaps))

        # For debug visualizations


        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob

        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = \
                np.array(bbox_inside_blob > 0).astype(np.float32)

    return blobs

def get_allrois_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    if cfg.TRAIN.HAS_RPN:
        # Doesn't support RPN yet.
        #assert False
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        #blobs['im_info'] = np.array(
        #    [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        #    dtype=np.float32)
        blobs['im_info'] = np.array(
            [np.hstack((im_blob.shape[2], im_blob.shape[3], im_scales[0]))],
            dtype=np.float32)
    else: # not using RPN
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)

        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
                = _all_rois(roidb[im_i], num_classes)

            # Add to RoIs blob
            rois = _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))

        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob

        # For debug visualizations
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = \
                np.array(bbox_inside_blob > 0).astype(np.float32)

    return blobs


#def get_ohem_minibatch(loss, rois, labels, bbox_targets=None,
#                       bbox_inside_weights=None, bbox_outside_weights=None):
def get_ohem_minibatch(im_data, loss, rois, labels, bbox_targets=None,
                       bbox_inside_weights=None, bbox_outside_weights=None):
    """Given rois and their loss, construct a minibatch using OHEM."""
    loss = np.array(loss)
    #print 'len(loss) = ', len(loss)
    if cfg.TRAIN.OHEM_USE_NMS:
        # Do NMS using loss for de-dup and diversity
        keep_inds = []
        nms_thresh = cfg.TRAIN.OHEM_NMS_THRESH
        source_img_ids = [roi[0] for roi in rois]
        for img_id in np.unique(source_img_ids):
            for label in np.unique(labels):
                sel_indx = np.where(np.logical_and(labels == label, \
                                    source_img_ids == img_id))[0]
                if not len(sel_indx):
                    continue
                boxes = np.concatenate((rois[sel_indx, 1:],
                        loss[sel_indx][:,np.newaxis]), axis=1).astype(np.float32)
                keep_inds.extend(sel_indx[nms(boxes, nms_thresh)])

        #print 'keep_inds_length = ', len(keep_inds)
        hard_keep_inds = select_hard_examples(loss[keep_inds])
        hard_inds = np.array(keep_inds)[hard_keep_inds]
    else:
        hard_inds = select_hard_examples(loss)
    #print "\nRoI examples #: %d\tSelect Hard Examples: %d" % (len(loss), len(hard_inds))
    #print "\t", [labels[hard_ind] for hard_ind in hard_inds]
    #print "\t", [loss[hard_ind] for hard_ind in hard_inds]

    # Alice: using both GAN and OHEM to train fc layers (make features as stack) 2018.01.24 >>
    if cfg.TRAIN.USE_OHEM_GAN:
        blobs = {'rois_hard': rois[hard_inds, :].copy(),
                 'labels_hard': np.hstack((labels[hard_inds].copy(), labels))}
    else:
        blobs = {'rois_hard': rois[hard_inds, :].copy(),
                 'labels_hard': labels[hard_inds].copy()}
    # Alice: using both GAN and OHEM to train fc layers (make features as stack) 2018.01.24 <<

    #print "labels_hard: ", labels[hard_inds].shape, "labels_total: ", np.hstack((labels[hard_inds].copy(), labels)).shape

    if bbox_targets is not None:
        assert cfg.TRAIN.BBOX_REG
        # original: without combining with ASDN
        #blobs['bbox_targets_hard'] = vstack(bbox_targets[hard_inds, :].copy())
        #blobs['bbox_inside_weights_hard'] = bbox_inside_weights[hard_inds, :].copy()
        #blobs['bbox_outside_weights_hard'] = bbox_outside_weights[hard_inds, :].copy()

        # Alice: using both GAN and OHEM to train fc layers (make features as stack) 2018.01.24 >>
        if cfg.TRAIN.USE_OHEM_GAN:
            blobs['bbox_targets_hard'] = np.vstack((bbox_targets[hard_inds, :].copy(), bbox_targets))
            blobs['bbox_inside_weights_hard'] = np.vstack((bbox_inside_weights[hard_inds, :].copy(), bbox_inside_weights))
            blobs['bbox_outside_weights_hard'] = np.vstack((bbox_outside_weights[hard_inds, :].copy(), bbox_outside_weights))
        else:
            blobs['bbox_targets_hard'] = vstack(bbox_targets[hard_inds, :].copy())
            blobs['bbox_inside_weights_hard'] = bbox_inside_weights[hard_inds, :].copy()
            blobs['bbox_outside_weights_hard'] = bbox_outside_weights[hard_inds, :].copy()
        # Alice: using both GAN and OHEM to train fc layers (make features as stack) 2018.01.24 <<


    #print 'rois_length', len(rois)
    #print 'rois_hard_length = ', len(blobs['rois_hard'])

    #print 'rois_hard', rois[hard_inds, :]
    #print 'labels_hard', labels[hard_inds]
    # debug for visualization for rois
    #_vis_minibatch(im_data, blobs['rois_hard'], blobs['labels_hard'], [])

    return blobs

def select_hard_examples(loss):
    """Select hard rois."""
    # Sort and select top hard examples.
    sorted_indices = np.argsort(loss)[::-1]
    #print 'loss', loss
    #print 'sorted_indices = ', sorted_indices
    #print(len(loss)/4, cfg.TRAIN.BATCH_SIZE/4, ' := ', np.minimum(len(loss)/4, cfg.TRAIN.BATCH_SIZE/4))
    hard_keep_inds = sorted_indices[0:np.minimum((len(loss)), (cfg.TRAIN.BATCH_SIZE))]
    #hard_keep_inds = sorted_indices[0:np.minimum((len(loss)/3), (cfg.TRAIN.BATCH_SIZE/3))]
    return hard_keep_inds

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

def _all_rois(roidb, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # To use custom cfg.TRAIN.BG_THRESH_LO, comment the following assert.
    assert cfg.TRAIN.BG_THRESH_LO == 0.0, \
        "OHEM works best with BG_THRESH_LO = 0.0 (current value is {}).".format(cfg.TRAIN.BG_THRESH_LO)

    # Select foreground (background) RoIs.
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    bg_inds = np.where(overlaps < cfg.TRAIN.BG_THRESH_HI)[0]

    # All RoIs.
    keep_inds = np.append(fg_inds, bg_inds)

    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[len(fg_inds):] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE,  cfg.TRAIN.SCALE_MULTIPLE_OF)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        start = int(start)
        end = int(end)
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = int(rois[0])
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls#, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
