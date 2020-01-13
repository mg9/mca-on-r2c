### https://github.com/rowanz/r2c/blob/master/utils/detector.py

class SimpleDetector(nn.Module):
    def __init__(self, pretrained=True, average_pool=True, semantic=True, final_dim=1024):
        """
        :param average_pool: whether or not to average pool the representations
        :param pretrained: Whether we need to load from scratch
        :param semantic: Whether or not we want to introduce the mask and the class label early on (default Yes)
        """
        super(SimpleDetector, self).__init__()

        backbone = resnet.resnet50(pretrained=true)
      
        for i in range(2, 4):
            getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
            getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
        # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        )

        self.roi_align = ROIAlign((7, 7), spatial_scale=1 / 16, sampling_ratio=0)
        self.object_embed = None
        self.mask_upsample = None

        after_roi_align = [backbone.layer4]
        self.final_dim = final_dim
        if average_pool:
            after_roi_align += [nn.AvgPool2d(7, stride=1), Flattener()]

        self.after_roi_align = torch.nn.Sequential(*after_roi_align)

        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2048, final_dim),
            torch.nn.ReLU(inplace=True),
        )

        self.regularizing_predictor = torch.nn.Linear(2048, 81)


    def forward(self,
                images: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                classes: torch.Tensor = None,
                segms: torch.Tensor = None,
                ):
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :return: object reps [batch_size, max_num_objects, dim]
        """
        # [batch_size, 2048, im_height // 32, im_width // 32
        img_feats = self.backbone(images)
        box_inds = box_mask.nonzero()
        assert box_inds.shape[0] > 0
        rois = torch.cat((
            box_inds[:, 0, None].type(boxes.dtype),
            boxes[box_inds[:, 0], box_inds[:, 1]],
        ), 1)

        # Object class and segmentation representations
        roi_align_res = self.roi_align(img_feats, rois)
        post_roialign = self.after_roi_align(roi_align_res)

        # Add some regularization, encouraging the model to keep giving decent enough predictions
        obj_logits = self.regularizing_predictor(post_roialign)
        obj_labels = classes[box_inds[:, 0], box_inds[:, 1]]
        cnn_regularization = F.cross_entropy(obj_logits, obj_labels, size_average=True)[None]

        feats_to_downsample = post_roialign
        roi_aligned_feats = self.obj_downsample(feats_to_downsample)

        print("ROI ALIGNED FEATS:", roi_aligned_feats)
        print("ROI ALIGNED FEATS shape:", roi_aligned_feats.shape)

        # Reshape into a padded sequence - this is expensive and annoying but easier to implement and debug...
        obj_reps = pad_sequence(roi_aligned_feats, box_mask.sum(1).tolist())
        return {
            'obj_reps_raw': post_roialign,
            'obj_reps': obj_reps,
            'obj_logits': obj_logits,
            'obj_labels': obj_labels,
            'cnn_regularization_loss': cnn_regularization
        }
