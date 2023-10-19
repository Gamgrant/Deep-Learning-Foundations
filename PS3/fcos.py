"""
This module contains classes and functions that are used for FCOS, a  one-stage 
object detector. You have to implement the functions here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""


import math
from typing import Dict, List, Optional

import torch
from cs639.loading import *
from torch import device, dtype, nn, scalar_tensor
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import torchvision
from torchvision.models import feature_extraction
from torchvision.ops import sigmoid_focal_loss


def hello_fcos():
    print("Hello from fcos.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights for faster convergence.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create three "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create three "output" 3x3 conv moduels to transform the merged#
        # FPN features to output (p3, p4, p5) features.                      #
        # Specifically, let us use (p3+p4+p5, p4+p5, p5) to denote the merged#
        # features. Then we use a single 3x3 conv layer for  p4+p5 and p5, while  #
        # use a (Conv, BN, ReLU, Conv) structure for p3+p4+p5. Here, all the #
        # input and output channels for conv layers should be                #
        # self.out_channels 

        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 moduels.                         #
        
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these modules.                         #
        
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add three lateral 1x1 conv and three output 3x3 moduels.
        self.fpn_params = nn.ModuleDict()
        # Replace "PASS" statement with your code
        # print(dummy_out_shapes)
        # print(dummy_out_shapes[0][1])
        # print(dummy_out_shapes[0][1][1])
        self.fpn_params['c3_lateral'] = nn.Conv2d(dummy_out_shapes[0][1][1], self.out_channels, kernel_size = 1)
        self.fpn_params['c4_lateral'] = nn.Conv2d(dummy_out_shapes[1][1][1], self.out_channels, kernel_size = 1)
        self.fpn_params['c5_lateral'] = nn.Conv2d(dummy_out_shapes[2][1][1], self.out_channels, kernel_size = 1)

        self.fpn_params['c5_conv3'] = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.fpn_params['c4_conv3'] = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.fpn_params['c3_conv3'] = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(self.out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
            )

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using Conv features    #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################
        # Replace "PASS" statement with your code
        c3_lateral = self.fpn_params['c3_lateral'](backbone_feats['c3'])
        c4_lateral = self.fpn_params['c4_lateral'](backbone_feats['c4'])
        c5_lateral = self.fpn_params['c5_lateral'](backbone_feats['c5'])

        # print(c4_lateral.shape)
        c5_lateral_unsampled = F.interpolate(c5_lateral, (c4_lateral.shape[2], c4_lateral.shape[3]), mode='nearest')
        c4_lateral_unsampled = F.interpolate(c4_lateral, (c3_lateral.shape[2], c3_lateral.shape[3]), mode='nearest')
        temp5 = c5_lateral
        temp4 = c5_lateral_unsampled + c4_lateral
        temp3 = c4_lateral_unsampled + c3_lateral

        fpn_feats['p5'] = self.fpn_params['c5_conv3'](c5_lateral)
        fpn_feats['p4'] = self.fpn_params['c4_conv3'](temp4)
        fpn_feats['p3'] = self.fpn_params['c3_conv3'](temp3)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "PASS" statement with your code
        # print(feat_shape)
        H, W = feat_shape[2:]
        x = torch.arange(0.5, W + 0.5, step=1, dtype=dtype, device=device) * level_stride
        y = torch.arange(0.5, H + 0.5, step=1, dtype=dtype, device=device) * level_stride
        # print(x.shape)
        grid_x, grid_y = torch.meshgrid(x, y)
        # print(grid_x.shape)
        # print(grid_y.shape)
        # print(grid_x.unsqueeze(-1).shape)
        # print(torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=-1).shape)
        location_coords[level_name] = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=-1).reshape(-1, 2)
        # print(location_coords[level_name].shape)

        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "PASS" statement with your code
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    _, _boxes = scores.sort(0, descending=True)

    keep = []

    while _boxes.numel() > 0:
        if _boxes.numel() == 1:
            i = _boxes.item()
            keep.append(i)
            break
        else:
            i = _boxes[0].item()
            keep.append(i)

        xx1 = torch.max(x1[_boxes[1:]], x1[i])
        yy1 = torch.max(y1[_boxes[1:]], y1[i])
        xx2 = torch.min(x2[_boxes[1:]], x2[i])
        yy2 = torch.min(y2[_boxes[1:]], y2[i])

        # side1 = torch.min(torch.tensor(0), xx2-xx1)
        # side2 = torch.min(torch.tensor(0), yy2-yy1)
        side1 = (xx2-xx1).clamp(min=0)
        side2 = (yy2-yy1).clamp(min=0)

        test = (xx2-xx1).clamp(min=0)
        intersection = side1 * side2

        Union = (areas[i]+areas[_boxes[1:]]-intersection)
        iou = intersection / Union
        idx = (iou <= iou_threshold).nonzero().squeeze()
        if idx.numel() == 0:
            break
        _boxes = _boxes[idx+1]
    keep = torch.Tensor(keep).to(device=scores.device).long()  
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]





class FCOSPredictionNetwork(nn.Module):
    """
    FCOS prediction network that accepts FPN feature maps from different levels
    and makes three predictions at every location: bounding boxes, class ID and
    centerness. This module contains a "stem" of convolution layers, along with
    one final layer per prediction. For a visual depiction, see Figure 2 (right
    side) in FCOS paper: https://arxiv.org/abs/1904.01355

    We will use feature maps from FPN levels (P3, P4, P5) and exclude (P6, P7).
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Number of object classes for classification.
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN, since the head directly
                operates on them.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
        """
        super().__init__()

        ######################################################################
        # TODO: Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules. Note there are two separate stems for class and
        # box stem. The prediction layers for box regression and centerness
        # operate on the output of `stem_box`.
        # See FCOS figure again; both stems are identical.
        #
        # Use `in_channels` and `stem_channels` for creating these layers, the
        # docstring above tells you what they mean. Initialize weights of each
        # conv layer from a normal distribution with mean = 0 and std dev = 0.01
        # and all biases with zero. Use conv stride = 1 and zero padding such
        # that size of input features remains same: remember we need predictions
        # at every location in feature map, we shouldn't "lose" any locations.
        ######################################################################
        # Fill these.
        stem_cls = []
        stem_box = []
        # Replace "PASS" statement with your code
        prev = in_channels
        for out in stem_channels:
          conv_cls = nn.Conv2d(prev, out, 3, padding=1)
          nn.init.normal_(conv_cls.weight, mean=0.0, std=0.01)
          nn.init.constant_(conv_cls.bias, 0.0)
          stem_cls.append(conv_cls)
          relu = nn.ReLU()
          stem_cls.append(relu)
          prev = out

        prev = in_channels
        for out in stem_channels:
          conv_box = nn.Conv2d(prev, out, 3, padding=1)
          nn.init.normal_(conv_box.weight, mean=0.0, std=0.01)
          nn.init.constant_(conv_box.bias, 0.0)
          stem_box.append(conv_box)
          relu = nn.ReLU()
          stem_box.append(relu)          
          prev = out

        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        ######################################################################
        # TODO: Create THREE 3x3 conv layers for individually predicting three
        # things at every location of feature map:
        #     1. object class logits (`num_classes` outputs)
        #     2. box regression deltas (4 outputs: LTRB deltas from locations)
        #     3. centerness logits (1 output)
        #
        # Class probability and actual centerness are obtained by applying
        # sigmoid activation to these logits. However, DO NOT initialize those
        # modules here. This module should always output logits; PyTorch loss
        # functions have numerically stable implementations with logits. During
        # inference, logits are converted to probabilities by applying sigmoid,
        # BUT OUTSIDE this module.
        #
        ######################################################################

        # Replace these lines with your code, keep variable names unchanged.
        self.pred_cls = None  # Class prediction conv
        self.pred_box = None  # Box regression conv
        self.pred_ctr = None  # Centerness conv

        # Replace "PASS" statement with your code
        self.pred_cls = nn.Conv2d(prev, num_classes, 3, padding=1)
        nn.init.normal_(self.pred_cls.weight, 0, 0.01)
        nn.init.constant_(self.pred_cls.bias, 0.0)

        self.pred_box = nn.Conv2d(prev, 4, 3, padding=1)
        nn.init.normal_(self.pred_box.weight, 0, 0.01)
        nn.init.constant_(self.pred_box.bias, 0.0)

        self.pred_ctr = nn.Conv2d(prev, 1, 3, padding=1)
        nn.init.normal_(self.pred_ctr.weight, 0, 0.01)
        nn.init.constant_(self.pred_ctr.bias, 0.0)        
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # OVERRIDE: Use a negative bias in `pred_cls` to improve training
        # stability. Without this, the training will most likely diverge.
        # STUDENTS: You do not need to get into details of why this is needed.
        torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict the desired outputs at every location
        (as described above). Format them such that channels are placed at the
        last dimension, and (H, W) are flattened (having channels at last is
        convenient for computing loss as well as perforning inference).

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}. Each
                tensor will have shape `(batch_size, fpn_channels, H, W)`. For an
                input (224, 224) image, H = W are (28, 14, 7) for (p3, p4, p5).

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Classification logits: `(batch_size, H * W, num_classes)`.
            2. Box regression deltas: `(batch_size, H * W, 4)`
            3. Centerness logits:     `(batch_size, H * W, 1)`
        """

        ######################################################################
        # TODO: Iterate over every FPN feature map and obtain predictions using
        # the layers defined above. Remember that prediction layers of box
        # regression and centerness will operate on output of `stem_box`,
        # and classification layer operates separately on `stem_cls`.
        #
        # CAUTION: The original FCOS model uses shared stem for centerness and
        # classification. Recent follow-up papers commonly place centerness and
        # box regression predictors with a shared stem, which we follow here.
        #
        # DO NOT apply sigmoid to classification and centerness logits.
        ######################################################################
        # Fill these with keys: {"p3", "p4", "p5"}, same as input dictionary.
        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}

        # Replace "PASS" statement with your code

        # bug!!!!!!!! problem in H and W
        # for key in feats_per_fpn_level.keys():
        #   class_logits[key] = self.pred_cls(self.stem_cls(feats_per_fpn_level[key])) # B N H W
        #   B, _, H, W = class_logits[key].shape
        #   class_logits[key] = class_logits[key].reshape(B, -1, H*W).transpose(1,2)

        #   boxreg_deltas[key] = self.pred_box(self.stem_box(feats_per_fpn_level[key]))
        #   boxreg_deltas[key] = boxreg_deltas[key].reshape(B, -1, H*W).transpose(1,2)
        #   centerness_logits[key] = self.pred_ctr(self.stem_box(feats_per_fpn_level[key]))
        #   centerness_logits[key] = centerness_logits[key].reshape(B, -1, H*W).transpose(1,2)

        for key in feats_per_fpn_level.keys():
          class_logits[key] = self.pred_cls(self.stem_cls(feats_per_fpn_level[key])) # B N H W
          B, _, H, W = class_logits[key].shape
          class_logits[key] = class_logits[key].permute(0, 3, 2, 1).reshape(B, H*W, -1)

          boxreg_deltas[key] = self.pred_box(self.stem_box(feats_per_fpn_level[key]))

          boxreg_deltas[key] = boxreg_deltas[key].permute(0, 3, 2, 1).reshape(B, H*W, -1)
          centerness_logits[key] = self.pred_ctr(self.stem_box(feats_per_fpn_level[key]))

          centerness_logits[key] = centerness_logits[key].permute(0, 3, 2, 1).reshape(B, H*W, -1)        
    
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return [class_logits, boxreg_deltas, centerness_logits]


@torch.no_grad()
def fcos_match_locations_to_gt(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    gt_boxes: torch.Tensor,
) -> TensorDict:
    """
    Match centers of the locations of FPN feature with a set of GT bounding
    boxes of the input image. Since our model makes predictions at every FPN
    feature map location, we must supervise it with an appropriate GT box.
    There are multiple GT boxes in image, so FCOS has a set of heuristics to
    assign centers with GT, which we implement here.

    NOTE: This function is NOT BATCHED. Call separately for GT box batches.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)` where H = W is the size of feature map.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `fcos.py` for more details.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`. In this
            codebase, this tensor is directly served by the dataloader.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(N, 5)` GT boxes, one for each center. They are
            one of M input boxes, or a dummy box called "background" that is
            `(-1, -1, -1, -1, -1)`. Background indicates that the center does
            not belong to any object.
    """

    matched_gt_boxes = {
        level_name: None for level_name in locations_per_fpn_level.keys()
    }

    # Do this matching individually per FPN level.
    for level_name, centers in locations_per_fpn_level.items():

        # Get stride for this FPN level.
        stride = strides_per_fpn_level[level_name]

        x, y = centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)

        # Pairwise distance between every feature center and GT box edges:
        # shape: (num_gt_boxes, num_centers_this_level, 4)
        pairwise_dist = pairwise_dist.permute(1, 0, 2)

        # The original FCOS anchor matching rule: anchor point must be inside GT.
        match_matrix = pairwise_dist.min(dim=2).values > 0

        # Multilevel anchor matching in FCOS: each anchor is only responsible
        # for certain scale range.
        # Decide upper and lower bounds of limiting targets.
        pairwise_dist = pairwise_dist.max(dim=2).values

        lower_bound = stride * 4 if level_name != "p3" else 0
        upper_bound = stride * 8 if level_name != "p5" else float("inf")
        match_matrix &= (pairwise_dist > lower_bound) & (
            pairwise_dist < upper_bound
        )

        # Match the GT box with minimum area, if there are multiple GT matches.
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
            gt_boxes[:, 3] - gt_boxes[:, 1]
        )

        # Get matches and their labels using match quality matrix.
        match_matrix = match_matrix.to(torch.float32)
        match_matrix *= 1e8 - gt_areas[:, None]

        # Find matched ground-truth instance per anchor (un-matched = -1).
        match_quality, matched_idxs = match_matrix.max(dim=0)
        matched_idxs[match_quality < 1e-5] = -1

        # Anchors with label 0 are treated as background.
        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        matched_boxes_this_level[matched_idxs < 0, :] = -1

        matched_gt_boxes[level_name] = matched_boxes_this_level

    return matched_gt_boxes


def fcos_get_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Compute distances from feature locations to GT box edges. These distances
    are called "deltas" - `(left, top, right, bottom)` or simply `LTRB`. The
    feature locations and GT boxes are given in absolute image co-ordinates.

    These deltas are used as targets for training FCOS to perform box regression
    and centerness regression. They must be "normalized" by the stride of FPN
    feature map (from which feature locations were computed, see the function
    `get_fpn_location_coords`). If GT boxes are "background", then deltas must
    be `(-1, -1, -1, -1)`.

    NOTE: This transformation function should not require GT class label. Your
    implementation must work for GT boxes being `(N, 4)` or `(N, 5)` tensors -
    without or with class labels respectively. You may assume that all the
    background boxes will be `(-1, -1, -1, -1)` or `(-1, -1, -1, -1, -1)`.

    Args:
        locations: Tensor of shape `(N, 2)` giving `(xc, yc)` feature locations.
        gt_boxes: Tensor of shape `(N, 4 or 5)` giving GT boxes.
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Tensor of shape `(N, 4)` giving deltas from feature locations, that
            are normalized by feature stride.
    """
    ##########################################################################
    # TODO: Implement the logic to get deltas from feature locations.        #
    ##########################################################################
    # Set this to Tensor of shape (N, 4) giving deltas (left, top, right, bottom)
    # from the locations to GT box edges, normalized by FPN stride.
    deltas = None

    # Replace "PASS" statement with your code
    # print(locations)
    # print(locations[:, 0]) -> xc
    # print(gt_boxes[:, 0]) -> x1
    deltas = torch.zeros((gt_boxes.shape[0], 4), dtype=gt_boxes.dtype, device=gt_boxes.device)
    l = (locations[:, 0] - gt_boxes[:, 0])/stride
    t = (locations[:, 1] - gt_boxes[:, 1])/stride
    r = (gt_boxes[:, 2] - locations[:, 0])/stride
    b = (gt_boxes[:, 3] - locations[:, 1])/stride
    deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3] = l, t, r, b

    # Background deltas should be all -1
    deltas[gt_boxes[:,0] == -1] = -1  #no GT boxes for the current image

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return deltas


def fcos_apply_deltas_to_locations(
    deltas: torch.Tensor, locations: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Implement the inverse of `fcos_get_deltas_from_locations` here:

    Given edge deltas (left, top, right, bottom) and feature locations of FPN, get
    the resulting bounding box co-ordinates by applying deltas on locations. This
    method is used for inference in FCOS: deltas are outputs from model, and
    applying them to anchors will give us final box predictions.

    Recall in above method, we were required to normalize the deltas by feature
    stride. Similarly, we have to un-normalize the input deltas with feature
    stride before applying them to locations, because the given input locations are
    already absolute co-ordinates in image dimensions.

    Args:
        deltas: Tensor of shape `(N, 4)` giving edge deltas to apply to locations.
        locations: Locations to apply deltas on. shape: `(N, 2)`
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Same shape as deltas and locations, giving co-ordinates of the
            resulting boxes `(x1, y1, x2, y2)`, absolute in image dimensions.
    """
    ##########################################################################
    # TODO: Implement the transformation logic to get boxes.                 #
    #                                                                        #
    # NOTE: The model predicted deltas MAY BE negative, which is not valid   #
    # for our use-case because the feature center must lie INSIDE the final  #
    # box. Make sure to clip them to zero.                                   #
    ##########################################################################
    output_boxes = None
    # Replace "PASS" statement with your code
    output_boxes = torch.zeros_like(deltas)
    deltas = deltas.clamp(min=0)
    l, t, r, b = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    output_boxes[:, 0] = locations[:, 0] - l * stride 
    output_boxes[:, 1] = locations[:, 1] - t * stride 
    output_boxes[:, 2] = locations[:, 0] + r * stride
    output_boxes[:, 3] = locations[:, 1] + b * stride

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return output_boxes


def fcos_make_centerness_targets(deltas: torch.Tensor):
    """
    Given LTRB deltas of GT boxes, compute GT targets for supervising the
    centerness regression predictor. See `fcos_get_deltas_from_locations` on
    how deltas are computed. If GT boxes are "background" => deltas are
    `(-1, -1, -1, -1)`, then centerness should be `-1`.

    For reference, centerness equation is available in FCOS paper
    https://arxiv.org/abs/1904.01355 (Equation 3).

    Args:
        deltas: Tensor of shape `(N, 4)` giving LTRB deltas for GT boxes.

    Returns:
        torch.Tensor
            Tensor of shape `(N, )` giving centerness regression targets.
    """
    ##########################################################################
    # TODO: Implement the centerness calculation logic.                      #
    ##########################################################################
    centerness = None
    # Replace "PASS" statement with your code
    # print(deltas)
    # print(deltas[:, 0]) -> left
    # print(deltas[:, 1]) -> top 
    # print(deltas[:, 2]) -> right
    # print(deltas[:, 3]) -> bottom
    # print(deltas[:,])

    dividend = (torch.min(deltas[:, 0], deltas[:, 2]) * torch.min(deltas[:, 1], deltas[:, 3]))
    divisor = (torch.max(deltas[:, 0], deltas[:, 2]) * torch.max(deltas[:, 1], deltas[:, 3]))
    centerness = torch.sqrt(dividend/divisor)
    # the lable is -1
    centerness[deltas[:,-1] == -1] = -1
    # print(deltas[:,-1])
    # print(centerness)

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return centerness


class FCOS(nn.Module):
    """
    FCOS: Fully-Convolutional One-Stage Detector

    This class puts together everything you implemented so far. It contains a
    backbone with FPN, and prediction layers (head). It computes loss during
    training and predicts boxes during inference.
    """

    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        super().__init__()
        self.num_classes = num_classes

        ######################################################################
        # TODO: Initialize backbone and prediction network using arguments.  #
        ######################################################################
        # Feel free to delete these two lines: (but keep variable names same)
        self.backbone = None
        self.pred_net = None
        # Replace "PASS" statement with your code
        self.backbone = DetectorBackboneWithFPN(fpn_channels)
        self.pred_net = FCOSPredictionNetwork(num_classes, fpn_channels, stem_channels)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Averaging factor for training loss; EMA of foreground locations.
        # STUDENTS: See its use in `forward` when you implement losses.
        self._normalizer = 150  # per image

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Args:
            images: Batch of images, tensors of shape `(B, C, H, W)`.
            gt_boxes: Batch of training boxes, tensors of shape `(B, N, 5)`.
                `gt_boxes[i, j] = (x1, y1, x2, y2, C)` gives information about
                the `j`th object in `images[i]`. The position of the top-left
                corner of the box is `(x1, y1)` and the position of bottom-right
                corner of the box is `(x2, x2)`. These coordinates are
                real-valued in `[H, W]`. `C` is an integer giving the category
                label for this bounding box. Not provided during inference.
            test_score_thresh: During inference, discard predictions with a
                confidence score less than this value. Ignored during training.
            test_nms_thresh: IoU threshold for NMS during inference. Ignored
                during training.

        Returns:
            Losses during training and predictions during inference.
        """

        ######################################################################
        # TODO: Process the image through backbone, FPN, and prediction head #
        # to obtain model predictions at every FPN location.                 #
        # Get dictionaries of keys {"p3", "p4", "p5"} giving predicted class #
        # logits, deltas, and centerness.                                    #
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = None, None, None
        # Replace "PASS" statement with your code
        fpn_feats = self.backbone(images)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net(fpn_feats)

        ######################################################################
        # TODO: Get absolute co-ordinates `(xc, yc)` for every location in
        # FPN levels.
        #
        # HINT: You have already implemented everything, just have to
        # call the functions properly.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        locations_per_fpn_level = None
        # Replace "pass" statement with your code
        shape_per_fpn_level = {
          "p3" : fpn_feats["p3"].shape,
          "p4" : fpn_feats["p4"].shape,
          "p5" : fpn_feats["p5"].shape
        }

        locations_per_fpn_level = get_fpn_location_coords(shape_per_fpn_level, self.backbone.fpn_strides, device = images.device)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass.
            # fmt: off
            return self.inference(
                images, locations_per_fpn_level,
                pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on

        ######################################################################
        # TODO: Assign ground-truth boxes to feature locations. We have this
        # implemented in a `fcos_match_locations_to_gt`. This operation is NOT
        # batched so call it separately per GT boxes in batch.
        ######################################################################
        # List of dictionaries with keys {"p3", "p4", "p5"} giving matched
        # boxes for locations per FPN level, per image. Fill this list:
        matched_gt_boxes = []
        # Replace "PASS" statement with your code
        
        # debug
        for gb in gt_boxes:
          matched_gt_boxes.append(fcos_match_locations_to_gt(locations_per_fpn_level, self.backbone.fpn_strides, gb))

        # Calculate GT deltas for these matched boxes. Similar structure
        # as `matched_gt_boxes` above. Fill this list:
        matched_gt_deltas = []
        # Replace "PASS" statement with your code
        # print(matched_gt_boxes)
        for gb in matched_gt_boxes:
          matched_gt_deltas.append(
            {
              "p3" : fcos_get_deltas_from_locations(locations_per_fpn_level['p3'], gb['p3'], self.backbone.fpn_strides['p3']),
              "p4" : fcos_get_deltas_from_locations(locations_per_fpn_level['p4'], gb['p4'], self.backbone.fpn_strides['p4']),
              "p5" : fcos_get_deltas_from_locations(locations_per_fpn_level['p5'], gb['p5'], self.backbone.fpn_strides['p5'])
            }
          )

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Collate lists of dictionaries, to dictionaries of batched tensors.
        # These are dictionaries with keys {"p3", "p4", "p5"} and values as
        # tensors of shape (batch_size, locations_per_fpn_level, 5 or 4)
        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)

        # print(matched_gt_boxes['p3'].shape) #test
        # print(matched_gt_boxes['p4'].shape) #test
        # print(matched_gt_boxes['p5'].shape) #test

        # Combine predictions and GT from across all FPN levels.
        # shape: (batch_size, num_locations_across_fpn_levels, ...)
        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        # Perform EMA update of normalizer by number of positive locations.
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image

        #######################################################################
        # TODO: Calculate losses per location for classification, box reg and
        # centerness. Remember to set box/centerness losses for "background"
        # positions to zero.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        loss_cls, loss_box, loss_ctr = None, None, None

        # Replace "PASS" statement with your code

        N, WH, C = pred_cls_logits.shape
        B = matched_gt_deltas.shape[0]
        # classification loss
        cls = torch.zeros((N*WH, C+1),device=pred_cls_logits.device)
        temp = matched_gt_boxes[:,:,-1].reshape(-1)
        loc = torch.arange(N*WH)
        classes = matched_gt_boxes[:,:,-1].reshape(-1).long()
        cls[loc, classes] = 1
        label = cls[:,:-1].reshape(N,WH,-1).float()
        loss_cls = sigmoid_focal_loss(pred_cls_logits, label)

        # box loss
        loss_box = 0.25 * F.l1_loss(pred_boxreg_deltas, matched_gt_deltas, reduction="none")
        loss_box[matched_gt_deltas < 0] *= 0.0

        # centerness loss
        center_gt = fcos_make_centerness_targets(matched_gt_deltas.reshape(-1,4)).reshape(B,WH,1)
        loss_ctr = F.binary_cross_entropy_with_logits(pred_ctr_logits, center_gt, reduction="none")
        loss_ctr[center_gt < 0] *= 0.0

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################
        # Sum all locations and average by the EMA of foreground locations.
        # In training code, we simply add these three and call `.backward()`
        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions: these values are `sqrt(class_prob * ctrness)`
                  where class_prob and ctrness are obtained by applying sigmoid
                  to corresponding logits.
        """

        # Gather scores and boxes from all FPN levels in this list. Once
        # gathered, we will perform NMS to filter highly overlapping predictions.
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():

            # Get locations and predictions from a single level.
            # We index predictions by `[0]` to remove batch dimension.
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            ##################################################################
            # TODO: FCOS uses the geometric mean of class probability and
            # centerness as the final confidence score. This helps in getting
            # rid of excessive amount of boxes far away from object centers.
            # Compute this value here (recall sigmoid(logits) = probabilities)
            #
            # Then perform the following steps in order:
            #   1. Get the most confidently predicted class and its score for
            #      every box. Use level_pred_scores: (N, num_classes) => (N, )
            #   2. Only retain prediction that have a confidence score higher
            #      than provided threshold in arguments.
            #   3. Obtain predicted boxes using predicted deltas and locations
            #   4. Clip XYXY box-cordinates that go beyond the height and
            #      and width of input image.
            ##################################################################
            # Feel free to delete this line: (but keep variable names same)
            level_pred_boxes, level_pred_classes, level_pred_scores = (
                None,
                None,
                None,  # Need tensors of shape: (N, 4) (N, ) (N, )
            )

            # Compute geometric mean of class logits and centerness:
            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid_() * level_ctr_logits.sigmoid_()
            )
            # Step 1:
            # Replace "PASS" statement with your code
            
            # print(level_cls_logits.shape)
            scores, classes = torch.max(level_pred_scores, dim=1)
            # print(scores.shape)


            # Step 2:
            # Replace "PASS" statement with your code
            
            level_pred_classes = classes[scores >= test_score_thresh]
            level_pred_scores = scores[scores >= test_score_thresh]

            # Step 3:
            # Replace "PASS" statement with your code
            
            level_pred_boxes = fcos_apply_deltas_to_locations(
              level_deltas[scores >= test_score_thresh].reshape(-1, 4),
              level_locations[scores >= test_score_thresh].reshape(-1, 2),
              self.backbone.fpn_strides[level_name]
            )

            # Step 4: Use `images` to get (height, width) for clipping.
            # Replace "PASS" statement with your code
    
            _, _ , H, W = images.shape
            # print(level_pred_boxes.shape)
            
            level_pred_boxes[:, 0] = level_pred_boxes[:, 0].clamp(min = 0)
            level_pred_boxes[:, 1] = level_pred_boxes[:, 1].clamp(min = 0)
            level_pred_boxes[:, 2] = level_pred_boxes[:, 2].clamp(max = H)
            level_pred_boxes[:, 3] = level_pred_boxes[:, 3].clamp(max = W)

            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################

            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        ######################################################################
        # Combine predictions from all levels and perform NMS.
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        # STUDENTS: This function depends on your implementation of NMS.
        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )
