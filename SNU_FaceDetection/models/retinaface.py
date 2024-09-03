import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

# from models.net import FPN as FPN
from .net import FPN as FPN
# from models.net import SSH as SSH
from .net import SSH as SSH

class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)
    
class IoUHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(IoUHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 1, kernel_size=(1, 1), stride=1, padding=0)


    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 1)


class RetinaFace(nn.Module):
    def __init__(self, phase='train', version='retina', anchor_num=3):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.phase = phase
        self.fpn_num = 3
        self.anchor_num = anchor_num
        self.network = 'resnet50'
        self.model_version = version

        backbone = None


        import torchvision.models as models
        # backbone = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        backbone = models.resnet50(pretrained=True)

        in_channel = 256
        out_channel = 256
        return_layers = {'layer2': 1, 'layer3': 2, 'layer4': 3}

        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)

        in_channels_stage2 = in_channel
        in_channels_list = [
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
            in_channels_stage2 * 16
        ]
        out_channels = out_channel

        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1_1 = SSH(out_channels, out_channels)
        self.ssh1_2 = SSH(out_channels, out_channels)
        self.ssh1_3 = SSH(out_channels, out_channels)

        self.ClassHead_1 = self._make_class_head(fpn_num=self.fpn_num, inchannels=out_channel, anchor_num=self.anchor_num)
        self.BboxHead_1 = self._make_bbox_head(fpn_num=self.fpn_num, inchannels=out_channel, anchor_num=self.anchor_num)
        self.LandmarkHead_1 = self._make_landmark_head(fpn_num=self.fpn_num, inchannels=out_channel, anchor_num=self.anchor_num)
        if version == 'tina':
            self.IoUHead_1 = self._make_iou_head(fpn_num=self.fpn_num, inchannels=out_channel, anchor_num=self.anchor_num)

        self.ssh2_1 = SSH(out_channels, out_channels)
        self.ssh2_2 = SSH(out_channels, out_channels)
        self.ssh2_3 = SSH(out_channels, out_channels)

        self.ClassHead_2 = self._make_class_head(fpn_num=self.fpn_num, inchannels=out_channel, anchor_num=self.anchor_num)
        self.BboxHead_2 = self._make_bbox_head(fpn_num=self.fpn_num, inchannels=out_channel, anchor_num=self.anchor_num)
        self.LandmarkHead_2 = self._make_landmark_head(fpn_num=self.fpn_num, inchannels=out_channel, anchor_num=self.anchor_num)
        if version == 'tina':
            self.IoUHead_2 = self._make_iou_head(fpn_num=self.fpn_num, inchannels=out_channel, anchor_num=self.anchor_num) # old

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead
    
    def _make_iou_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        iouhead = nn.ModuleList()
        for i in range(fpn_num):
            iouhead.append(IoUHead(inchannels, anchor_num))
        return iouhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1_1 = self.ssh1_1(fpn[0])
        feature1_2 = self.ssh1_2(fpn[1])
        feature1_3 = self.ssh1_3(fpn[2])
        features_1 = [feature1_1, feature1_2, feature1_3]

        classifications_1 = torch.cat([self.ClassHead_1[i](feature) for i, feature in enumerate(features_1)], dim=1)
        ldm_regressions_1 = torch.cat([self.LandmarkHead_1[i](feature) for i, feature in enumerate(features_1)], dim=1)
        
        if self.model_version == 'tina':
            bbox_regressions_1 = torch.cat([self.BboxHead_1[i](feature) for i, feature in enumerate(features_1)], dim=1)
            iou_regressions_1 = torch.cat([self.IoUHead_1[i](feature) for i, feature in enumerate(features_1)], dim=1)
        else:
            iou_regressions_1 = None
            bbox_regressions_1 = torch.cat([self.BboxHead_1[i](feature) for i, feature in enumerate(features_1)], dim=1)

        # Define context head module 2
        feature2_1 = self.ssh2_1(fpn[0])
        feature2_2 = self.ssh2_2(fpn[1])
        feature2_3 = self.ssh2_3(fpn[2])
        features_2 = [feature2_1, feature2_2, feature2_3]
            
        classifications_2 = torch.cat([self.ClassHead_2[i](feature) for i, feature in enumerate(features_2)], dim=1)
        ldm_regressions_2 = torch.cat([self.LandmarkHead_2[i](feature) for i, feature in enumerate(features_2)], dim=1)
        if self.model_version == 'tina':
            bbox_regressions_2 = torch.cat([self.BboxHead_2[i](feature) for i, feature in enumerate(features_2)], dim=1) # old
            iou_regressions_2 = torch.cat([self.IoUHead_2[i](feature) for i, feature in enumerate(features_2)], dim=1)
        else:
            iou_regressions_2 = None
            bbox_regressions_2 = torch.cat([self.BboxHead_2[i](feature) for i, feature in enumerate(features_2)], dim=1)

        bbox_regressions_2 = bbox_regressions_1 + bbox_regressions_2

        if self.phase == 'train':
            output = (bbox_regressions_1, classifications_1, ldm_regressions_1, iou_regressions_1)
            output2 = (bbox_regressions_2, classifications_2, ldm_regressions_2, iou_regressions_2)
        else:
            output = (bbox_regressions_1, F.softmax(classifications_1, dim=-1), ldm_regressions_1, iou_regressions_1) 
            output2 = (bbox_regressions_2, F.softmax(classifications_2, dim=-1), ldm_regressions_2, iou_regressions_2)


        # shape of output will be (batch, 16800, 2/4/10) when anchor=2, fpn=3
        return output, output2