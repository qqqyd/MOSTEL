import torch.nn as nn
from rec_modules import TPS_SpatialTransformerNetwork, ResNet_FeatureExtractor, BidirectionalLSTM, Attention
import time

class Rec_Model(nn.Module):
    def __init__(self, cfg):
        super(Rec_Model, self).__init__()
        """ TPS Transformation """
        input_channel = 3 if cfg.use_rgb else 1
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=20, I_size=(32, 100), I_r_size=(32, 100), I_channel_num=input_channel) # num_fiducial, imgH, imgW, input_channel

        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, 512) # input_channel, output_channel
        self.FeatureExtraction_output = 512  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ BiLSTM Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, 256, 256), # hidden_size, hidden_size
            BidirectionalLSTM(256, 256, 256)) # hidden_size, hidden_size, hidden_size
        self.SequenceModeling_output = 256

        """ Prediction """
        self.Prediction = Attention(self.SequenceModeling_output, 256, 38) # hidden_size, num_class


    def forward(self, input, text, is_train=True):
        """ TPS Transformation stage """
        input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2).contiguous())  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ BiLSTM Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Attention Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=34) # batch_max_length

        return prediction
