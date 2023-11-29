import torch
import torch.nn as nn
import torch.nn.functional as F

def my_KLDivLoss(x, y):
  """Returns K-L Divergence loss
  Different from the default PyTorch nn.KLDivLoss in that
  a) the result is averaged by the 0th dimension (Batch size)
  b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
  """
  loss_func = nn.KLDivLoss(reduction='sum')
  y += 1e-16
  n = y.shape[0]
  loss = loss_func(x, y) / n
  return loss

class SFCN(nn.Module):
  def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True):
    super(SFCN, self).__init__()
    n_layer = len(channel_number)
    self.feature_extractor = nn.Sequential()
    for i in range(n_layer):
      if i == 0:
        in_channel = 1
      else:
        in_channel = channel_number[i-1]
      out_channel = channel_number[i]
      if i < n_layer-1:
        self.feature_extractor.add_module('conv_%d' % i,
                                          self.conv_layer(in_channel,
                                                          out_channel,
                                                          maxpool=True,
                                                          kernel_size=3,
                                                          padding=1))
      else:
        self.feature_extractor.add_module('conv_%d' % i,
                                          self.conv_layer(in_channel,
                                                          out_channel,
                                                          maxpool=False,
                                                          kernel_size=1,
                                                          padding=0))
    self.classifier = nn.Sequential()
    # have to adjust the average shape size to make sure the output tensor has the same shape as labels
    avg_shape = [2, 2, 2]
    self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
    if dropout is True:
      self.classifier.add_module('dropout', nn.Dropout(0.5))
    i = n_layer
    in_channel = channel_number[-1]
    out_channel = output_dim
    self.classifier.add_module('conv_%d' % i,
                                nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

  @staticmethod
  def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
    if maxpool is True:
      layer = nn.Sequential(
        nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
        nn.BatchNorm3d(out_channel),
        nn.MaxPool3d(2, stride=maxpool_stride),
        nn.ReLU(),
      )
    else:
      layer = nn.Sequential(
        nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
        nn.BatchNorm3d(out_channel),
        nn.ReLU()
      )
    return layer

  def forward(self, x):
    out = list()
    x_f = self.feature_extractor(x)
    x = self.classifier(x_f)
    x = F.log_softmax(x, dim=1)
    out.append(x)
    return out