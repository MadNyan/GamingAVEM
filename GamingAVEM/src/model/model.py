import torch
import torch.nn as nn

class FisionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FisionClassifier, self).__init__()
        self.fc_0 = nn.Linear(in_features=num_classes*2, out_features=num_classes*2)
        self.fc_1 = nn.Linear(in_features=num_classes*2, out_features=num_classes)
        self.softmax = nn.Softmax()

    def forward(self, visual_feature, audio_feature):
        mix = torch.cat([visual_feature, audio_feature], 1)
        x = self.fc_0(mix)
        x = self.fc_1(x)
        x = self.softmax(x)

        return x