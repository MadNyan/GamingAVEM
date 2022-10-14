import torch
import torch.nn as nn
from torchkeras import summary
from src.model.dense_FaceLiveNet import Stem_layer, Inception_1, Inception_2, Translate_layer, Inception_3

class Dense_FaceLiveNet_Variant(nn.Module):
    def __init__(self):
        super(Dense_FaceLiveNet_Variant, self).__init__()
        self.stem_layer = Stem_layer()
        self.inception_1 = Inception_1()
        self.inception_2 = Inception_2()
        self.translate_layer = Translate_layer()
        self.inception_3 = Inception_3()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.stem_layer(x)
        x = self.inception_1(x)
        x = self.inception_2(x)
        x = self.translate_layer(x)
        x = self.inception_3(x)
        x = self.pool(x)
        x = self.flatten(x)

        return x

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        assert len(x.size()) > 2

        # reshape input data --> (batchs * timesteps, input_size)
        reshaped_input = x.contiguous().view(-1, x.size(-3), x.size(-2), x.size(-1))

        output = self.module(reshaped_input)

        if self.batch_first:
            # (batchs, timesteps, output_size)
            output = output.contiguous().view(x.size(0), -1, output.size(-1))
        else:
            # (timesteps, batchs, output_size)
            output = output.contiguous().view(-1, x.size(0), output.size(-1))

        return output

class Dense_FaceLiveNet_BiLSTM_Classification_Head(nn.Module):
    def __init__(self, num_classes=6):
        super(Dense_FaceLiveNet_BiLSTM_Classification_Head, self).__init__()
        
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x

class Dense_FaceLiveNet_BiLSTM(nn.Module):
    def __init__(self, num_classes=6):
        super(Dense_FaceLiveNet_BiLSTM, self).__init__()

        self.timedistributed_cnn = TimeDistributed(module=Dense_FaceLiveNet_Variant())
        
        self.lstm_0 = nn.LSTM(input_size=4224, hidden_size=1024, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_1 = nn.LSTM(input_size=2048, hidden_size=1024, num_layers=1, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(in_features=2048, out_features=1024)
        self.classifier = Dense_FaceLiveNet_BiLSTM_Classification_Head(num_classes)

    def forward(self, x):

        x = self.timedistributed_cnn(x)
        
        out, _ = self.lstm_0(x)
        out, _ = self.lstm_1(out)
        x = out[:, -1, :]

        x = self.fc(x)
        x = self.classifier(x)

        return x

#if __name__ == '__main__':
#    model = Dense_FaceLiveNet_BiLSTM(num_classes=8)
#    data = torch.randn((16, 10, 1, 224, 224))
#    model(data)
