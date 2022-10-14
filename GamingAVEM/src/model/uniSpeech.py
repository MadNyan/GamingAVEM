import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.unispeech_sat.modeling_unispeech_sat import (
    UniSpeechSatPreTrainedModel,
    UniSpeechSatModel
)
from transformers.models.unispeech.modeling_unispeech import (
    UniSpeechPreTrainedModel,
    UniSpeechModel
)

class UniSpeechClassificationHead(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.dense = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.0)
        self.out_proj = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class UniSpeechForSpeechClassification(UniSpeechPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)        
        self.unispeech = UniSpeechModel(config) # microsoft/unispeech-sat-large
        self.classifier = UniSpeechClassificationHead(config.num_labels)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.unispeech.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states):
        outputs = torch.mean(hidden_states, dim=1)

        return outputs

    def forward(self, input_values, attention_mask=None):
        outputs = self.unispeech(input_values, attention_mask=attention_mask)

        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states)
        logits = self.classifier(hidden_states)

        return logits