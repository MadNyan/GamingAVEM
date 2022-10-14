import torch
import torch.nn as nn
from transformers import Wav2Vec2Model as wav2vec2model
from transformers import AutoConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from torchsummary import summary

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task. This class stack an MLP on top of the output of the transformer,
    after a pooling layer, defined in Wav2Vec2ForSpeechClassification class"""

    def __init__(self, num_classes=6):
        super().__init__()
        self.dense = nn.Linear(1024, 1024) #Dense of 1024 neurons
        self.dropout = nn.Dropout(0.0)
        self.out_proj = nn.Linear(1024, num_classes) #Dense layer of as many neurons as the # of classes, 8 in our case

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    """Wav2Vec2.0 model with the default architecture plus a pooling layer and the MLP defined in the class Wav2Vec2ClassificationHead."""
    def __init__(self, config):
        super().__init__(config)        
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config.num_labels) # MLP stacked on top of the transformer (after the pooling layer)
        
        self.init_weights()

    def freeze_feature_extractor(self):
        """Function to freeze the layers of the feature encoder composed by a set of CNN layers"""
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states):
        outputs = torch.mean(hidden_states, dim=1)

        return outputs

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)

        hidden_states = outputs[0] #Outputs of the transformer module of Wav2Vec2.0 (with their timesteps)
        hidden_states = self.merged_strategy(hidden_states) #output after passing the pooling layer (that reduced the timesteps into a single vector)
        logits = self.classifier(hidden_states) #output after passing from the MLP stacked on top of the transformer and pooling layer

        return logits


if __name__ == '__main__':
    model_id = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'
    # config
    config = AutoConfig.from_pretrained(
        model_id,
        num_labels=10,
        label2id={label: i for i, label in enumerate((0,1,2,3,4,5,6,7,8,9))},
        id2label={i: label for i, label in enumerate((0,1,2,3,4,5,6,7,8,9))},
        finetuning_task="wav2vec2_clf",
    )
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_id,
        config=config,
    )
    model.freeze_feature_extractor()

