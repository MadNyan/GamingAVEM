import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LinearLR
import torchaudio
from transformers import Wav2Vec2Processor
from transformers import AutoConfig
import numpy as np
from src.common.utilities import *
from src.model.wav2vec2 import Wav2Vec2ForSpeechClassification, Wav2Vec2ClassificationHead
from src.train.trainer import trainer

class trainerWav2vec2(trainer):
    def __init__(self, results_path: str, dataset_name: str, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, 
                 epochs: int = 30, batch_size: int = 4, learn_rate: float = 1e-4, gradient_accumulation_steps: int = 2, is_transfer: bool = False, load_path: str = '', save_path: str = '', img_path: str = '', nickname:str = '', fold: str = '', model_id: str = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'):
        super(trainerWav2vec2, self).__init__(results_path, dataset_name, epochs, batch_size, learn_rate, gradient_accumulation_steps, is_transfer, load_path, save_path, img_path, nickname, fold)
        
        # 資料集
        self.train_loader, self.val_loader, self.test_loader = self.get_dataset(x_train, x_val, x_test, y_train, y_val, y_test, self.num_classes, self.batch_size, model_id)
                
        self.model_name = 'Wav2vec2'
        self.model_type = 'Audio'

        self.init_model(model_id)
        self.init_loss()
        self.init_optimizer_and_scheduler()

        self.dataset_name = get_dataset_name(self.dataset_name)

    def training_step(self, model: nn.Module, inputs: list):
        input_values, attention_mask, labels_one_hot = inputs

        input_values = input_values.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels_one_hot = labels_one_hot.to(self.device)
            
        # Forward pass
        outputs = model(input_values, attention_mask)

        return outputs, labels_one_hot
    
    def get_dataset(self, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, num_classes: int, batch_size: int = 4, model_id: str = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'):
        x_train_input_values, x_train_attention_mask, x_val_input_values, x_val_attention_mask, x_test_input_values, x_test_attention_mask = self.get_dataset_x(x_train, x_val, x_test, model_id)
        
        y_train = torch.from_numpy(y_train.astype(np.int64))
        y_val = torch.from_numpy(y_val.astype(np.int64))
        y_test = torch.from_numpy(y_test.astype(np.int64))
        
        y_train_one_hot = nn.functional.one_hot(y_train, num_classes=num_classes).to(torch.float32)
        y_val_one_hot = nn.functional.one_hot(y_val, num_classes=num_classes).to(torch.float32)
        y_test_one_hot = nn.functional.one_hot(y_test, num_classes=num_classes).to(torch.float32)

        train_dataset = TensorDataset(x_train_input_values, x_train_attention_mask, y_train_one_hot)
        val_dataset = TensorDataset(x_val_input_values, x_val_attention_mask, y_val_one_hot)
        test_dataset = TensorDataset(x_test_input_values, x_test_attention_mask, y_test_one_hot)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def get_feature(self, model: nn.Module, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, model_id: str = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'):
        x_train_input_values, x_train_attention_mask, x_val_input_values, x_val_attention_mask, x_test_input_values, x_test_attention_mask = self.get_dataset_x(x_train, x_val, x_test, model_id)

        train_dataset = TensorDataset(x_train_input_values, x_train_attention_mask)
        val_dataset = TensorDataset(x_val_input_values, x_val_attention_mask)
        test_dataset = TensorDataset(x_test_input_values, x_test_attention_mask)

        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    
        train_feature = []
        val_feature = []
        test_feature = []

        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for input_values, attention_mask in train_loader:
                outputs = model(input_values=input_values.to(self.device), attention_mask=attention_mask.to(self.device))
                train_feature.append(outputs.detach().cpu().numpy())
            for input_values, attention_mask in val_loader:
                outputs = model(input_values=input_values.to(self.device), attention_mask=attention_mask.to(self.device))
                val_feature.append(outputs.detach().cpu().numpy())
            for input_values, attention_mask in test_loader:
                outputs = model(input_values=input_values.to(self.device), attention_mask=attention_mask.to(self.device))
                test_feature.append(outputs.detach().cpu().numpy())

        train_feature = torch.from_numpy(np.array(train_feature))
        val_feature = torch.from_numpy(np.array(val_feature))
        test_feature = torch.from_numpy(np.array(test_feature))

        train_feature = torch.reshape(train_feature, (train_feature.size()[0], train_feature.size()[-1]))
        val_feature = torch.reshape(val_feature, (val_feature.size()[0], val_feature.size()[-1]))
        test_feature = torch.reshape(test_feature, (test_feature.size()[0], test_feature.size()[-1]))

        return train_feature, val_feature, test_feature

    def get_dataset_x(self, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, model_id: str = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'):
        processor = Wav2Vec2Processor.from_pretrained(model_id) 
        target_sampling_rate = processor.feature_extractor.sampling_rate

        x_train = [self.speech_file_to_array_fn(path, target_sampling_rate) for path in x_train]
        x_val = [self.speech_file_to_array_fn(path, target_sampling_rate) for path in x_val]
        x_test = [self.speech_file_to_array_fn(path, target_sampling_rate) for path in x_test]
    
        x_train = processor(x_train, sampling_rate=target_sampling_rate, return_tensors='pt', padding=True)
        x_val = processor(x_val, sampling_rate=target_sampling_rate, return_tensors='pt', padding=True)
        x_test = processor(x_test, sampling_rate=target_sampling_rate, return_tensors='pt', padding=True)   
        return x_train.input_values, x_train.attention_mask, x_val.input_values, x_val.attention_mask, x_test.input_values, x_test.attention_mask

    def speech_file_to_array_fn(self, path, target_sampling_rate):
        '''
        Loader of audio recordings. It loads the recordings and convert them to a specific sampling rate if required, and returns
        an array with the samples of the audio.

        :param path:[str] Path to the wav file.
        :param target_sampling_rate:[int] Global variable with the expected sampling rate of the model
        '''
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def load_model(self, path: str = ''):
        is_load = super().load_model(path)
        if is_load:
            self.model.classifier = Wav2Vec2ClassificationHead(self.num_classes)
            self.init_loss()
            self.init_optimizer_and_scheduler()


    def init_model(self, model_id: str = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'):
        # config
        config = AutoConfig.from_pretrained(
            model_id, #path to the model of HuggingFace lib. that we will use as baseline to fine-tune.
            num_labels=self.num_classes, # num classes
            label2id={label: i for i, label in enumerate(self.data_labels)}, # dict that maps emotions -> numbers
            id2label={i: label for i, label in enumerate(self.data_labels)}, # dict that maps numbers -> emotions
            #finetuning_task="wav2vec2_clf",
        )
        self.model = Wav2Vec2ForSpeechClassification.from_pretrained(
            model_id,
            config=config,
        )
        #self.model.freeze_feature_extractor()

    def init_loss(self):
        self.loss_func = nn.CrossEntropyLoss()

    def init_optimizer_and_scheduler(self):
        total_step = self.epochs * (len(self.train_loader) // self.gradient_accumulation_steps)
        if len(self.train_loader) % self.gradient_accumulation_steps != 0:
            total_step += self.epochs
        
        self.optimizer = optim.Adamax(self.model.parameters(), lr=self.learn_rate)
        self.scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_step)