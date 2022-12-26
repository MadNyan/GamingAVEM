import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from src.common.utilities import *
from src.model.model import *
from src.train.trainer import trainer

class trainerFusion(trainer):
    def __init__(self, results_path: str, dataset_name: str, 
                 visual_model: str, audio_model: str,
                 visual_train_feature: np.ndarray, visual_val_feature: np.ndarray, visual_test_feature: np.ndarray,
                 audio_train_feature: np.ndarray, audio_val_feature: np.ndarray, audio_test_feature: np.ndarray,
                 y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, 
                 epochs: int = 100, batch_size: int = 64, learn_rate: float = 1e-3, gradient_accumulation_steps: int = 1, 
                 is_transfer: bool = False, load_path: str = '', save_path: str = '', img_path: str = '', nickname:str = '', fold: str = ''):
        super(trainerFusion, self).__init__(results_path, dataset_name, epochs, batch_size, learn_rate, gradient_accumulation_steps, is_transfer, load_path, save_path, img_path, nickname, fold)
        
        # 資料集
        #visual_train_feature, visual_val_feature, visual_test_feature, audio_train_feature, audio_val_feature, audio_test_feature = self.load_feature(visual_model, audio_model)
        self.train_loader, self.val_loader, self.test_loader = self.get_dataset(visual_train_feature, visual_val_feature, visual_test_feature, audio_train_feature, audio_val_feature, audio_test_feature, y_train, y_val, y_test, self.num_classes, batch_size)
        self.model_name = visual_model + '_' + audio_model
        self.model_type = 'Fusion'

        self.init_model()
        self.init_loss()
        self.init_optimizer_and_scheduler()

    def training_step(self, model: nn.Module, inputs: list):
        visual_feature, audio_feature, labels_one_hot = inputs

        visual_feature = visual_feature.to(self.device)
        audio_feature = audio_feature.to(self.device)
        labels_one_hot = labels_one_hot.to(self.device)
            
        # Forward pass
        outputs = model(visual_feature, audio_feature)

        return outputs, labels_one_hot
    
    def get_dataset(self, visual_train_feature: np.ndarray, visual_val_feature: np.ndarray, visual_test_feature: np.ndarray, audio_train_feature: np.ndarray, audio_val_feature: np.ndarray, audio_test_feature: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, num_classes: int, batch_size: int = 64):
        #visual_train_feature = torch.from_numpy(visual_train_feature.astype(np.float32))
        #visual_val_feature = torch.from_numpy(visual_val_feature.astype(np.float32))
        #visual_test_feature = torch.from_numpy(visual_test_feature.astype(np.float32))
        #audio_train_feature = torch.from_numpy(audio_train_feature.astype(np.float32))
        #audio_val_feature = torch.from_numpy(audio_val_feature.astype(np.float32))
        #audio_test_feature = torch.from_numpy(audio_test_feature.astype(np.float32))
        
        y_train = torch.from_numpy(y_train.astype(np.int64))
        y_val = torch.from_numpy(y_val.astype(np.int64))
        y_test = torch.from_numpy(y_test.astype(np.int64))
        
        y_train_one_hot = nn.functional.one_hot(y_train, num_classes=num_classes).to(torch.float32)
        y_val_one_hot = nn.functional.one_hot(y_val, num_classes=num_classes).to(torch.float32)
        y_test_one_hot = nn.functional.one_hot(y_test, num_classes=num_classes).to(torch.float32)
    
        train_dataset = TensorDataset(visual_train_feature, audio_train_feature, y_train_one_hot)
        val_dataset = TensorDataset(visual_val_feature, audio_val_feature, y_val_one_hot)
        test_dataset = TensorDataset(visual_test_feature, audio_test_feature, y_test_one_hot)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def load_feature(self, visual_model: str, audio_model: str):
        audio_dataset_name = get_dataset_name(self.dataset_name)

        visual_train_feature = np.load(self.results_path + '/' + visual_model + '_' + self.dataset_name + '_' + self.fold + '_train_feature.npy', allow_pickle=True)
        visual_val_feature = np.load(self.results_path + '/' + visual_model + '_' + self.dataset_name + '_' + self.fold + '_val_feature.npy', allow_pickle=True)
        visual_test_feature = np.load(self.results_path + '/' + visual_model + '_' + self.dataset_name + '_' + self.fold + '_test_feature.npy', allow_pickle=True)
        audio_train_feature = np.load(self.results_path + '/' + audio_model + '_' + audio_dataset_name + '_' + self.fold + '_train_feature.npy', allow_pickle=True)
        audio_val_feature = np.load(self.results_path + '/' + audio_model + '_' + audio_dataset_name + '_' + self.fold + '_val_feature.npy', allow_pickle=True)
        audio_test_feature = np.load(self.results_path + '/' + audio_model + '_' + audio_dataset_name + '_' + self.fold + '_test_feature.npy', allow_pickle=True)

        return visual_train_feature, visual_val_feature, visual_test_feature, audio_train_feature, audio_val_feature, audio_test_feature
    
    def load_model(self, path: str = ''):
        is_load = False
        #is_load = super().load_model(path)
        if is_load:
            self.model.classifier = nn.Linear(in_features=100, out_features=self.num_classes)
            self.init_loss()
            self.init_optimizer_and_scheduler()

    def init_model(self):
        self.model = FisionClassifier(self.num_classes)

    def init_loss(self):
        self.loss_func = nn.CrossEntropyLoss()
    
    def init_optimizer_and_scheduler(self):
        total_step = self.epochs * (len(self.train_loader) // self.gradient_accumulation_steps)
        if len(self.train_loader) % self.gradient_accumulation_steps != 0:
            total_step += self.epochs

        self.optimizer = optim.Adamax(self.model.parameters(), lr=self.learn_rate)
        self.scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_step)