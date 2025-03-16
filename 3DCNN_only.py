# import wandb
import datetime
from argparse import Namespace
# wandb.login()

import numpy as np
import pandas as pd
import random


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image, ImageFile
import copy
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed):
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed for CPU and GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set PyTorch deterministic operations for cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)


config=Namespace(
    project_name="VGG19_TRANSFORMER",
    batch_size = 16,
    epochs = 150,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    criterion = nn.CrossEntropyLoss()
    )
    
print(config.device)


root_dir="../data/Thermal Camera Images" #480x640 "..":表上級目錄
file=[]
label=["NoGas","Perfume","Smoke","Mixture"]

for i, cls in enumerate(label):
    # print(cls)
    class_dir = os.path.join(root_dir, cls)
    class_image=[f for f in os.listdir(class_dir)]
    class_image = sorted(class_image, key=lambda x: int(x.split("_")[0]))
    for img in class_image:
        file.append((os.path.join(class_dir, img), i))

file = np.array(file)

print(file)


df=pd.read_csv("../data/Gas Sensors Measurements/Gas_Sensors_Measurements.csv")

# 繪製趨勢圖
plt.figure(figsize=(10, 6))
for column in df.columns[1:8]:  # 選擇MQ2到MQ135列
    plt.plot(df[column], label=column)

plt.xlabel('Serial Number')
plt.ylabel('Sensor Value')
plt.title('Trend of Gas Sensors')
plt.legend()
plt.grid(True)
plt.show()

data=pd.read_csv("../data/Gas Sensors Measurements/Gas_Sensors_Measurements.csv")
data=data.drop("Corresponding Image Name",axis=1)
data=data.drop("Gas",axis=1)
# data=data.drop("Serial Number",axis=1)
data.insert(1,"index", range(0, len(data))) #新增feature 1-6400
data=np.array(data)

data_cat=np.concatenate((data, file), axis=1)
scaler = MinMaxScaler(feature_range=(0,1))
data_cat[:,2:9]=scaler.fit_transform(data[:,2:9]) # feature 1 (1-1600)*4|feature 2 1-6400|feature 3-9 sensor
print(data_cat)
# train_ratio = 0.6
# valid_ratio = 0.2
# test_ratio = 0.2

# # 計算相應的樣本數量
# total_samples = len(data_cat)
# num_train = int(train_ratio * total_samples)
# num_valid = int(valid_ratio * total_samples)

# # 使用 numpy 的切片功能進行分割
# train, X_temp= data_cat[:num_train], data_cat[num_train:num_train + num_valid]
# valid, test= data_cat[num_train:num_train + num_valid], data_cat[num_train + num_valid:]

indices = [i for i in range(len(data_cat)) if i%5!=0]
train_=data_cat[indices] #5120
# print(train_)
indices_tr=[i for i in indices if i%6!=0]
# print(indices_vl)
train_id=train_[:,1].astype(int)
train=train_[np.isin(train_id,indices_tr)]
indices_vl=[i for i in indices if i%6==0]
valid=train_[np.isin(train_id,indices_vl)]
indices = [i for i in range(len(data_cat)) if i%5==0]
test=data_cat[indices]
print(len(train),len(valid),len(test))
# train_,test=train_test_split(data_cat,train_size=0.8,random_state=42)
# train,valid=train_test_split(train_,train_size=0.8,random_state=42)

# sorted_indices_tr = np.argsort(train[:, 0])
# sorted_indices_vl = np.argsort(valid[:, 0])
# sorted_indices_tt = np.argsort(test[:, 0])
# train=train[sorted_indices_tr]
# valid=train[sorted_indices_vl]
# test=train[sorted_indices_tt]

y_train=train[:,-1]
# print(y_train)
MQ_train=train[:,2:9]

image_train=train[:,-2]
print(image_train)
y_valid=valid[:,-1]
MQ_valid=valid[:,2:9]
image_valid=valid[:,-2]

y_test=test[:,-1]
MQ_test=test[:,2:9]
image_test=test[:,-2]


class sensordata(Dataset):  # inheriting Dataset, not nn.Module
    def __init__(self, target, MQ=None, image=None, transform=None, seq_len=12):
        self.seq_len = seq_len
        self.transform = transform
        self.img = image
        self.MQ = MQ
        self.target = target

    def __len__(self):
        return len(self.target) - self.seq_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        while len(set(self.target[idx:idx + self.seq_len])) > 1:  # If the sequence contains more than one label
            idx += 1

        target = self.target[idx].astype(np.int64)

        if self.img is not None:
            img_seq = []
            for i in range(idx, idx + self.seq_len):
                image = Image.open(self.img[i])
                if self.transform:
                    image = self.transform(image)
                img_seq.append(image)
            img_seq = torch.stack(img_seq)
            if self.MQ is None:
                return img_seq, target
        
        if self.MQ is not None:
            MQ_seq = self.MQ[idx:idx + self.seq_len].astype(float)
            MQ_idx = torch.FloatTensor(MQ_seq)
            if self.img is None:
                return MQ_idx, target

        return MQ_idx, img_seq, target


# Assuming y_train, MQ_train, image_train are defined somewhere
train_transform = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tr_data_all=sensordata(y_train,MQ=None,image=image_train,transform=train_transform)
vl_data_all=sensordata(y_valid,MQ=None,image=image_valid,transform=test_transform)
tt_data_all=sensordata(y_test,MQ=None,image=image_test,transform=test_transform)

tr_loader_all = DataLoader(tr_data_all, shuffle=True, batch_size=config.batch_size)
vl_loader_all = DataLoader(vl_data_all, shuffle=False, batch_size=config.batch_size)
tt_loader_all=DataLoader(tt_data_all,shuffle=False, batch_size=config.batch_size)
img ,labels= next(iter(tr_loader_all))
print(img.shape)
print(labels)




# 3DCNN
class SimplifiedVGG19_3D(nn.Module):
    def __init__(self, num_classes):
        super(SimplifiedVGG19_3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # nn.Conv3d(32, 64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=2, stride=2),
            # nn.Conv3d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(32*4*4 , 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        y_pred_prob = self.softmax(x)
        return x, y_pred_prob
 

model_ft=SimplifiedVGG19_3D(4).to(config.device)

# # 3Dresnet
# class BasicBlock3D(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BasicBlock3D, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.bn1 = nn.BatchNorm3d(out_channels)
#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm3d(out_channels)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
#                 nn.BatchNorm3d(out_channels)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class ResNet3D(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=4):
#         super(ResNet3D, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm3d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.fc = nn.Linear(128, num_classes)

#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels
#         for _ in range(1, num_blocks):
#             layers.append(block(out_channels, out_channels, stride=1))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = F.adaptive_avg_pool3d(out, (1, 1, 1))
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#         return out

# def ResNet3D18():
#     return ResNet3D(BasicBlock3D, [2, 2])


# class Ensemble3D(nn.Module):
#     def __init__(self, modelA, modelB, num_classes):
#         super(Ensemble3D, self).__init__()
#         self.modelA = modelA
#         self.modelB = modelB
#         self.fc = nn.Linear(num_classes * 2, num_classes)

#     def forward(self, x):
#         outA = self.modelA(x)
#         outB = self.modelB(x)
#         out = torch.cat((outA, outB), dim=1)
#         out = self.fc(out)
#         return out

# # Instantiate the individual models
# num_classes = 4  # Example number of classes
# modelA = SimplifiedVGG19_3D(num_classes)
# modelB = ResNet3D18()
# # Instantiate the ensemble model
# model_ft= Ensemble3D(modelA, modelB, num_classes).to(config.device)

# Positional Encoding for Transformer
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000): #d_model:embedding size
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #(5000,1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term) #越後面的dim頻率越小 sin填入偶數行
#         pe[:, 1::2] = torch.cos(position * div_term) #cos填入奇數行
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :] #x是time series不用另外embedding
#         return self.dropout(x)
    
    
# Model definition using Transformer
# class TransformerModel(nn.Module):
#     def __init__(self, input_dim=7, d_model=64, nhead=4, num_layers=2, dropout=0.2):
#         super(TransformerModel, self).__init__()

#         self.encoder = nn.Linear(input_dim, d_model) #(16,12,64)
#         self.pos_encoder = PositionalEncoding(d_model, dropout) #(16,12,64)
#         encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
#         self.decoder = nn.Linear(d_model, 4)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.pos_encoder(x)
#         x = self.transformer_encoder(x) #16,12,64
#         x = self.decoder(x[:, -1, :]) #seq_len的最後一步 #16,4
#         # print(x.size())
#         return x
# transformmodel = TransformerModel().to(config.device)

# class CombinedFCModel(nn.Module):
#     def __init__(self,
#                  model_dim=8,
#                  drop_and_BN='drop-BN',
#                  num_labels=4,
#                  dropout=0.2):

#         super(CombinedFCModel, self).__init__()

#         self.model_dim = model_dim
#         # self.drop_and_BN = drop_and_BN

#         # self.dropout = nn.Dropout(dropout)

#         #sequence
#         self.transformer=transformmodel
#         # self.bn_sequence = nn.BatchNorm1d(model_dim)

#         #image
#         self.vgg=model_ft
#         # self.bn_vgg = nn.BatchNorm1d(model_dim)

#         #classifier
#         self.linear1 = nn.Linear(model_dim, 16)
#         # self.bn_1 = nn.BatchNorm1d(16)
#         self.linear2 = nn.Linear(16, num_labels)
#         self.softmax = nn.Softmax(dim=1)

#     # def drop_BN_layer(self, x, part='seq'):
#     #     if part == 'seq':
#     #         bn = self.bn_sequence
#     #     elif part == 'vgg':
#     #         bn = self.bn_vgg

#     #     if self.drop_and_BN == 'drop-BN':
#     #         x = self.dropout(x)
#     #         x = bn(x)
#     #     elif self.drop_and_BN == 'BN-drop':
#     #         x = bn(x)
#     #         x = self.dropout(x)
#     #     elif self.drop_and_BN == 'drop-only':
#     #         x = self.dropout(x)
#     #     elif self.drop_and_BN == 'BN-only':
#     #         x = bn(x)
#     #     elif self.drop_and_BN == 'none':
#     #         pass

#     #     return x

#     def forward(self, image,sensor):


#         #visual feature
        
#         image=image.transpose(1, 2)
#         # print(image.size())
#         output = self.vgg(image)
#         # print(output.size())
#         # output = F.relu(self.linear_image(output))
#         # output = self.drop_BN_layer(output, part='vgg')


#         #sequence
        
#         se_out=self.transformer(sensor)
#         # print(se_out.size())
#         # se_out=self.drop_BN_layer(se_out,part="seq")

#         output = torch.cat([output, se_out], dim=1)
        

#         output = torch.relu(self.linear1(output))
#         # output = self.dropout(output)
#         # output = self.bn_1(output)
#         output = self.linear2(output)  #(16,4)
#         # print('output_size:{}'.format(output.shape))
#         y_pred_prob = self.softmax(output)

#         return output, y_pred_prob
    
# combined_model = CombinedFCModel().to(config.device)

class BaggingEnsemble(nn.Module):
    def __init__(self, base_model_class, num_models, *args, **kwargs):
        super(BaggingEnsemble, self).__init__()
        self.models = nn.ModuleList([base_model_class(*args, **kwargs) for _ in range(num_models)])
        self.num_models = num_models

    def forward(self, image, sensor):
        outputs = []
        y_pred_probs = []
        for model in self.models:
            output, y_pred_prob = model(image, sensor)
            outputs.append(output)
            y_pred_probs.append(y_pred_prob)

        # Average predictions
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        avg_y_pred_prob = torch.mean(torch.stack(y_pred_probs), dim=0)

        return avg_output, avg_y_pred_prob
    
    
def train_combined(model,train_loader,device, optimizer, loss_fn):
    model.train()

    n_corrects = 0
    total = 0
    train_loss = 0.
    step = 0
    for idx, (img_data,label) in enumerate(train_loader):
        img_data= img_data.to(device)
        # sensor_data=sensor_data.to(device)
        labels=label.to(device)
        optimizer.zero_grad()
        img_data=img_data.transpose(1,2)
        output_combined = model(img_data)
        loss_combined = loss_fn(output_combined[0], labels)

        _, predictions = torch.max(output_combined[1], dim=1)
        n_corrects += predictions.eq(labels).sum().item()
        total += labels.size(0)
        train_loss += loss_combined.item()


        loss_combined.backward()
        optimizer.step()

        current_lr = optimizer.param_groups[0]["lr"]
        step += 1

        train_accuracy = 100. * n_corrects/total

        avg_train_loss = train_loss/(idx+1)
        if (idx+1) % 16 == 0:
            print(f'Batch: [{idx+1}/{len(train_loader)}], Training Loss: {avg_train_loss:.3f} | Training Acc: {train_accuracy:.2f}% | lr: {current_lr:.5f}')

    avg_train_loss = train_loss/(step+1)
    train_accuracy = 100. * n_corrects/total
    return avg_train_loss, train_accuracy

def valid_combined(model,valid_loader,device, loss_fn):
    model.eval()


    n_corrects = 0
    total = 0
    valid_loss = 0.
    for idx, (img_data,label) in enumerate(valid_loader):
        with torch.no_grad():
            img_data= img_data.to(device)
            # sensor_data=sensor_data.to(device)
            labels=label.to(device)
            img_data=img_data.transpose(1,2)
        output_combined = model(img_data)
        loss_combined = loss_fn(output_combined[0], labels)
        _, predictions = torch.max(output_combined[1], dim=1)
        n_corrects += predictions.eq(labels).sum().item()
        total += labels.size(0)
        valid_loss += loss_combined.item()
        idx += 1
    avg_valid_loss = valid_loss/(idx+1)
    valid_accuracy = 100 * n_corrects/total
    return avg_valid_loss, valid_accuracy,model
from sklearn.metrics import accuracy_score
def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.device)
            images=images.transpose(1,2)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy
def test_combined(model, test_loader, device):
    model.eval()

    n_corrects = 0
    total = 0
    all_predictions = []
    all_labels = []

    for idx, ( img_data, label) in enumerate(test_loader):

        with torch.no_grad():
            img_data= img_data.to(device)
            # sensor_data=sensor_data.to(device)
            labels=label.to(device)
            img_data=img_data.transpose(1,2)
        output_combined = model(img_data)

        _, predictions = torch.max(output_combined[1], dim=1)
        n_corrects += predictions.eq(labels).sum().item()
        total += labels.size(0)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    valid_accuracy = 100 * n_corrects / total
    # Compute confusion matrix
    confusion_mat = confusion_matrix(all_labels, all_predictions)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.set(font_scale=1.5)  # Adjust font size
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=range(4), yticklabels=range(4))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # Compute precision, recall, and F1 score
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
    
    return valid_accuracy, confusion_mat, precision, recall, f1_score


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss,model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(model.state_dict(), 'models/img_only.pth')
            print('Saving model (loss = {:.4f})'
            .format(self.min_validation_loss))
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print(f"|||early stop:{self.counter}")
            if self.counter >= self.patience:
                return True
        return False
    
criterion = nn.CrossEntropyLoss()
# ======multimodal=====
optimizer = optim.SGD(
                      model_ft.parameters(), lr=0.0005, momentum=0.9)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                           factor=0.5, patience=7, verbose=True)


# -------------------------
# ---- train model ----
# -------------------------
early_stopper = EarlyStopper(patience=10)
epoch_nums=[]
training_loss=[]
validation_loss=[]

for epoch in range(config.epochs):
    
    train_loss, train_accuracy = train_combined(model_ft,tr_loader_all, config.device, optimizer, config.criterion)
    valid_loss, valid_accuracy,model= valid_combined(model_ft,vl_loader_all, config.device, config.criterion)
    print(f'[{epoch+1}/{config.epochs}] | validation loss: {valid_loss:.4f} | validation accuracy: {valid_accuracy:.2f}%')
    scheduler.step(valid_loss)
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(valid_loss)
    if early_stopper.early_stop(valid_loss,model):
        break
    

test_acc, confusion_mat, precision, recall, f1_score= test_combined(model_ft,tt_loader_all, config.device)
print(f"test acc:{test_acc:.4f}")


# del combined_model
combined_model =  SimplifiedVGG19_3D(4).to(config.device)
ckpt = torch.load('models/img_only.pth', map_location='cpu')  # Load your best model
combined_model.load_state_dict(ckpt)
test_acc, confusion_mat, precision, recall, f1_score= test_combined(combined_model,tt_loader_all, config.device)
print(f"test acc:{test_acc:.4f}")


print(combined_model)
print(precision,recall,f1_score)
