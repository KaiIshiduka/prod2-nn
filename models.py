from torch import nn
import models
import torch

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.flatten(x)
        return logits
    
def test_accuracy(model, dataloader):
    
    n_corrects = 0
    
    model.eval()
    for image_batch, label_batch in dataloader:
            
        with torch.no_grad():
            logits_batch = model(image_batch)
            
        predict_batch = logits_batch.argmax(dim=1)
        n_corrects += (label_batch == predict_batch).sum().item()
        
    accuracy = n_corrects / len(dataloader.dataset)
    
    return accuracy

def train(model, dataloader, loss_fn, optimizer):
    """1 epoch の学習を行う"""
    model.train()
    for image_batch, label_batch in dataloader:
        
        logits_batch = model(image_batch)
        
        loss = loss_fn(logits_batch, label_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return loss.item()

def test(model, dataloader, loss_fn):
    """
    モデルの評価を行い、テストデータの平均ロスを計算する。

    Args:
        model: PyTorchのモデル
        dataloader: テストデータローダー
        loss_fn: 損失関数 (例: CrossEntropyLoss)

    Returns:
        float: テストデータの平均ロス
    """
    model.eval()  # 評価モードに切り替え
    total_loss = 0.0  # 全バッチのロスを格納
    total_batches = 0  # バッチ数をカウント

    with torch.no_grad():  # 勾配計算を無効化
        for image_batch, label_batch in dataloader:
            # モデルの出力を計算
            logits_batch = model(image_batch)

            # ロスを計算
            loss = loss_fn(logits_batch, label_batch)
            total_loss += loss.item()
            total_batches += 1

    # 平均ロスを計算して返す
    average_loss = total_loss / total_batches
    return average_loss

