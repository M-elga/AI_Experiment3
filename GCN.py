import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_edge

# Add these new imports for visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load Cora dataset
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]


# Define improved GAT model
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(hidden_channels * heads)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        edge_index, _ = dropout_edge(edge_index, p=0.2, training=self.training)
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.norm2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(
    in_channels=dataset.num_node_features,
    hidden_channels=8,
    out_channels=dataset.num_classes,
    heads=8,
    dropout=0.6
).to(device)
data = data.to(device)

optimizer = Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# Initialize tracking
history = {
    'epoch': [],
    'loss': [],
    'train_acc': [],
    'val_acc': [],
    'test_acc': [],
    'lr': []
}


# Training function with gradient clipping
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    optimizer.step()
    return loss.item()


# Evaluation
@torch.no_grad()
def evaluate():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
        accs.append(acc)
    return accs


# Training loop with tracking
best_val_acc = 0
best_test_acc = 0
patience = 100
counter = 0

for epoch in range(1, 1001):
    loss = train()
    train_acc, val_acc, test_acc = evaluate()
    current_lr = optimizer.param_groups[0]['lr']

    # Track metrics
    history['epoch'].append(epoch)
    history['loss'].append(loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['test_acc'].append(test_acc)
    history['lr'].append(current_lr)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        best_epoch = epoch
        counter = 0
        # Save best predictions
        with torch.no_grad():
            best_out = model(data.x, data.edge_index)
            best_preds = best_out.argmax(dim=1)
    else:
        counter += 1

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, LR: {current_lr:.6f}")

    if counter >= patience:
        print("Early stopping triggered!")
        break

# Visualization and reporting
df = pd.DataFrame(history)
df['phase'] = df['epoch'].apply(lambda x: 'warmup' if x < 50 else 'convergence' if x < best_epoch else 'cooldown')

# 1. Training curves
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.lineplot(data=df, x='epoch', y='loss', hue='phase')
plt.title('Training Loss')
plt.subplot(2, 2, 2)
sns.lineplot(data=df, x='epoch', y='train_acc', label='Train')
sns.lineplot(data=df, x='epoch', y='val_acc', label='Validation')
sns.lineplot(data=df, x='epoch', y='test_acc', label='Test')
plt.title('Accuracy Trends')
plt.subplot(2, 2, 3)
sns.lineplot(data=df, x='epoch', y='lr')
plt.title('Learning Rate')
plt.tight_layout()
plt.savefig('training_metrics.png')
plt.close()

# 2. Confusion matrix
cm = confusion_matrix(data.y[data.test_mask].cpu(), best_preds[data.test_mask].cpu())
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(dataset.num_classes),
            yticklabels=range(dataset.num_classes))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# 3. Final report
print(f"\n=== Best Results ===")
print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Test Accuracy at Best Val: {best_test_acc:.4f}")

# Save data
df.to_csv('training_history.csv', index=False)
print("\nSaved visualizations and training history to:")
print("- training_metrics.png")
print("- confusion_matrix.png")
print("- training_history.csv")