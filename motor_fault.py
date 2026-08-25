import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import gradio as gr

# 固定随机种子，消除初始化随机性
np.random.seed(42)
torch.manual_seed(42)

# 参数
seq_len = 30
n_samples = 1000

# 生成模拟数据集
# 正常样本
normal_data = np.random.normal(loc=0.2, scale=0.05, size=(n_samples//2, seq_len))
# 故障样本
fault_data = np.random.normal(loc=0.8, scale=0.08, size=(n_samples//2, seq_len))

X = np.concatenate([normal_data, fault_data], axis=0)
y = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)], axis=0)

# 划分训练集
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转为LSTM输入格式 (样本数,序列长度,特征维度=1)
X_train = X_train.reshape(-1, seq_len, 1)
X_test = X_test.reshape(-1, seq_len, 1)

# ---------------------- 随机森林基线 ----------------------
X_train_flat = X_train.reshape(-1, seq_len)
X_test_flat = X_test.reshape(-1, seq_len)
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train_flat, y_train)
y_pred_rf = rf.predict(X_test_flat)

print("=====随机森林基线模型结果=====")
from sklearn.metrics import accuracy_score, recall_score
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
print(f"准确率：{rf_acc:.2f}")
print(f"故障召回率：{rf_recall:.2f}")

# ---------------------- LSTM模型 ----------------------
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 2)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTMClassifier()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 转tensor
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

epochs = 30
loss_list = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())

    # 验证
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_pred_cls = torch.argmax(test_pred, dim=1)
        test_acc = (test_pred_cls == y_test_t).float().mean().item()
        test_recall = recall_score(y_test, test_pred_cls.numpy())
    print(f"Epoch{epoch+1:2d} | Loss:{loss.item():.3f} | 测试准确率:{test_acc:.2f} | 故障召回:{test_recall:.2f}")

# 绘图
import matplotlib.pyplot as plt
plt.figure("Training Loss Curve")
plt.plot(range(1, epochs+1), loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# 最终评估
model.eval()
with torch.no_grad():
    final_pred = model(X_test_t)
    final_pred_cls = torch.argmax(final_pred, dim=1).numpy()
final_acc = accuracy_score(y_test, final_pred_cls)
final_recall = recall_score(y_test, final_pred_cls)
print("\n=====LSTM深度学习模型最终效果=====")
print(f"准确率：{final_acc:.2f}")
print(f"故障召回率：{final_recall:.2f}")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, final_pred_cls)
print("混淆矩阵：")
print(cm)

# 导出ONNX
dummy_input = torch.randn(1, seq_len, 1, dtype=torch.float32)
torch.onnx.export(model, dummy_input, "motor_fault_model.onnx", input_names=["input"], output_names=["output"])
print("模型已导出 motor_fault_model.onnx")

# Gradio推理函数
def predict_fault(vibration_data):
    arr = np.array([float(i) for i in vibration_data.split(",")])
    arr = scaler.transform(arr.reshape(1,-1))
    tensor = torch.FloatTensor(arr.reshape(1, seq_len, 1))
    model.eval()
    with torch.no_grad():
        res = model(tensor).argmax(1).item()
    if res == 1:
        return "警告：电机存在故障风险，建议停机检修"
    else:
        return "状态正常"

demo = gr.Interface(
    fn=predict_fault,
    inputs=gr.Textbox(label="输入30个振动传感器数值，逗号分隔"),
    outputs=gr.Textbox(label="AI预测结果"),
    title="工厂电机故障预测AI"
)

if __name__ == "__main__":
    demo.launch()
    