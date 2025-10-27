# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from arguments import *
from dataloader import UnifiedDataLoader
args = AttnWGAIN_arguments()
# ������ **生成随机二值矩阵并用 softmax 处理**
def M_prob_gen(M_matrix):
    # 将 M_matrix 展平为一维 (batch_size * height * width)
    M_matrix_flatten = torch.flatten(M_matrix)  # 展平为一维

    # 将 M_matrix_flatten 转换为 one-hot 编码
    M_matrix_one_hot = F.one_hot(M_matrix_flatten.to(torch.int64), num_classes=2)  # (batch_size * height * width, 2)

    # 使用温度参数的 softmax 生成平滑的概率分布
    temperature = torch.rand(M_matrix_one_hot.shape[0], 1, device=M_matrix.device)  # (batch_size * height * width, 1)
    temperature = temperature.expand(M_matrix_one_hot.shape)  # 扩展到与 one-hot 相同的形状

    # 将 M_matrix_one_hot 移动到与 temperature 相同的设备
    M_matrix_one_hot = M_matrix_one_hot.to(temperature.device)
    M_matrix_one_hot_prob = F.softmax(M_matrix_one_hot / temperature, dim=1)  # (batch_size * height * width, 2)

    # 提取最大值和最小值概率
    M_matrix_one_hot_prob_max, _ = torch.max(M_matrix_one_hot_prob, dim=1)  # (batch_size * height * width)
    M_matrix_one_hot_prob_min, _ = torch.min(M_matrix_one_hot_prob, dim=1)  # (batch_size * height * width)

    # 根据原始值选择概率
    M_matrix_prob = torch.where(M_matrix_flatten == 1, M_matrix_one_hot_prob_max, M_matrix_one_hot_prob_min)
    M_matrix_prob = M_matrix_prob.reshape(M_matrix.shape)  # 恢复为原始的三维形状
    return M_matrix_prob

def new_train(dataloader):
    evalMask_collector = []  # 收集指示掩码
    for idx, data in enumerate(dataloader):
        indices, X, missing_mask, H, deltaPre, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)
        evalMask_collector.append(indicating_mask)
        print("indicating_mask: ", indicating_mask.shape, torch.min(indicating_mask), torch.max(indicating_mask))
        #print("indicating_mask number: ", indicating_mask)
    evalMask_collector = torch.cat(evalMask_collector)
    print("evalMask_collector: ", evalMask_collector.shape, torch.min(evalMask_collector), torch.max(evalMask_collector))
    #print("evalMask_collector number: ", evalMask_collector)
    # 返回收集的数据
    return evalMask_collector
    
# **定义 Variational Autoencoder (VAE)**
class VAE(nn.Module):
    def __init__(self, data_sample_shape=(300, 128), latent_dim=128):
        super(VAE, self).__init__()
        self.data_sample_shape = data_sample_shape
        self.input_dim = data_sample_shape[0] * data_sample_shape[1]
        self.latent_dim = latent_dim
        # print("input_dim: ", self.input_dim, "latent_dim: ", self.latent_dim)

        # 编码器
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc21 = nn.Linear(64, self.latent_dim)  # 均值
        self.fc22 = nn.Linear(64, self.latent_dim)  # 方差

        # 解码器
        self.fc3 = nn.Linear(self.latent_dim, 1024)
        self.fc4 = nn.Linear(1024, self.input_dim)
    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.leaky_relu(self.fc3(z), negative_slope=0.01)
        recon_x = torch.sigmoid(self.fc4(h3))
        recon_x = recon_x.view(-1, self.data_sample_shape[0], self.data_sample_shape[1])
        return recon_x

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# **VAE 损失函数**
def loss_function(recon_x, x, mu, logvar):
    #BCE = nn.BCELoss(reduction='mean')(recon_x, x)  # 计算重构误差
    mse_loss = nn.MSELoss()(recon_x, x)  # 使用 MSELoss 替代 BCELoss
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  # 重构损失，交叉熵
    D_KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # 计算 KL 散度
    return mse_loss, BCE, D_KL, BCE + D_KL


# **训练 VAE**
def train_vae(vae, M_prob, num_epochs=10000, batch_size=32):
    optimizer = optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-6)
    # 创建数据集
    dataset = TensorDataset(M_prob, M_prob)  # 鐩爣涔熸槸 M_prob
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 早停技术相关变量
    best_rmse_loss = float('inf')
    patience_counter = 0
    early_stop = False
    patience = 1000
    for epoch in range(num_epochs):
        rmse_total = 0
        train_loss = 0
        for batch in train_loader:
            batch_rmse = 0
            data, target = batch
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(data)
            mse_loss, BCE, D_KL, loss = loss_function(recon_x, target, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_rmse += torch.sqrt(torch.mean((recon_x - target) ** 2)).item()
        # 每 100 个 epoch 输出一次
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader):.4f}, RMSE: {batch_rmse / len(train_loader):.4f}, mse_loss: {mse_loss:.4f}")
        
        if batch_rmse < best_rmse_loss:
            best_rmse_loss = batch_rmse
            patience_counter = 0
            torch.save(vae.state_dict(), "vae_model.pth")
            #print(f"Epoch {epoch + 1}: 模型已保存（最佳损失: {best_rmse_loss:.4f}）")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                early_stop = True
                print(f"Epoch {epoch + 1}: 早停触发，训练结束。")
                break
    #torch.save(vae.state_dict(), "vae_model.pth")
    if not early_stop:
        print("训练完成！")
    print("VAE model saved successfully.")
    return vae


def generate_multiple_samples(model, num_samples=10):
    model.eval()
    z_samples = torch.randn(num_samples, model.latent_dim)
    generated_samples = model.decode(z_samples)
    # print("generated_samples: ", generated_samples.shape)
    return generated_samples



if __name__ == "__main__":
    args = AttnWGAIN_arguments()
    #
    unified_dataloader = UnifiedDataLoader(args.data_path, args.seq_len, args.feature_num, args.model_type,
                                            args.hint_rate,
                                            args.miss_rate, args.batch_size, args.num_workers, args.MIT)
    #
    test_dataloader = unified_dataloader.get_test_dataloader()
    #
    device = torch.device("cuda")
    print(f"Current device: {device}")  # 输出当前设备
    #
    # # 调用 new_train 获取 evalMask_collector
    evalMask_collector = new_train(test_dataloader)
    #
    # # 将 evalMask_collector 传递给 M_prob_gen3
    # M_matrix, M_matrix_prob = M_prob_gen(evalMask_collector)

    size = (100, 300, 128)  # 原始输入（B, L, K）
    # data_sample_shape = (300, 128)
    data_sample_shape = (evalMask_collector.shape[1], evalMask_collector.shape[2])
    # M_matrix = torch.randint(0, 2, size, dtype=torch.float32)  # 生成一个随机的二值矩阵 (0 或 1)  # 生成随机 0/1 二值矩阵
    M_matrix = evalMask_collector
    M_matrix_prob = M_prob_gen(M_matrix)
    print("M_matrix: ", M_matrix.shape, "M_matrix_prob: ", M_matrix_prob.shape)

    # 初始化VAE模型
    latent_dim = 256
    vae = VAE(data_sample_shape, latent_dim).to(device)  # 閫傞厤
    # train VEA model
    print("Starting VAE training...")
    vae = train_vae(vae, M_matrix_prob, num_epochs=10000, batch_size=256)  # 璁粌 VAE
    print("VAE training finished.")

    # # 加载 VAE 模型权重，允许部分层的权重不匹配
    try:
        vae.load_state_dict(torch.load("vae_model.pth", weights_only=True), strict=False)
        vae.to("cpu")
        generated_samples = generate_multiple_samples(vae, num_samples=evalMask_collector.shape[0])
        print("VAE model loaded successfully.")
        print("generated_samples: ", generated_samples.shape, generated_samples[0, 0, :10])
        train_M_matrix = torch.where(generated_samples > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        print("train missing ratio: ", torch.sum(train_M_matrix) / train_M_matrix.numel())
    except Exception as e:
        print(f"Error loading model: {e}")





