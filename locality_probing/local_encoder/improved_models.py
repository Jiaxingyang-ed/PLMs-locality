import torch
from torch import nn

class DeepLocalSeqEncoder(nn.Module):
    """升级版局部序列编码器：更深、支持可变窗口和 ESM-2 初始化"""
    def __init__(self, vocab_size=20, embed_dim=64, hidden_dim=256, output_dim=1024,
                 num_layers=4, nhead=4, max_window_len=41, use_esm_init=False):
        super().__init__()
        self.max_window_len = max_window_len
        # 词嵌入
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if use_esm_init:
            # 尝试加载 ESM-2 8M 的嵌入层权重（需要先下载）
            try:
                from transformers import EsmModel
                esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
                # 只复制嵌入权重（注意 ESM-2 词汇表包含特殊 token，这里简单截取）
                self.embed.weight.data[:vocab_size] = esm.embeddings.word_embeddings.weight.data[:vocab_size]
                print("ESM-2 嵌入权重加载成功")
            except:
                print("ESM-2 加载失败，使用随机初始化")
        self.pos_embed = nn.Embedding(max_window_len, embed_dim)  # 可选位置编码
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                                   dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, seq_idx, mask=None):
        # seq_idx: (batch, seq_len)
        batch, seq_len = seq_idx.shape
        x = self.embed(seq_idx)  # (batch, seq_len, embed_dim)
        # 添加位置编码（可选）
        positions = torch.arange(seq_len, device=seq_idx.device).unsqueeze(0).expand(batch, -1)
        x = x + self.pos_embed(positions)
        x = self.transformer(x, src_key_padding_mask=mask)
        # 取平均值池化（或使用 CLS token）
        x = x.mean(dim=1)  # (batch, embed_dim)
        return self.fc(x)   # (batch, output_dim)