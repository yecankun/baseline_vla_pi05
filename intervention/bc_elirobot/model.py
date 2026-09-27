import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FrameMultimodalFusion(nn.Module):
    """使用 Transformer 融合 ResNet 图像特征 + pose + count（单帧）"""

    def __init__(self, d_model=128, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        resnet = resnet18(weights=weights)

        # ResNet18 去掉 fc，输出 [B, 512, 1, 1]
        self.img_backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.img_proj = nn.Linear(512, d_model)
        self.pose_proj = nn.Linear(6, d_model)
        self.count_proj = nn.Linear(1, d_model)

        # 3 个 token：img / pose / count
        self.pos_embed = nn.Parameter(torch.randn(1, 3, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image, pose, count):
        # image: [B, C, H, W]
        # pose : [B, 6]
        # count: [B] or [B, 1]
        img_feat = self.img_backbone(image).flatten(1)  # [B, 512]
        img_token = self.img_proj(img_feat).unsqueeze(1)  # [B, 1, d_model]
        pose_token = self.pose_proj(pose).unsqueeze(1)    # [B, 1, d_model]

        if count.dim() == 1:
            count = count.unsqueeze(-1)                   # [B, 1]
        count_token = self.count_proj(count).unsqueeze(1) # [B, 1, d_model]

        tokens = torch.cat([img_token, pose_token, count_token], dim=1)  # [B, 3, d_model]
        tokens = tokens + self.pos_embed
        tokens = self.dropout(tokens)

        fused_tokens = self.transformer(tokens)           # [B, 3, d_model]
        fused_feat = fused_tokens.mean(dim=1)             # [B, d_model]
        return fused_feat


class SingleFrameElirobotPolicyModel(nn.Module):
    """
    单帧策略：
    - 输入：image, pose, count, task_id
    - 输出：
        large_pred: [B, 6]   大臂 6 维回归
        small_logits: [B]    小臂 Bernoulli logits
    """

    def __init__(self, d_model=256, nhead=8, num_transformer_layers=2,
                 num_tasks=2, dropout=0.1):
        super().__init__()
        self.frame_fusion = FrameMultimodalFusion(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dropout=dropout
        )
        self.task_emb = nn.Embedding(num_tasks, d_model)

        # 用 MLP 代替 LSTM 输出头的输入特征
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 大臂：6维回归
        self.large_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

        # # 小臂：二分类 logits
        # self.small_logits_head = nn.Sequential(
        #     nn.Linear(d_model, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )

    def forward(self, image, pose, count, task_id):
        """
        image: [B, C, H, W]
        pose : [B, 6]
        count: [B] or [B, 1]
        task_id: [B] (Long)
        """
        feat = self.frame_fusion(image, pose, count)       # [B, d_model]
        feat = feat + self.task_emb(task_id)               # [B, d_model]
        feat = self.trunk(feat)                            # [B, d_model]

        large_pred = self.large_head(feat)                 # [B, 6]
        return large_pred

    # inference / ppo 接口（单帧）
    def get_action(self, image, pose, count, task_id):
        large_pred = self.forward(image, pose, count, task_id)

        return large_pred

    def evaluate_actions(self, image, pose, count, task_id, large_action, small_action):
        """
        说明：
        - large 是回归（无分布），这里给 PPO 占位 log_prob/entropy=0
        - small 是 Bernoulli，返回其 log_prob 与 entropy（更合理）
        """
        large_pred, small_logits = self.forward(image, pose, count, task_id)

        small_dist = torch.distributions.Bernoulli(logits=small_logits)
        small_action = small_action.float()  # Bernoulli 需要 0/1 float
        small_log_prob = small_dist.log_prob(small_action)  # [B]
        small_entropy = small_dist.entropy()                # [B]

        # 大臂回归不提供概率项，占位为0（或你也可以设计为高斯分布做 PPO）
        batch_size = large_pred.shape[0]
        zero = torch.zeros(batch_size, device=large_pred.device)

        total_log_prob = zero + small_log_prob
        total_entropy = zero + small_entropy
        return total_log_prob, total_entropy, large_pred