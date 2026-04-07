import math
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor

from models.dgcnn import GeoFeatGenerator
from HEM.models.Rotary_Transformer import SwinEncoder, SwinConfig



import numpy as np

import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist

from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import tempfile

import warnings
warnings.filterwarnings("ignore")



class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm1d(256),
        )
        self.activation = nn.LeakyReLU()
        
    def forward(self, x):
        residual = x
        x = self.conv_block(x)
        x = x + residual  # 残差连接
        x = self.activation(x)
        return x

class AgentSA(pl.LightningModule):
    def __init__(self, cfg):
        super(AgentSA, self).__init__()
        self.cfg = cfg
        # 初始化验证阶段的比特数累加器
        self.train_step_outputs = []
        self.validation_step_outputs = []
        # 初始化测试阶段的比特数累加器
        self.test_step_outputs = []

        self.glscfg = None


        self.geo_feat_generator = GeoFeatGenerator(max_level=cfg.model.max_level)

        swin_cfg = SwinConfig(
            num_channels=256,
            embed_dim=256,
            depths=[10, 4, 2],
            num_heads=[4, 4, 4],
            window_size=512,
        )

        self.swin_self_transformer = SwinEncoder(swin_cfg, 8192, False)

        cross_swin_cfg = SwinConfig(
            num_channels=256,
            embed_dim=256,
            depths=[4, 2, 1],#2，2，1，1，
            # depths=[2, 2, 1, 1],
            num_heads=[4, 4, 4],
            window_size=512,
        )
        
        self.swin_cross_transformer = SwinEncoder(cross_swin_cfg, 4096, True)

        self.ancient_mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )


        self.output_pred_mlp1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

        self.pre_occ_mlp = nn.Sequential(
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )

        self.pre_attn_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 240),
            nn.GELU(),
            nn.Linear(240, 240),
        )

        self.output_pred_mlp2 = nn.Sequential(
            nn.Linear(1024, 768),
            nn.GELU(),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        self.prob1 = nn.Linear(256, 255)
        self.prob2 = nn.Linear(256, 255)

        self.num1 = nn.Linear(256,8)
        self.num2 = nn.Linear(256,8)


        # 1. 专门用于 feat_a1 的邻域卷积模块
        self.context_conv_block_a1 = ResidualBlock()

        # 2. 方案一中已有的融合网络
        self.fusion_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, 256)
        )

        self.save_hyperparameters("cfg")
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters(cfg)

    def repeat_state(self, state, csz, bsz, dim):
        return state.repeat(1, 2, 1).reshape(bsz, -1, csz, dim).transpose(1, 2).reshape(bsz, -1, dim)

    def concat_states(self, hidden_states):
        bsz, _, dim = hidden_states[0].shape
        states = []
        for i in range(len(hidden_states)-1, 1, -1):
            state = hidden_states[i]
            cur_csz = state.shape[1]
            for j in range(len(states)):
                states[j] = self.repeat_state(states[j], cur_csz, bsz, dim)[:, :hidden_states[i-1].shape[1]]
            state = self.repeat_state(state, cur_csz, bsz, dim)[:, :hidden_states[i-1].shape[1]]
            states.append(state)
        states.append(hidden_states[1])
        return torch.concat(states[::-1], 2)

    def forward(self, data, extents,pos, enc=False):
        '''
        data: bsz, context size, ancients + current node (4), level + octant + occ (3)
        extents: bsz, context size, 2 (z_extents, x_extents)
        '''
        padded = False
        if data.shape[1] % 2 == 1:
            padded = True
            pad = torch.zeros_like(data[:, :1])
            pad[:, :, :, 2] = 255
            data = torch.cat((data, pad), dim=1)
            extent_pad = torch.zeros_like(extents[:, :1])
            extents = torch.cat((extents, extent_pad), dim=1)
            pos_pad = torch.zeros_like(pos[:, :, :1])
            pos = torch.cat((pos, pos_pad), dim=2)

        bsz = data.shape[0]
        csz = data.shape[1]

        pre_occ = data[:, ::2, -1, -1]
        data = data.reshape(bsz, csz, -1)[:, :, :-1] # bsz, csz, 11. 11: 4*(level, oct, occ), except occ of current voxel

        feat = self.geo_feat_generator(data, extents,pos)

        new_feat = feat

        self_output = self.swin_self_transformer(new_feat, csz, code = None, output_hidden_states=True, output_hidden_states_before_downsampling=True)
        self_output = self.concat_states(self_output.hidden_states)
        new_feat_a = self.ancient_mlp(self_output)

        feat_a = new_feat_a
        # feat_a = new_feat_a

        feat_a1 = feat_a[:, ::2]
        feat_a2 = feat_a[:, 1::2]
        feat1 = self.output_pred_mlp1(feat_a1)

        prob1 = self.prob1(feat1)

        num1 = self.num1(feat1)

        # 2. 然后，只对 feat_a1 应用卷积，增强其内部上下文
        # feat_a1 形状: (B, L/2, C) -> permute -> (B, C, L/2)
        feat_a1_permuted = feat_a1.permute(0, 2, 1)
        feat_a1_contextual_permuted = self.context_conv_block_a1(feat_a1_permuted)
        # 换回维度 -> (B, L/2, C)
        feat_a1_contextual = feat_a1_contextual_permuted.permute(0, 2, 1)
        
        pre_occ_embed = self.geo_feat_generator.embed_occ(pre_occ)
        pre_occ_feat = self.pre_occ_mlp(pre_occ_embed)
        pre_attn_feat = self.pre_attn_mlp(feat_a1_contextual)



        # 4. 使用 FusionMLP 进行智能融合
        combined_feat = torch.concat((pre_occ_feat, pre_attn_feat), dim=2)
        pre_feat = self.fusion_mlp(combined_feat) 

        cross_output = self.swin_cross_transformer(pre_feat, feat_a2.shape[1], query=feat_a2, output_hidden_states=True, output_hidden_states_before_downsampling=True)
        cross_output = self.concat_states(cross_output.hidden_states)
        feat_a2 = torch.concat((cross_output, feat_a2), 2)

        feat2 = self.output_pred_mlp2(feat_a2)
        
        if padded:
            feat2 = feat2[:, :-1]

        prob2 = self.prob2(feat2)
        num2 = self.num2(feat2)

        if not enc:
            probs = torch.zeros((prob1.shape[0], prob1.shape[1] + prob2.shape[1], prob1.shape[2])).to(prob1.device)
            probs[:, ::2] = prob1
            probs[:, 1::2] = prob2

            nums = torch.zeros((num1.shape[0], num1.shape[1] + num2.shape[1], num1.shape[2])).to(num1.device)
            nums[:, ::2] = num1
            nums[:, 1::2] = num2
            return probs, nums
        else :
            return prob1, prob2


    def decode(self, data, pos, pre_occ=None):
        '''
        data: bsz, context size, ancients + current node (4), level + octant + occ (3)
        '''
        padded = False
        if data.shape[1] % 2 == 1:
            padded = True
            pad = torch.zeros_like(data[:, :1])
            pad[:, :, :, 2] = 255
            data = torch.cat((data, pad), dim=1)
            pos_pad = torch.zeros_like(pos[:, :, :1])
            pos = torch.cat((pos, pos_pad), dim=2)

        bsz = data.shape[0]
        csz = data.shape[1]

        if pre_occ is None:
            data = data.reshape(bsz, csz, -1)[:, :, :-1] # bsz, csz, 11. 11: 4*(level, oct, occ), except occ of current voxel

            feat = self.geo_feat_generator(data, pos)
            self_output = self.swin_self_transformer(feat, csz, output_hidden_states=True, output_hidden_states_before_downsampling=True)
            self_output = self.concat_states(self_output.hidden_states)
            feat_a = self.ancient_mlp(self_output)

            self.feat_a1 = feat_a[:, ::2]
            self.feat_a2 = feat_a[:, 1::2]
            prob1 = self.output_pred_mlp1(self.feat_a1)
            return prob1

        pre_occ_embed = self.geo_feat_generator.embed_occ(pre_occ)
        pre_occ_feat = self.pre_occ_mlp(pre_occ_embed)
        pre_attn_feat = self.pre_attn_mlp(self.feat_a1)

        pre_feat = torch.concat((pre_occ_feat, pre_attn_feat), dim=2)
        cross_output = self.swin_cross_transformer(pre_feat, csz//2, query=self.feat_a2, output_hidden_states=True, output_hidden_states_before_downsampling=True)
        cross_output = self.concat_states(cross_output.hidden_states)

        feat_a2 = torch.concat((cross_output, self.feat_a2), 2)
        prob2 = self.output_pred_mlp2(feat_a2)

        if padded:
            prob2 = prob2[:, :-1]

        return prob2


    # <<< 新增方法：專門用於載入和凍結 Swin 模塊 >>>
    def load_and_freeze_swin_modules(self, checkpoint_path):
        """
        從一個大的預訓練模型 checkpoint 中，精準地為 swin 模塊載入權重，
        並只解凍 swin 模塊中的 mona 部分。
        """
        print(f"--- 開始從 {checkpoint_path} 精準載入和凍結 Swin 權重 ---")
        
        # 1. 載入整個大模型的預訓練 state_dict
        #    Lightning 的 checkpoint 格式為 {'state_dict': ...}
        full_state_dict = torch.load(checkpoint_path, map_location=self.device)['state_dict']

        # 2. 定義要操作的目標模塊名稱列表
        target_modules = ["swin_self_transformer", "swin_cross_transformer"]

        for module_name in target_modules:
            print(f"\n處理模塊: {module_name}")
            
            # 獲取模塊實例
            submodule = getattr(self, module_name)

            # 3. 從 full_state_dict 中篩選出只屬於當前子模塊的權重
            sub_state_dict = {}
            prefix = module_name + '.'
            for key, value in full_state_dict.items():
                if key.startswith(prefix):
                    # 關鍵：去掉前綴 (e.g., "swin_self_transformer.layer1...") 
                    # -> "layer1..."
                    sub_key = key[len(prefix):]
                    sub_state_dict[sub_key] = value
            
            if not sub_state_dict:
                print(f"警告：在 checkpoint 中找不到任何鍵帶有前綴 '{prefix}' 的權重。")
                continue

            # 4. 為子模塊載入權重 (strict=False 以兼容新增的 mona 參數)
            submodule.load_state_dict(sub_state_dict, strict=False)
            print(f"成功為 {module_name} 載入 {len(sub_state_dict)} 個參數層。")

            # 5. 精準凍結/解凍
            # 首先，凍結該子模塊的所有參數
            for param in submodule.parameters():
                param.requires_grad = False
            
            # 然後，只解凍該子模塊內部的 mona 參數
            found_mona = False
            for name, param in submodule.named_parameters():
                if "mona" in name:
                    param.requires_grad = True
                    found_mona = True
            
            if found_mona:
                print(f"{module_name} 的 Mona 部分已解凍，其餘部分已凍結。")
            else:
                print(f"警告：在 {module_name} 中未找到 Mona 參數進行解凍。")
        
        print("\n--- 所有 Swin 模塊處理完畢 ---")


    def configure_optimizers(self):
        print("\n--- 配置優化器 ---")
        
        # 關鍵：篩選出整個模型中所有 requires_grad = True 的參數
        # 這樣，無論您是只想訓練 Mona，還是也想訓練其他模塊，
        # 這裡的代碼都無需改變，它會自動找到所有需要訓練的參數。
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        if not trainable_params:
            print("警告：沒有找到任何可訓練的參數！請檢查您的凍結邏輯。")
            # 如果真的沒有可訓練參數，可以返回 None 或引發錯誤
            return None

        print(f"找到 {len(trainable_params)} 個可訓練的參數張量。")
        # 打印部分可訓練參數的名稱以供檢查
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"  - [不可訓練] {name}")
        optim_cfg = self.cfg.train.optimizer
        sched_cfg = self.cfg.train.lr_scheduler
        if optim_cfg.name == "Adam":
            optimizer = torch.optim.Adam(trainable_params, lr=self.cfg.train.lr)
        elif optim_cfg.name == "AdamW":
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.cfg.train.lr,
                weight_decay=optim_cfg.weight_decay,
            )
        else:
            raise NotImplementedError()
        if sched_cfg.name == "StepLR":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma
            )
        else:
            raise NotImplementedError()

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    

    def compute_bitrate_transformer(self,pred,gt):
        batch_size = pred.shape[0]
        sequence_size = pred.shape[1]
        logit = F.softmax(pred,dim=2)

        one_hot_gt = torch.zeros(batch_size,sequence_size,self.cfg.model.token_num)
        one_hot_gt = one_hot_gt.scatter_(dim=2,index=gt.reshape(batch_size,sequence_size,1).data.long().cpu(),value=1)
        ans = torch.mul(one_hot_gt.cuda(),logit.cuda())
        ans[ans == 0] = 1
        bits = -torch.log2(ans).sum()
        return bits


    def training_step(self, batch): 
        data, extents, pos, labels, nums = batch
        prob, num = self(data, extents, pos)
        loss1 = self.criterion(
            prob.view(-1, self.cfg.model.token_num), labels.long().reshape(-1)
        ) / math.log(2)
        loss2 = self.criterion(
            num.view(-1, 8), nums.long().reshape(-1)
        ) / math.log(2)
        self.train_step_outputs.append(loss1.item())
        
        self.log("occupancy_loss", np.mean(self.train_step_outputs), on_step=True, on_epoch=False, prog_bar=True)
        self.log("num_loss", loss2, on_step=True, on_epoch=False, prog_bar=True)
        return loss1 + loss2
    

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is None:
                print(f"Unused parameter: {name}")

    
    def validation_step(self, batch, batch_idx):
        data, extents, pos, labels, _ = batch
        pred, _ = self(data, extents, pos)
        loss = self.criterion(
            pred.view(-1, self.cfg.model.token_num), labels.reshape(-1)
        ) / math.log(2)
        self.log('val_loss', loss,  on_step=False, on_epoch=True, prog_bar=True)

        # 计算当前批次的比特数
        bits = self.compute_bitrate_transformer(pred, labels)
        # 累加验证阶段的比特数
        self.validation_step_outputs.append(bits)


    def test_step(self, batch, batch_idx):
        data, extents, pos, labels = batch
        pred = self(data, extents, pos)
        loss = self.criterion(
            pred.view(-1, self.cfg.model.token_num), labels.reshape(-1)
        ) / math.log(2)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # 计算当前批次的比特数
        bits = self.compute_bitrate_transformer(pred, labels)
        # 累加测试阶段的比特数
        self.test_step_outputs.append(bits)


    # 1. 在每个验证epoch开始时，初始化用于存储各step输出的列表
    def on_validation_epoch_start(self):
        self.validation_step_outputs = []


    def on_validation_epoch_end(self):
        # 计算验证阶段的平均 BPP
        # 聚合所有进程的结果（分布式训练安全）
        all_bits = self.all_gather(self.validation_step_outputs)

        stacked_bits = torch.stack(all_bits)
        total_bits = torch.sum(stacked_bits)


        # total_bits = sum(torch.cat(all_bits))
        total_points = self.cfg.train.val_total_points
        
        # 计算 BPP（需处理除零错误）
        val_bpp = total_bits / total_points if total_points != 0 else Tensor(0.0)
        
        # 记录指标（自动同步多进程）
        self.log("val_bpp", val_bpp, prog_bar=True, on_epoch=True, sync_dist=True)
        # # 重置验证阶段的累加器
        self.validation_step_outputs.clear() 


    def on_test_epoch_end(self):
        # 计算test阶段的平均 BPP
        # 聚合所有进程的结果（分布式训练安全）
        all_bits = self.all_gather(self.test_step_outputs)
        stacked_bits = torch.stack(all_bits)
        total_bits = torch.sum(stacked_bits)
        # total_bits = sum(torch.cat(all_bits))
        total_points = self.cfg.train.test_total_points
        
        # 计算 BPP（需处理除零错误）
        test_bpp = total_bits / total_points if total_points != 0 else Tensor(0.0)
        
        # 记录指标（自动同步多进程）
        self.log("test_bpp", test_bpp, prog_bar=True,  on_epoch=True, sync_dist=True)
        # # 重置测试阶段的累加器
        self.test_step_outputs.clear() 


    def load_pretrain(self, path, strict=True):
        """
        仅加载预训练模型的权重，忽略其他信息（如配置）
        """
        try:
            # 尝试以 weights_only=True 加载模型权重
            checkpoint = torch.load(path, weights_only=True)
            # 检查是否包含 state_dict 或 model_state_dict
            if 'state_dict' in checkpoint:
                sd = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                sd = checkpoint['model_state_dict']
            else:
                # 如果都没有，假设整个检查点就是模型权重
                sd = checkpoint
        except Exception as e:
            # 如果 weights_only=True 失败，尝试以不安全方式加载
            print(f"警告: 以不安全方式加载检查点: {e}")
            checkpoint = torch.load(path, weights_only=False)
            if 'state_dict' in checkpoint:
                sd = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                sd = checkpoint['model_state_dict']
            else:
                sd = checkpoint
        
        # 过滤不匹配的键
        ref_sd = self.state_dict()
        keys = list(sd.keys())
        for k in keys:
            if k not in ref_sd or ref_sd[k].shape != sd[k].shape:
                sd.pop(k)
                print(f"移除不匹配的键: {k}")
        
        return self.load_state_dict(sd, strict)


# input : (N,4,6) -> [selfOcc,layer,index,x,y,z]
# input : (N,4,8) -> [selfOcc,layer,index,extent,z_extent,x,y,z]

