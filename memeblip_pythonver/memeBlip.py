from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torch import nn
import torch
import pytorch_lightning as pl
import torchmetrics
from torch.nn import functional as F
from Adapter import Adapter
from LinearProjection import LinearProjection
from CosineClassifierWithBias import ImprovedClassifier
from torchmetrics.functional import f1_score
from CrossModelAttention import CrossModalAttention

class MemeBLIP(pl.LightningModule):
    def __init__(self, cfg,  use_adapter=True, use_text_project=True, use_image_project=True, use_pre_output_layer=True, use_cosine_classifier=True, fusion_mode="attention"):
        super().__init__()
        self.cfg = cfg
        self.use_adapter = use_adapter
        self.use_text_project = use_text_project
        self.use_image_project = use_image_project
        self.use_pre_output_layer = use_pre_output_layer
        self.use_cosine_classifier = use_cosine_classifier
        self.fusion_mode = fusion_mode  # Controlled directly by args


        if self.use_image_project:
            # 动态线性投影
            self.image_projection = LinearProjection(
                input_dim=1408,
                output_dim=cfg.map_dim,
                num_layers=1,
                drop_probs=cfg.drop_probs
            ).to(self.cfg.device)
        else:
            print("不使用图像投影层")
            self.image_projection = nn.Linear(1408, cfg.map_dim).to(self.cfg.device)
        
        if self.use_text_project:
            # 动态线性投影
            self.text_projection = LinearProjection(
                input_dim=768,
                output_dim=cfg.map_dim,
                num_layers=1,
                drop_probs=cfg.drop_probs
            ).to(self.cfg.device)
        else:
            print("不使用文本投影层")
            # 直接使用线性投影
            self.text_projection = nn.Linear(768, cfg.map_dim).to(self.cfg.device)
        
        # Adapter
        if self.use_adapter:
            print("使用Adapter")
            self.image_adapter = Adapter(cfg.map_dim, reduction=2).to(self.cfg.device)
            self.text_adapter = Adapter(cfg.map_dim, reduction=2).to(self.cfg.device)
            x = torch.randn(16, 1024, device=cfg.device)  # batch_size=16，特征维度512
            print(self.image_adapter.fc)
            output = self.image_adapter(x)
            print("输出维度:", output.shape)
        else:
            print("不使用Adapter")
            self.image_adapter = nn.Linear(cfg.map_dim, cfg.map_dim).to(self.cfg.device)
            self.text_adapter = nn.Linear(cfg.map_dim, cfg.map_dim).to(self.cfg.device)
        
        self.cross_attn = CrossModalAttention(cfg.hidden_dim)

        if self.use_pre_output_layer:
            print("使用Pre-Output层")
            self.pre_output_layer = nn.Sequential(
                nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )
        else:
            print("不使用Pre-Output层")
            self.pre_output_layer = nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim).to(self.cfg.device)
            for p in self.pre_output_layer.parameters():  # freeze this layer
                p.requires_grad = False

        # 加载 BLIP 模型和处理器
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16
        ).to(self.cfg.device)
        self.model.eval()

        self.map_dim = cfg.map_dim  # BLIP 模型的隐藏层大小

        # 分类器
        if self.use_cosine_classifier:
            print("使用自定义分类器")
            # self.classifier = nn.Sequential(
            #     nn.Linear(cfg.hidden_dim, 512),
            #     nn.LayerNorm(512),
            #     nn.GELU(),
            #     nn.Dropout(p=0.5),
            #     CosineClassifierWithBias(512, cfg.num_classes)
            # )
            self.classifier = ImprovedClassifier(cfg.hidden_dim, cfg.num_classes).to(self.cfg.device)
        else:
            print("使用线性分类器")
            self.classifier = nn.Linear(cfg.hidden_dim, cfg.num_classes).to(self.cfg.device)
        # 初始化分类器权重
        self.init_head_text_feat()

        # 损失函数
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # 评估指标
        self.acc = torchmetrics.Accuracy(task="binary", num_classes=cfg.num_classes)
        self.auroc = torchmetrics.AUROC(task="binary", num_classes=cfg.num_classes)
        self.f1 = torchmetrics.F1Score(task="binary", num_classes=cfg.num_classes)
        self.model = self.model.to(cfg.device)
        self.classifier = self.classifier.to(cfg.device)
        # 存储梯度
        self.gradients = {}

    def save_gradient(self, name):
        # 定义保存梯度的钩子
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
        return hook

    def print_gradients(self, modules_to_check, batch_idx):
        for mod_name, module in modules_to_check.items():
            for name, param in module.named_parameters():
                if param.grad is not None:
                    grad_mean = param.grad.abs().mean().item()
                    grad_std = param.grad.std().item()
                    print(f"[Batch {batch_idx}] {mod_name} 梯度 {name}: mean_abs={grad_mean:.8f}, std={grad_std:.8f}")
                else:
                    print(f"[Batch {batch_idx}] {mod_name} 梯度 {name}: 无梯度")


    def register_hooks(self):
        # 为关键层注册钩子
        self.image_projection.register_backward_hook(self.save_gradient("image_projection"))
        self.text_projection.register_backward_hook(self.save_gradient("text_projection"))
        # 检查是否有 fc 属性（适配器结构）
        if hasattr(self.image_adapter, 'fc'):
            self.image_adapter.fc.register_backward_hook(self.save_gradient("image_adapter"))
        else:
            # 简单线性层情况
            self.image_adapter.register_backward_hook(self.save_gradient("image_adapter"))
            
        if hasattr(self.text_adapter, 'fc'):
            self.text_adapter.fc.register_backward_hook(self.save_gradient("text_adapter"))
        else:
            self.text_adapter.register_backward_hook(self.save_gradient("text_adapter"))

        self.pre_output_layer.register_backward_hook(self.save_gradient("pre_output_layer"))


        if isinstance(self.classifier, nn.Sequential):
            for i, layer in enumerate(self.classifier):
                if isinstance(layer, nn.Linear):
                    layer.register_backward_hook(self.save_gradient(f"classifier_{i}"))
        else:
            # 如果 classifier 不是 nn.Sequential，直接注册钩子
            self.classifier.register_backward_hook(self.save_gradient("classifier"))
        # for i, layer in enumerate(self.classifier):
        #     if isinstance(layer, nn.Linear):
        #         layer.register_backward_hook(self.save_gradient(f"classifier_{i}"))

    def init_head_text_feat(self):
        print("Initialize head with text features")

        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in self.cfg.class_names]

        prompts = self.processor.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.cfg.device)

        prompts = {k: v for k, v in prompts.items() if k in ["input_ids", "attention_mask"]}

        encoded_text = self.model.language_model.model(
            input_ids=prompts["input_ids"],
            attention_mask=prompts["attention_mask"],
            return_dict=True
        )
        text_features = encoded_text.last_hidden_state.mean(dim=1)
        text_features = F.normalize(text_features, dim=-1)

        target_dim = 512

        if text_features.size(1) != target_dim:
            # Create the projection in float16 too
            projection = nn.Linear(
                text_features.size(1), 512
            ).to(self.cfg.device, dtype=torch.float16)
            text_features = projection(text_features)

        if isinstance(self.classifier, nn.Sequential) and hasattr(self.classifier[-1], "apply_weight"):
            if text_features.size(1) != target_dim:
                projection = nn.Linear(text_features.size(1), 512).to(self.cfg.device)
                text_features = projection(text_features)
                self.classifier[-1].apply_weight(text_features)
        else:
            print("Warning: Classifier -1 does not have 'apply_weight' method. Skipping initialization.")

    def forward(self, batch):
        # 提取特征
        image_features = batch['image_features']
        text_features = batch['text_features']

        if isinstance(image_features, tuple):
          image_features = image_features[0].to(self.cfg.device)
        if isinstance(text_features, tuple):
          text_features = text_features[0].to(self.cfg.device)

        image_dim = image_features.size(1)
        text_dim = text_features.size(1)

        # Linear Projection
        if self.use_image_project:
            image_proj = self.image_projection(image_features).to(self.cfg.device)
        else:
            # 如果没有投影层，确保维度依然匹配
            image_proj = image_features
            if image_dim != self.map_dim:
                image_proj = nn.Linear(image_dim, self.map_dim).to(self.cfg.device)(image_proj)
        
        if self.use_text_project:
            text_proj = self.text_projection(text_features).to(self.cfg.device)
        else:
            # 确保文本维度一致
            text_proj = text_features
            if text_dim != self.map_dim:
                text_proj = nn.Linear(text_dim, self.map_dim).to(self.cfg.device)(text_proj)
    
        # 确保在乘法操作前两者维度一致
        assert image_proj.size(1) == text_proj.size(1), f"Feature dimensions don't match: {image_proj.size()} vs {text_proj.size()}"
    
        # Adapter
        if self.use_adapter:
            adapted_image = self.image_adapter(image_proj).to(self.cfg.device)
            adapted_text = self.text_adapter(text_proj).to(self.cfg.device)
        else:
            adapted_image = image_proj
            adapted_text = text_proj

        #text_adapted_features = self.cfg.ratio  * adapted_text + (1 - self.cfg.ratio ) * text_proj
        #image_adapted_features = self.cfg.ratio  * adapted_image + (1 - self.cfg.ratio ) * image_proj

        #image_adapted_features = image_adapted_features / image_adapted_features.norm(dim=-1, keepdim=True)
        #text_adapted_features = text_adapted_features / text_adapted_features.norm(dim=-1, keepdim=True)

        if self.fusion_mode == "attention":
            i2t = self.cross_attn(adapted_image, adapted_text).to(self.cfg.device)
            t2i = self.cross_attn(adapted_text, adapted_image).to(self.cfg.device)
            combined_features = torch.cat([i2t, t2i], dim=-1).to(self.cfg.device)
            residual = torch.cat([adapted_image, adapted_text], dim=-1).to(self.cfg.device)  # (B, 2D)
            combined_features = combined_features + 0.5 * residual
        elif self.fusion_mode == "mul":
            combined_features = adapted_image * adapted_text
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")


        # 特征融合
        #combined_features = torch.mul(image_adapted_features, text_adapted_features).to(self.cfg.device)

        # Pre-Output Transformation
        pre_output_features = self.pre_output_layer(combined_features).to(self.cfg.device)

        logits = self.classifier(pre_output_features).squeeze(dim=1).to(self.cfg.device)
        return logits

    def common_step(self, batch):
        logits = self.forward(batch)  # 使用分类器的输出
        # 使用 softmax 获取概率分布
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        # 使用原始的 logits 计算损失
        loss = self.cross_entropy_loss(logits, batch["labels"])
        
        # 计算 AUROC
        auroc_input = probs[:, 1]
        auroc = self.auroc(auroc_input, batch["labels"])
        # 使用硬预测计算 accuracy 和 F1
        acc = self.acc(preds, batch["labels"])
        f1 = self.f1(preds, batch["labels"])

        # preds_proxy = torch.sigmoid(logits)
        # _ , preds = logits.data.max(1)
        # loss = self.cross_entropy_loss(logits, batch["labels"])  # 标签大小为 [batch_size]
        # # preds = torch.argmax(logits, dim=-1)
        # acc = self.acc(preds, batch["labels"])
        # auroc = self.auroc(preds_proxy, batch['labels'])
        # f1 = self.f1(preds, batch["labels"])
        return {"loss": loss, "acc": acc, "auroc": auroc, "f1": f1}

    def training_step(self, batch, batch_idx):
        #logits = self.forward(batch)
        #loss = self.cross_entropy_loss(logits, batch["labels"])
        #preds_proxy = torch.sigmoid(logits)
        #_ , preds = logits.data.max(1)
        #acc = self.acc(preds, batch["labels"])
        #auroc = self.auroc(preds_proxy, batch['labels'])
        #f1 = self.f1(preds, batch["labels"])

        logits = self.forward(batch)  # (B, 2)
        labels = batch["labels"]

        # Loss
        loss = self.cross_entropy_loss(logits, labels)

        # 分类预测（用于 Accuracy 和 F1）
        preds = torch.argmax(logits, dim=1)

        # 用于 AUROC：使用 logits[:, 1] 表示正类置信度
        auroc_input = logits[:, 1]

        # Metrics
        acc = self.acc(preds, labels)
        f1 = self.f1(preds, labels)
        auroc = self.auroc(auroc_input, labels)


        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_auroc", auroc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_step=True, on_epoch=True, prog_bar=True)

        # 每30个batch打印一次梯度信息（可调）
        # if batch_idx > 0 and batch_idx % 20 == 0:
        #     modules_to_check = {
        #         "image_projection": self.image_projection,
        #         "text_projection": self.text_projection,
        #         "image_adapter": self.image_adapter,
        #         "text_adapter": self.text_adapter,
        #         "pre_output_layer": self.pre_output_layer,
        #         "classifier": self.classifier,
        #     }
        #     self.print_gradients(modules_to_check, batch_idx)
            
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            # Backbone (BLIP 原始模型) - 小weight_decay
            {"params": self.model.parameters(), "weight_decay": 1e-5},
    
            # Image/Text Projection层 - 中等weight_decay
            {"params": list(self.image_projection.parameters()) +
                       list(self.text_projection.parameters()), "weight_decay": 5e-4},
    
            # Adapter层 - 强weight_decay
            {"params": list(self.image_adapter.parameters()) +
                       list(self.text_adapter.parameters()), "weight_decay": 1e-3},
    
            # Pre-output 层 - 中强weight_decay
            {"params": self.pre_output_layer.parameters(), "weight_decay": 5e-4},
    
            # 分类器层 - 最强weight_decay
            {"params": self.classifier.parameters(), "weight_decay": 1e-3},
        ], lr=self.cfg.lr)
    
        # Warmup + Cosine LR不变
        warmup_epochs = 3
        total_epochs = self.cfg.max_epochs
        cosine_epochs = total_epochs - warmup_epochs
    
        scheduler_warmup = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler_cosine = CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=1e-6
        )
    
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs]
        )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

    def validation_step(self, batch, batch_idx):
        # 前向传播
        logits = self.forward(batch)
        loss = self.cross_entropy_loss(logits, batch["labels"])

        #preds_proxy = torch.sigmoid(logits)
        #_ , preds = logits.data.max(1)
        # 预测和计算指标
        # preds = torch.argmax(logits, dim=-1)
        # 分类预测（用于 Accuracy 和 F1）
        preds = torch.argmax(logits, dim=1)

        # 用于 AUROC：使用 logits[:, 1] 表示正类置信度
        auroc_input = logits[:, 1]

        # Metrics
        acc = self.acc(preds, batch["labels"])
        f1 = self.f1(preds, batch["labels"])
        auroc = self.auroc(auroc_input, batch["labels"])
        #acc = self.acc(preds, batch["labels"])
        #auroc = self.auroc(torch.softmax(logits, dim=-1), batch["labels"])
        #f1 = self.f1(preds, batch["labels"])

        # 日志记录
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

        return {"loss": loss, "acc": acc, "auroc": auroc, "f1": f1}

    def on_validation_epoch_end(self):
        # 拿到 Lightning 最终聚合好的所有指标
        metrics = self.trainer.callback_metrics
        # 里头应该包含: "val_loss", "val_acc", "val_auroc", "val_f1" 等
        epoch = self.current_epoch
    
        # 这里若取不到键，说明还没在 self.log() 里记录
        val_loss = metrics["val_loss"].item()
        val_acc = metrics["val_acc"].item()
        val_auroc = metrics["val_auroc"].item()
        val_f1 = metrics["val_f1"].item()
    
        print(f"Epoch {epoch} - val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, "
              f"val_auroc: {val_auroc:.4f}, val_f1: {val_f1:.4f}")
        
    def on_train_epoch_end(self):
        # 每个epoch结束时释放显存缓存
        torch.cuda.empty_cache()

def initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)  # Xavier 初始化
            if module.bias is not None:
                nn.init.zeros_(module.bias)