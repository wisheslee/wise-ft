# WiSE-FT 学习指南

## 项目概述

WiSE-FT (Weight-Space Ensembles for Fine-Tuning) 通过权重空间插值融合zero-shot模型和fine-tuned模型，在保持分布内准确率的同时提升分布外鲁棒性。

核心思想：`theta = (1-alpha) * theta_zeroshot + alpha * theta_finetuned`

---

## 代码架构

### 核心文件功能

1. **src/wise_ft.py** - 主流程
   - `wise_ft()`: 完整流程（训练或加载→融合→评估）
   - `_merge()`: 权重融合核心实现

2. **src/models/modeling.py** - 模型定义
   - `ImageEncoder`: CLIP图像编码器封装
   - `ClassificationHead`: 分类头（线性层）
   - `ImageClassifier`: 完整分类器

3. **src/models/zeroshot.py** - Zero-shot分类器
   - `get_zeroshot_classifier()`: 构建文本权重分类头

4. **src/models/finetune.py** - 微调训练
   - `finetune()`: 训练循环，支持冻结encoder或端到端

5. **src/models/eval.py** - 评估
   - `eval_single_dataset()`: 单数据集评估
   - `evaluate()`: 多数据集评估并记录结果

6. **src/args.py** - 参数配置
   - `parse_arguments()`: 统一参数管理

---

## 学习路径

### 阶段1：理解基础概念

#### 1.1 CLIP模型基础
- CLIP = Contrastive Language-Image Pre-training
- 双编码器：Image Encoder + Text Encoder
- 训练目标：图像-文本对的对比学习

#### 1.2 Zero-shot分类原理

**流程**：
```
1. 类别名 → Prompt模板 → 文本描述
   例：cat → ["a photo of a cat", "a picture of a cat", ...]

2. 文本描述 → Text Encoder → 文本特征
   embeddings = clip_model.encode_text(texts)

3. 多个prompt的特征平均 → 类别原型
   class_embedding = embeddings.mean(dim=0).normalize()

4. 所有类别原型 → 分类头权重矩阵
   W = [class_1_emb, class_2_emb, ..., class_N_emb]^T

5. 推理：图像特征 × 权重矩阵 = logits
   logits = image_features @ W^T
```

**关键代码** (src/models/zeroshot.py:34-57):
```python
for classname in dataset.classnames:
    texts = [template(classname) for template in templates]
    texts = clip.tokenize(texts)
    embeddings = clip_model.encode_text(texts)
    embeddings = embeddings.mean(dim=0).normalize()
    zeroshot_weights.append(embeddings)
```

#### 1.3 Fine-tuning策略

**策略A：Linear Probe** (--freeze-encoder)
- 冻结CLIP image encoder
- 仅训练分类头
- 适用场景：小数据集、快速适配

**策略B：End-to-End Fine-tuning**
- 整个模型参与训练
- 更强的适应能力
- 但可能过拟合、丢失zero-shot泛化能力

#### 1.4 WiSE-FT核心思想

**问题**：Fine-tuning虽然提升ID准确率，但损害OOD鲁棒性

**解决方案**：权重空间线性插值
```python
# 简单但有效！
theta = (1-alpha) * theta_zeroshot + alpha * theta_finetuned
```

**为什么有效**？
- Zero-shot模型：泛化能力强，但ID准确率一般
- Fine-tuned模型：ID准确率高，但OOD泛化差
- 插值：在两者之间找平衡点

**Alpha调优**：
- alpha=0: 纯zero-shot
- alpha=0.3~0.7: 通常最佳平衡点
- alpha=1: 纯fine-tuned

---

### 阶段2：代码深度解析

#### 2.1 模型结构 (src/models/modeling.py)

**ImageEncoder**:
```python
class ImageEncoder:
    def __init__(self, args, keep_lang=False):
        # 加载CLIP模型（包含图像和文本编码器）
        self.model, self.train_preprocess, self.val_preprocess = clip.load(args.model)

        # 如果不需要文本编码器，删除以节省内存
        if not keep_lang:
            delattr(self.model, 'transformer')

    def forward(self, images):
        return self.model.encode_image(images)
```

**ClassificationHead**:
```python
class ClassificationHead(nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        # weights: [num_classes, feature_dim]
        # normalize: 是否归一化输入特征（zero-shot需要）
        self.normalize = normalize
        self.weight = nn.Parameter(weights)

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)  # 线性变换
```

**ImageClassifier**:
```python
class ImageClassifier(nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images  # 控制是否先编码图像

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)  # 图像→特征
        return self.classification_head(inputs)  # 特征→logits
```

#### 2.2 Zero-shot构建流程 (src/models/zeroshot.py)

**完整流程**：
```python
def get_zeroshot_classifier(args, clip_model):
    # 1. 加载数据集获取类别名
    dataset = dataset_class(location=args.data_location, classnames=args.classnames)

    # 2. 加载prompt模板
    template = getattr(templates, args.template)
    # 例如：openai_imagenet_template = [
    #     lambda c: f"a photo of a {c}",
    #     lambda c: f"a rendering of a {c}",
    #     ...
    # ]

    # 3. 对每个类别生成文本embeddings
    zeroshot_weights = []
    for classname in dataset.classnames:
        # 应用所有prompt模板
        texts = [t(classname) for t in template]
        # 例如：["a photo of a cat", "a rendering of a cat", ...]

        # Tokenize文本
        texts = clip.tokenize(texts)

        # 编码文本
        embeddings = clip_model.encode_text(texts)  # [num_prompts, dim]
        embeddings /= embeddings.norm(dim=-1, keepdim=True)

        # 平均所有prompt的embeddings
        embeddings = embeddings.mean(dim=0, keepdim=True)
        embeddings /= embeddings.norm()

        zeroshot_weights.append(embeddings)

    # 4. 堆叠为权重矩阵
    zeroshot_weights = torch.stack(zeroshot_weights, dim=0)  # [N, 1, dim]
    zeroshot_weights = zeroshot_weights.squeeze()  # [N, dim]
    zeroshot_weights = zeroshot_weights.T  # [dim, N]

    # 5. 应用logit scale（CLIP的温度参数）
    zeroshot_weights *= clip_model.logit_scale.exp()

    # 6. 创建分类头
    return ClassificationHead(normalize=True, weights=zeroshot_weights.T)
```

**重点理解**：
- Prompt ensemble：多个prompt描述同一类别，提升鲁棒性
- 特征归一化：CLIP训练时使用，推理时必须保持
- Logit scale：控制温度，影响softmax分布

#### 2.3 微调训练流程 (src/models/finetune.py)

**关键部分**：
```python
def finetune(args):
    # 1. 加载预训练模型
    image_classifier = ImageClassifier.load(args.load)

    # 2. 确定微调模式
    if args.freeze_encoder:
        # 模式A：仅训练分类头
        model = image_classifier.classification_head
        input_key = 'features'  # 输入预提取特征
        image_enc = image_classifier.image_encoder
    else:
        # 模式B：端到端训练
        model = image_classifier
        input_key = 'images'  # 输入原始图像
        image_enc = None

    # 3. 准备数据
    dataset = dataset_class(preprocess_fn, location=args.data_location, ...)

    # 4. 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, total_steps)

    # 5. 训练循环
    for epoch in range(args.epochs):
        for batch in dataloader:
            # 前向传播
            inputs = batch[input_key]  # images或features
            labels = batch['labels']
            logits = model(inputs)

            # 损失计算
            loss = loss_fn(logits, labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)  # 梯度裁剪
            optimizer.step()
            scheduler(step)

        # 6. 每个epoch后评估和保存
        evaluate(image_classifier, args)
        image_classifier.save(f'checkpoint_{epoch+1}.pt')
```

**训练技巧**：
- Cosine learning rate schedule with warmup
- 梯度裁剪防止梯度爆炸
- 可选label smoothing正则化
- 每个epoch保存checkpoint

#### 2.4 WiSE-FT融合 (src/wise_ft.py)

**主函数**：
```python
def wise_ft(args):
    # 场景1：从零开始
    if args.load is None:
        # 步骤1：构建并保存zero-shot模型
        image_encoder = ImageEncoder(args, keep_lang=True)
        classification_head = get_zeroshot_classifier(args, image_encoder.model)
        classifier = ImageClassifier(image_encoder, classification_head)
        zeroshot_checkpoint = 'zeroshot.pt'
        classifier.save(zeroshot_checkpoint)

        # 步骤2：微调训练
        finetuned_checkpoint = finetune(args)

    # 场景2：使用已有模型
    else:
        zeroshot_checkpoint, finetuned_checkpoint = args.load

    # 步骤3：加载模型权重
    zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    finetuned = ImageClassifier.load(finetuned_checkpoint)
    theta_0 = zeroshot.state_dict()
    theta_1 = finetuned.state_dict()

    # 步骤4：对每个alpha进行融合和评估
    for alpha in args.alpha:
        # 融合权重
        theta = _merge(alpha, theta_0, theta_1, fishers=None, fisher_floor=1e-8)

        # 加载融合权重
        finetuned.load_state_dict(theta)

        # 保存融合模型
        finetuned.save(f'wise_ft_alpha={alpha:.3f}.pt')

        # 评估
        evaluate(finetuned, args)
```

**融合函数详解**：
```python
def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    # 基础版本：简单线性插值
    if fishers is None:
        return {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

    # 高级版本：Fisher信息加权
    fisher_0, fisher_1 = fishers
    theta = {}
    for key in theta_0.keys():
        # 获取Fisher信息（参数重要性度量）
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        # Fisher加权插值
        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1
        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta
```

**Fisher加权的直觉**：
- Fisher信息衡量参数对损失的敏感度
- 敏感参数（重要参数）给予更大权重
- 更精细的融合策略，但需要额外计算Fisher矩阵

#### 2.5 评估流程 (src/models/eval.py)

**单数据集评估**：
```python
def eval_single_dataset(image_classifier, dataset, args):
    # 1. 确定输入类型
    if args.freeze_encoder:
        model = image_classifier.classification_head
        input_key = 'features'
        image_enc = image_classifier.image_encoder
    else:
        model = image_classifier
        input_key = 'images'
        image_enc = None

    # 2. 遍历数据集
    correct, n = 0, 0
    for batch in dataloader:
        x = batch[input_key]
        y = batch['labels']

        # 3. 推理
        logits = model(x)

        # 4. 可选的logit映射（处理类别不对齐）
        if hasattr(dataset, 'project_logits'):
            logits = dataset.project_logits(logits)

        # 5. 计算准确率
        pred = logits.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        n += y.size(0)

    top1_accuracy = correct / n
    return {'top1': top1_accuracy}
```

**多数据集评估**：
```python
def evaluate(image_classifier, args):
    results = {}
    for dataset_name in args.eval_datasets:
        # 加载数据集
        dataset = getattr(datasets, dataset_name)(...)

        # 评估
        metrics = eval_single_dataset(image_classifier, dataset, args)
        results[dataset_name] = metrics

        print(f"{dataset_name} Top-1: {metrics['top1']:.4f}")

    # 保存结果到JSONL文件
    if args.results_db:
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(results) + '\n')

    return results
```

---

### 阶段3：实战操作步骤

#### 3.1 环境配置

```bash
# 创建conda环境
cd /Users/jili/github-code/wise-ft
conda env create -f environment.yml
conda activate wiseft

# 设置Python路径
export PYTHONPATH="$PYTHONPATH:$PWD"
```

**注意**：environment.yml针对Linux CUDA环境，macOS需要调整：
- 移除CUDA相关包
- 使用CPU版本PyTorch

#### 3.2 数据准备

**选择数据集**：
- 完整ImageNet：需要约150GB存储
- 小规模测试：使用CIFAR-10/100（自动下载）

```bash
# 设置数据路径
export DATA_LOCATION=~/data
mkdir -p $DATA_LOCATION

# 下载ImageNet distribution shifts（可选）
cd $DATA_LOCATION

# ImageNet-V2
wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz
tar -xvf imagenetv2-matched-frequency.tar.gz

# ImageNet-R
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar -xvf imagenet-r.tar

# ImageNet-A
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar -xvf imagenet-a.tar
```

#### 3.3 快速测试（使用CIFAR-10）

**目的**：在小数据集上验证流程

```bash
# 仅评估zero-shot性能
python src/models/zeroshot.py \
    --model=ViT-B/32 \
    --train-dataset=CIFAR10 \
    --eval-datasets=CIFAR10 \
    --template=openai_imagenet_template \
    --data-location=$DATA_LOCATION

# 微调训练
python src/models/finetune.py \
    --model=ViT-B/32 \
    --train-dataset=CIFAR10 \
    --epochs=5 \
    --lr=0.0001 \
    --batch-size=128 \
    --eval-datasets=CIFAR10 \
    --load=models/cifar10_zeroshot.pt \
    --save=models/cifar10_finetuned

# WiSE-FT融合
python src/wise_ft.py \
    --load=models/cifar10_zeroshot.pt,models/cifar10_finetuned/checkpoint_5.pt \
    --eval-datasets=CIFAR10 \
    --save=models/wiseft \
    --data-location=$DATA_LOCATION \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --results-db=results_cifar10.jsonl
```

#### 3.4 完整流程（ImageNet）

```bash
# 一键运行：训练 + 融合 + 评估
python src/wise_ft.py \
    --train-dataset=ImageNet \
    --model=ViT-B/32 \
    --epochs=10 \
    --lr=0.00003 \
    --batch-size=512 \
    --warmup_length=500 \
    --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch \
    --template=openai_imagenet_template \
    --save=models/wiseft_vitb32 \
    --data-location=$DATA_LOCATION \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --results-db=results_vitb32.jsonl
```

**预期输出**：
```
Getting zeroshot weights: 100%|██████████| 1000/1000
Saving model to models/wiseft_vitb32/zeroshot.pt
Fine-tuning end-to-end
Train Epoch: 0 [0% 0/2502] Loss: 4.234567
...
Saving model to models/wiseft_vitb32/finetuned/checkpoint_10.pt
Evaluating alpha=0.0
ImageNet Top-1 accuracy: 0.6342
ImageNetV2 Top-1 accuracy: 0.5456
...
Evaluating alpha=0.5
ImageNet Top-1 accuracy: 0.7123
ImageNetV2 Top-1 accuracy: 0.6245
...
```

#### 3.5 结果分析

**查看结果**：
```bash
# 查看JSONL结果
cat results_vitb32.jsonl | jq .

# 提取关键指标
cat results_vitb32.jsonl | jq '{alpha, ImageNet_top1: .["ImageNet:top1"], ImageNetV2_top1: .["ImageNetV2:top1"]}'
```

**生成可视化**：
```bash
python src/scatter_plot.py \
    --eval-datasets=ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch \
    --results-db=results_vitb32.jsonl \
    --save plots/vitb32
```

**预期观察**：
- Alpha=0（zero-shot）：ID准确率中等，OOD鲁棒性强
- Alpha=1（fine-tuned）：ID准确率高，OOD性能下降
- Alpha=0.3~0.7：ID和OOD最佳平衡

---

### 阶段4：深入实验

#### 4.1 对比实验：冻结 vs 端到端

**实验A：仅训练分类头**
```bash
python src/wise_ft.py \
    --train-dataset=ImageNet \
    --model=ViT-B/32 \
    --epochs=10 \
    --freeze-encoder \
    --lr=0.001 \
    --save=models/linear_probe \
    --alpha 0 0.5 1.0 \
    --results-db=results_linear.jsonl
```

**实验B：端到端微调**
```bash
python src/wise_ft.py \
    --train-dataset=ImageNet \
    --model=ViT-B/32 \
    --epochs=10 \
    --lr=0.00003 \
    --save=models/end2end \
    --alpha 0 0.5 1.0 \
    --results-db=results_end2end.jsonl
```

**对比分析**：
- Linear probe：训练快，泛化好，但ID提升有限
- End-to-end：ID提升大，但需要更小学习率和更多正则化

#### 4.2 超参数调优

**学习率搜索**：
```bash
for lr in 0.00001 0.00003 0.0001 0.0003; do
    python src/wise_ft.py \
        --train-dataset=ImageNet \
        --model=ViT-B/32 \
        --epochs=10 \
        --lr=$lr \
        --save=models/lr_$lr \
        --alpha 0.5 \
        --results-db=results_lr_sweep.jsonl
done
```

**Alpha网格搜索**（精细化）：
```bash
python src/wise_ft.py \
    --load=models/zeroshot.pt,models/finetuned.pt \
    --alpha 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 \
    --eval-datasets=ImageNet,ImageNetV2 \
    --results-db=results_alpha_fine.jsonl
```

#### 4.3 不同CLIP模型对比

```bash
# 测试多个模型架构
for model in RN50 RN101 ViT-B/32 ViT-B/16 ViT-L/14; do
    python src/wise_ft.py \
        --train-dataset=ImageNet \
        --model=$model \
        --epochs=10 \
        --save=models/$model \
        --alpha 0 0.5 1.0 \
        --results-db=results_models.jsonl
done
```

**预期发现**：
- 更大模型（ViT-L/14）：zero-shot性能更强，微调提升空间小
- 小模型（RN50）：zero-shot性能弱，微调提升空间大
- WiSE-FT对所有模型都有效

#### 4.4 Fisher加权实验（高级）

**步骤1：计算Fisher信息**
```bash
# 需要实现Fisher计算（src/models/fisher.py）
python src/models/fisher.py \
    --model=models/zeroshot.pt \
    --dataset=ImageNet \
    --save=fisher_zeroshot.pt

python src/models/fisher.py \
    --model=models/finetuned.pt \
    --dataset=ImageNet \
    --save=fisher_finetuned.pt
```

**步骤2：Fisher加权融合**
```bash
python src/wise_ft.py \
    --load=models/zeroshot.pt,models/finetuned.pt \
    --fisher=fisher_zeroshot.pt,fisher_finetuned.pt \
    --fisher_floor=1e-8 \
    --alpha 0.5 \
    --eval-datasets=ImageNet,ImageNetV2
```

---

### 阶段5：代码修改与扩展

#### 5.1 添加自定义数据集

**创建新数据集类** (`src/datasets/my_dataset.py`):
```python
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MyDataset:
    def __init__(self, preprocess, location, batch_size=32, **kwargs):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size

        # 定义类别名
        self.classnames = ['class1', 'class2', 'class3']

        # 构建数据集
        self.train_dataset = MyDatasetImpl(
            root=os.path.join(location, 'my_dataset/train'),
            transform=preprocess
        )
        self.test_dataset = MyDatasetImpl(
            root=os.path.join(location, 'my_dataset/test'),
            transform=preprocess
        )

        # 创建DataLoader
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )

class MyDatasetImpl(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        # 扫描图像文件
        self.samples = self._load_samples()

    def _load_samples(self):
        # 实现数据加载逻辑
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
```

**注册数据集** (`src/datasets/__init__.py`):
```python
from .my_dataset import MyDataset
```

**使用自定义数据集**：
```bash
python src/wise_ft.py \
    --train-dataset=MyDataset \
    --model=ViT-B/32 \
    --data-location=$DATA_LOCATION
```

#### 5.2 自定义Prompt模板

**创建模板** (`src/templates.py`):
```python
# 添加新模板
my_custom_template = [
    lambda c: f"这是一张{c}的照片",
    lambda c: f"一个{c}的图像",
    lambda c: f"{c}",
]
```

**使用模板**：
```bash
python src/wise_ft.py \
    --template=my_custom_template \
    --train-dataset=ImageNet
```

#### 5.3 实现渐进融合

**创建新脚本** (`src/progressive_merge.py`):
```python
import torch
from src.models.modeling import ImageClassifier
from src.models.eval import evaluate

def progressive_merge(zeroshot_path, finetuned_path, args):
    """渐进式融合：从zero-shot逐步过渡到fine-tuned"""
    zeroshot = ImageClassifier.load(zeroshot_path)
    finetuned = ImageClassifier.load(finetuned_path)

    theta_0 = zeroshot.state_dict()
    theta_1 = finetuned.state_dict()

    # 分层融合：不同层使用不同alpha
    layer_groups = {
        'encoder.early': 0.2,  # 早期层更接近zero-shot
        'encoder.middle': 0.5,
        'encoder.late': 0.7,
        'classifier': 0.9      # 分类头更接近fine-tuned
    }

    theta = {}
    for key in theta_0.keys():
        # 根据层名确定alpha
        alpha = determine_alpha(key, layer_groups)
        theta[key] = (1-alpha) * theta_0[key] + alpha * theta_1[key]

    finetuned.load_state_dict(theta)
    evaluate(finetuned, args)
```

#### 5.4 添加实时监控

**使用Weights & Biases**：
```python
import wandb

# 在finetune.py中添加
wandb.init(project="wise-ft", name=args.exp_name)

# 训练循环中记录
wandb.log({
    "train/loss": loss.item(),
    "train/lr": optimizer.param_groups[0]['lr'],
    "epoch": epoch
})

# 评估后记录
wandb.log({
    f"eval/{dataset_name}": metrics['top1']
    for dataset_name, metrics in results.items()
})
```

---

## 常见问题与调试

### Q1: CUDA out of memory
**解决方案**：
- 减小batch size：`--batch-size=64`
- 使用梯度累积
- 冻结encoder：`--freeze-encoder`

### Q2: 训练不收敛
**检查**：
- 学习率是否过大（end-to-end用0.00003，linear probe用0.001）
- 是否使用warmup：`--warmup_length=500`
- 数据预处理是否正确

### Q3: Alpha=1结果不等于fine-tuned模型
**原因**：
- 浮点数精度问题
- 模型保存/加载时的状态差异
- 应该非常接近，误差<0.1%

### Q4: Zero-shot性能异常低
**检查**：
- Prompt模板是否适合数据集
- 类别名是否正确
- 是否正确归一化特征

---

## 进阶学习资源

### 论文阅读
1. **原论文**：[Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903)
2. **CLIP论文**：[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
3. **Model soups**：[Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482)

### 相关技术
- **Task Arithmetic**: 任务向量代数
- **Model Merging**: 多模型融合策略
- **Few-shot Learning**: 小样本学习

### 扩展方向
1. **多模态任务**：视频、音频+文本
2. **零样本目标检测**：扩展到检测任务
3. **参数高效微调**：LoRA + WiSE-FT
4. **持续学习**：在新任务上微调并融合

---

## 总结

WiSE-FT是一个简单但强大的方法，核心思想只有一行代码，但效果显著。通过本指南，你应该：

✅ 理解CLIP zero-shot分类原理
✅ 掌握fine-tuning的两种策略
✅ 理解权重空间融合的动机和实现
✅ 能够运行完整的训练-融合-评估流程
✅ 学会分析和可视化结果
✅ 具备扩展和修改代码的能力

**下一步行动**：
1. 运行快速CIFAR-10实验，验证环境
2. 阅读核心代码（modeling.py, wise_ft.py）
3. 在小数据集上做实验，理解超参数影响
4. 尝试自定义数据集和prompt模板
