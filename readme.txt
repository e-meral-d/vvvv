项目概览
--------
本项目聚焦“零样本、对象无关”的校园视频异常检测方案，目标是实现论文《一个用于校园视频异常检测的零样本、对象无关的增强框架》提出的 TAnomalyCLIP 模型。整体思路如下：
- 静态视觉表示：采用预训练并冻结的 `open_clip` ViT-L/14@336px 模型，从抽帧的图像中提取通用视觉特征。
- 时序建模：通过专用的 Transformer 编码器学习帧间动态模式，捕捉追逐、打闹等连续动作。
- 语义注入（SKI）：借助 CLIP 文本编码器预先编码的行为基元（正常与异常语义短语），在时序特征上执行跨模态注意力式的语义增强。
- 异常检测头：沿用 AnomalyCLIP 的可学习提示机制，通过对增强特征与“正常/异常”提示的相似度比较，输出帧级与视频级异常分数。

建成后模型应具备以下能力：
1. 训练只使用非校园公开异常数据（如 UCF-Crime），测试可直接在校园监控视频上推断异常，无需微调，实现零样本泛化。
2. 输出帧级概率序列，支持 AUC/Precision/Recall/F1 等评估指标，并可生成可视化（时间分数曲线、空间热力图）。
3. CLI 接口一键完成训练、评估与可视化操作，方便后续集成与部署。

当前目录结构（核心部分）
------------------------
- configs/
  - dataset_ucf.yaml：训练数据加载示例配置（需替换真实路径、batch 等参数）。
  - dataset_campus.yaml：校园测试/评估数据加载示例配置。
  - annotations/
    - ucf_train.yaml：UCF-Crime 训练集注释模板（使用占位路径）。
    - campus_test.yaml：校园测试集注释模板（使用占位路径）。
- src/
  - data/：视频 clip 切分、帧采样、图像增强、DataLoader 构建。
  - models/：模型核心组件（时序编码器、语义注入、主模型）。
  - utils/：日志工具。
- train.py / evaluate.py / visualize.py：训练、评估、可视化命令行脚本的入口骨架。
- requirements.txt：依赖清单，建议配合 `.venv` 虚拟环境安装。
- readme.txt：当前项目说明文件。

数据与注释
----------
- 真实数据尚未接入，`configs/annotations/*.yaml` 中使用 `<relative_path_to_...>` 作为占位符。待数据准备后，请将其替换为相对项目根目录的真实路径，并补充异常时间段（秒）。
- 数据管线实现于 `src/data/`，默认参数：8 fps 抽帧、clip 长度 16 帧、滑窗步长 8 帧、分辨率 336x336，并完成 CLIP 风格的归一化。
- DataLoader 会输出 `frames (C,T,H,W)`、clip 标签、帧级标签、时间戳等信息，后续训练和评估脚本可直接消费。

模型组件
--------
- TemporalEncoder (`src/models/temporal_encoder.py`)：TransformerEncoder 结构，含可学习位置编码与层归一化，负责对帧特征序列建模。
- SKIModule (`src/models/ski_module.py`)：内置行为语义库（正常/异常短语），先通过 CLIP 文本编码器得到先验向量，再以 sigmoid 权重注入到时序特征，输出拼接后的语义增强表示。
- TAnomalyCLIP (`src/models/t_anomalyclip.py`)：加载并冻结 CLIP ViT-L/14；将帧特征输入 TemporalEncoder 与 SKIModule；使用可学习提示的 PromptAnomalyHead 生成帧级、视频级异常得分，并可返回语义注入中间结果。
- 模型整体预计完成以下功能：
  1. 支持端到端前向推理，输出帧级 logits、视频级 logits 及提示向量。
  2. 可选返回中间特征，方便调试或可视化异常热力图。
  3. 保留接口以便后续加入损失函数、多尺度异常检测、Prompt 微调等扩展。

环境准备
--------
1. 进入项目根目录 `d:\code\ucfac`。
2. 创建虚拟环境：`python -m venv .venv`。
3. 激活：`.\.venv\Scripts\activate`。
4. 安装依赖：`pip install -r requirements.txt`。

代码结构与关键模块
------------------
- 数据管线（`src/data/`）
  - `dataset.py`：读取注释文件，将长视频按 16 帧 clip 切片，执行 8 fps 抽帧、重采样与帧级标签时间对齐。
  - `transforms.py`：提供 CLIP 风格归一化及轻量数据增强（可选随机水平翻转）。
  - `dataloader.py`：结合 YAML/JSON 注释和配置参数，构建标准的 PyTorch DataLoader。
- 模型组件（`src/models/`）
  - `temporal_encoder.py`：多层 TransformerEncoder，带可学习位置编码与 LayerNorm，用于时序建模。
  - `ski_module.py`：内置正常/异常行为语义库，调用 CLIP 文本编码器获得先验嵌入，并以 sigmoid 权重注入时序特征。
  - `t_anomalyclip.py`：整合冻结的 CLIP ViT-L/14、TemporalEncoder、SKIModule 与 PromptAnomalyHead，输出帧级/视频级 logits 以及中间特征。
- 公共工具（`src/utils/`）
  - `config.py`、`data.py`：加载 YAML 配置并构建 DataLoader，可覆盖 batch、线程、shuffle 等参数。
  - `metrics.py`：纯 NumPy 实现二分类 AUC、Precision/Recall/F1。
  - `evaluation.py`：在推理时收集分数、视频 ID、clip 起止时间，统一调用指标计算。
  - `model.py`：封装 `TAnomalyCLIP` 构建逻辑（是否冻结骨干等）。
  - `checkpoint.py`：保存与恢复模型/优化器状态。
  - `__init__.py`：集中导出常用工具，便于脚本直接引用。
- 训练/评估脚本
  - `train.py`：命令行训练入口，支持配置化数据加载、帧/视频级 BCE 损失、混合精度、阈值化指标统计、断点恢复与历史记录写入（JSON）。
  - `evaluate.py`：加载 checkpoint 对测试集推理，输出同套指标，可将详细分数 JSON 落盘为后续可视化输入。
  - `visualize.py`：占位脚本，后续可基于评估分数生成异常分数曲线与热力图。

使用步骤
--------
1. 将真实视频文件存放在 `data/ucf/`（训练）与 `data/video/`（测试）目录，并在 `configs/annotations/ucf_train.yaml`、`configs/annotations/campus_test.yaml` 中替换 `<relative_path_to_...>` 占位符，同时填写异常时间段。
2. 训练：`python train.py --dataset-config configs/dataset_ucf.yaml --val-annotation configs/annotations/campus_test.yaml --val-data-root data/video --output-dir outputs/train_run`。脚本将自动记录指标与 checkpoint。
3. 评估：`python evaluate.py --dataset-config configs/dataset_campus.yaml --checkpoint outputs/train_run/checkpoint_epoch_XXX.pt --output outputs/eval_metrics.json --save-scores outputs/eval_scores.json`。
4. 可视化：待在 `visualize.py` 中实现帧分数曲线、热力图导出逻辑后，可使用同一套注释与分数文件生成可视化结果。
5. 根据需要扩展损失函数、调参脚本以及模型导出/部署流程，完善端到端解决方案。
