# Python训练营进阶部分

本项目包含一系列Python深度学习算法和计算机视觉实现的练习。你需要补全 `exercises` 目录下的 Python 文件中缺失的代码，并通过测试来验证实现的正确性。

## 练习内容

### 第1部分: 深度学习算法基础
1.  `exercises/iou.py`: 实现目标检测中的交并比 (IoU) 计算。
2.  `exercises/nms.py`: 实现目标检测中的非极大值抑制 (NMS) 算法 (包含 IoU 计算)。
3.  `exercises/conv.py`: 手写实现二维卷积操作。
4.  `exercises/leaky_relu.py`: 实现 LeakyReLU 激活函数。
5.  `exercises/maxpool.py`: 实现最大池化操作。
6.  `exercises/cross_entropy.py`: 实现交叉熵损失函数。
7.  `exercises/smooth_l1.py`: 实现 Smooth L1 损失函数。

### 第2部分: 计算机视觉基础
8.  `exercises/image_processing.py`: 使用 OpenCV 实现图像边缘检测等基本处理。
9.  `exercises/contour_detection.py`: 使用 OpenCV 实现图像轮廓检测。

## 使用方法

1.  **Fork 本仓库**: 将此仓库 Fork 到你自己的 GitHub 账户。
2.  **克隆仓库**: 将你 Fork 的仓库克隆到本地计算机：
    ```bash
    git clone https://github.com/YOUR_USERNAME/Python-Training-Camp-Advanced.git
    cd Python-Training-Camp-Advanced
    ```
    (将 `YOUR_USERNAME` 替换为你的 GitHub 用户名)
3.  **设置环境**: 参考下方的 "环境要求" 和 "运行方法" 安装必要的依赖。
4.  **完成练习**: 
    *   打开 `exercises/` 目录中的练习文件。
    *   仔细阅读文件顶部的说明和代码中的注释。
    *   在标记为 `# TODO:` 或包含 `pass` 的地方编写代码。
5.  **本地测试**: 使用 "运行方法" 中的命令在本地测试代码。
6.  **提交与评分**: 
    *   将修改后的代码 `git add`, `git commit`, 并 `git push` 到你 Fork 的仓库。
    *   推送代码后，GitHub Actions 会自动运行测试并计算分数。
    *   访问你 Fork 仓库的 "Actions" 页面查看结果。
    *   (若 Actions 未自动运行，请手动启用或触发)
    *   在 Actions 日志中查找 "Print test results score" 步骤查看分数，并在 "Summary" 页面下载构建产物获取详细报告。

## 建议学习顺序

建议按照练习文件名的隐含顺序或上述列表顺序学习，从基础算法到图像处理应用：
1.  先掌握目标检测的基础算法 (`iou.py`, `nms.py`)。
2.  学习深度学习的核心组件实现 (`conv.py`, `leaky_relu.py`, `maxpool.py`, `cross_entropy.py`, `smooth_l1.py`)。
3.  最后学习 OpenCV 图像处理 (`image_processing.py`, `contour_detection.py`)。

## 环境要求
*   Python 3.10
*   依赖库见 `requirements.txt` (主要包括 `numpy`, `opencv-python`, `pytest`)

## 目录结构

```
Python-Training-Camp-Advanced/
├── .github/workflows/     # GitHub Actions 工作流配置
│   └── test.yml
├── exercises/             # 学员练习目录 (需要填空的 .py 文件)
│   ├── contour_detection.py
│   ├── conv.py
│   ├── cross_entropy.py
│   ├── image_processing.py
│   ├── iou.py
│   ├── leaky_relu.py
│   ├── maxpool.py
│   ├── nms.py
│   └── smooth_l1.py
├── picture/               # 可能包含测试用的图片资源
├── tests/                 # Pytest 测试文件
│   ├── test_contour_detection.py
│   ├── test_conv.py
│   ├── # ... (其他测试文件)
│   └── test_smooth_l1.py
├── .gitignore             # Git 忽略文件配置
├── README.md              # 项目说明文件 (本文件)
├── requirements.txt       # Python 依赖库列表
├── score_calculator.py    # 用于计算分数的脚本
└── test_score.txt         # 分数计算结果（示例或运行时生成）
```

## 运行方法

1.  **安装依赖**:
    ```bash
    # 建议在虚拟环境中操作
    # python -m venv .venv
    # source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate  # Windows
    pip install -r requirements.txt
    ```
2.  **本地运行测试**:
    ```bash
    # 运行所有测试
    python -m pytest tests/ -v

    # 运行特定测试文件 (例如 iou.py)
    python -m pytest tests/test_iou.py -v
    ```
3.  **自动测试与评分 (GitHub Actions)**:
    *   将您的代码 `git push` 到您 Fork 的 GitHub 仓库。
    *   稍等片刻，进入 GitHub 仓库页面，点击 "Actions" 标签。
    *   找到最新的工作流运行实例，点击进入查看详情（若未触发自动测试，可能是因为一开始未启动Actions功能，手动启动测试即可）。
    *   您可以查看测试日志、`pytest` 的输出以及 `score_calculator.py` 生成的分数报告 (`test_score.txt` 会作为 artifact 上传)。

## 注意事项
- 每个练习文件中都有详细的注释说明和提示，请仔细阅读。
- 需要填写的代码部分主要是完成带有 `pass` 语句的函数体。

## 评分标准

当您推送代码后，GitHub Actions 会自动执行以下流程：
1.  运行 `pytest` 对您在 `exercises/` 目录下的代码进行测试。
2.  即使部分测试失败，流程也会继续 (`continue-on-error: true`)。
3.  运行 `score_calculator.py` 脚本，该脚本可能会根据 `pytest` 的测试结果（例如解析生成的 `test-results.xml` 文件）来计算一个分数。
4.  最终的分数会保存在 `test_score.txt` 文件中。
5.  您可以在 GitHub Actions 的运行日志中看到分数报告的打印输出。
6.  `test_score.txt` 和 `test-results.xml` 文件会作为构建产物 (Artifacts) 上传，您可以在 Actions 运行详情页面下载它们以查看详细结果。

最终评价基于 `score_calculator.py` 计算出的分数。 

## 学习

进阶阶段：计算机视觉与目标检测 - 慧眼识物，洞察秋毫！ (1-2 周)
目标： 为最终目标检测项目做技术准备，深入理解并实践目标检测中常用的函数、模型组件及训练技巧，尤其侧重 OpenCV 在图像预处理和后处理中的应用，并结合 YOLO 算法进行实践！通过练习评测与AI评测以及人工评测后达到指定分数后，进入最终项目阶段。

学习资料：

● OpenCV 官方文档：https://docs.opencv.org/ (掌握图像处理的基础工具)

● YOLO论文原文 (Redmonetal., 2016)：https://pjreddie.com/darknet/yolo/ (论文链接) (理解 YOLO 算法的设计思想)

● SSD 论文 (Liuetal.,2016)：https://arxiv.org/abs/1512.02325 (Single Shot MultiBox Detector, 了解另一种经典的目标检测算法)

● Faster R-CNN 论文 (Renetal., 2015)：https://arxiv.org/abs/1506.01497 (Two-stage目标检测的代表作)

● GitHub 上的小型目标检测项目： (参考代码思路，避免直接使用，搜索关键词: "object detection github")

晋级指标：

完成实验Python-Training-Camp-Advanced

完成步骤（具体步骤请参考仓库README中的内容）：



将仓库 Fork 到自己的 GitHub 账户，形成个人仓库副本
将仓库clone到本地，并在本地完成实验并运行本地测试
将本地实验push提交至个人 Github 仓库中自动运行评测
若运行结果达到满分即可自动晋级，若未达标则需修改代码后重复提交运行直至通过。
自动运行测评成功后请查看晋级排行榜，若未显示请检查个人信息中的·Github name是否填写正确
确保理解每个核心概念，能够手写实现关键函数和组件，并能熟练运用 OpenCV 进行图像处理。

你将学到什么？

● 图像处理基石： 掌握 OpenCV 库的使用，包括图像读取 (cv2.imread)、颜色空间转换 (cv2.cvtColor)、滤波 (cv2.GaussianBlur)、几何变换等，为目标检测提供高质量的输入！

● 数据集操控： 熟悉目标检测数据集的处理流程，包括加载、预处理、数据增强等，为模型训练准备充足的“养料”！

● 核心函数复现： 手写实现 IoU (Intersection over Union)、NMS (Non-Maximum Suppression) 等关键函数，深入理解目标检测算法的底层原理！

● 组件级构建： 手写实现卷积层、ReLU激活函数、MaxPool 池化层等神经网络组件，掌握深度学习模型的基本构成！

● 损失函数设计： 理解分类损失（如 Cross Entropy Loss）和回归损失（如 Smooth L1 Loss）在目标检测中的作用，并能够手写实现常用的损失函数！

● 策略优化之道： 学习常用的学习率调整策略（如 Step Decay、Cosine Annealing）和数据增强技术，提升模型训练的效果和泛化能力！

● 后处理炼金术： 掌握 Confidence Thresholding 和 NMS 阈值调整等后处理技术，并能熟练运用 OpenCV 对检测结果进行优化和可视化！

你将会做到：

● 数据预处理专家： 能够使用 OpenCV 对图像进行各种预处理操作，提高 YOLO 模型的输入质量。

● 算法理解大师： 能够手写实现目标检测中的关键函数和组件，深入理解算法的底层原理。

● 模型调优能手： 能够根据实际情况调整模型参数和后处理策略，优化目标检测效果。

● 问题解决达人： 能够独立解决目标检测过程中遇到的各种问题，如数据处理、模型训练、结果优化等。

行动指南 (一步一步来)：

1.  图像处理基础回顾：

a.  使用 OpenCV 读取一张图像：cv2.imread("image.jpg")，并将其转换为灰度图像：cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)。

b.  使用 OpenCV 对图像进行高斯滤波，去除噪声：cv2.GaussianBlur(image, (5, 5), 0)。

c.  使用 OpenCV 对图像进行边缘检测，提取图像的轮廓：cv2.Canny(image, 100, 200)。

2.  目标检测数据集处理：

a.  加载 COCO 或 Pascal VOC 数据集，并解析标注文件。

b.  对数据集进行随机裁剪、翻转、颜色抖动等数据增强操作。(可参考 albumentations 库)

3.  常用函数复现：

a.  实现计算两个 bounding box 的 IoU 的函数。(提示：计算两个矩形相交的面积)

b.  实现 NMS 算法。(提示：按照 confidence score 排序，迭代删除重叠度高的 bounding box)

c.  (如果涉及 Anchor-based 的模型) 实现生成 Anchor Box 的函数。(提示：根据预定义的宽高比例和尺度生成)

4.  神经网络搭建：函数级别组件实现：

a.  手写实现一个简单的卷积层的前向传播函数。(提示：使用循环实现滑动窗口计算)

b.  手写实现 ReLU 激活函数：relu(x) = max(0, x)。

c.  手写实现 MaxPool 池化层的前向传播函数。(提示：使用循环实现滑动窗口取最大值)

5.  损失函数设计：

a.  手写实现 Cross Entropy Loss 函数。(提示：结合 softmax 函数)

b.  手写实现 Smooth L1 Loss 函数。(提示：Huber Loss 的一种变体)

6.  训练策略优化：

a.  (概念理解) 学习常用的学习率调整策略，如 Step Decay、Cosine Annealing。

b.  (概念理解) 应用常见的数据增强技术，例如随机翻转，随机裁剪等。

7.  后处理技术：

a.  (概念理解) 理解 confidence threshold 的作用，学会如何根据实际情况调整 confidence threshold。(提示：过滤掉 confidence score 低于阈值的 bounding box)

b.  (概念理解) 理解 NMS 阈值的作用，学会如何根据实际情况调整 NMS 阈值。(提示：调整 NMS 算法的重叠度容忍度)

c.  使用 OpenCV 绘制边界框、显示类别标签等。

核心命令 (一定要熟练！) :

● cv2.imread(): 读取图像

● cv2.cvtColor(): 颜色空间转换

● cv2.GaussianBlur(): 高斯滤波

● cv2.Canny(): 边缘检测

● cv2.rectangle(): 绘制矩形框

● cv2.putText(): 添加文本标签

关键参数考核：

● 理解不同颜色空间（如 RGB、HSV）的特点和应用场景。

● 掌握不同滤波方法的原理和效果，例如高斯滤波、中值滤波。

● 理解 IoU 和 NMS 在目标检测中的作用，并能手动计算。

● 掌握卷积操作的原理和计算过程，理解卷积核的作用。

● 理解不同激活函数的特点和应用场景，例如 ReLU、Sigmoid、Tanh。

● 理解分类损失和回归损失在目标检测中的作用，例如 Cross Entropy Loss、Smooth L1 Loss。

代码风格规范 (同 PEP 8)：

遵循 PEP 8 规范，可以使你的代码更加易读、易懂、易维护。

1.  缩进： 使用 4 个空格进行缩进。

2.  行长： 每行代码不超过 79 个字符。

3.  空行： 函数和类定义之间空两行，函数内部空一行。

4.  命名： 变量名、函数名使用小写字母，单词之间用下划线分隔；类名使用驼峰命名法。

5.  注释： 使用清晰、简洁的注释，解释代码的功能和作用。

学习提示：

● 勤于实践，熟能生巧！ 通过大量的手写代码，加深对目标检测算法的理解。

● 善用资源，事半功倍！ 参考优秀的开源项目，学习代码思路和实现技巧。

● 遇到难题，迎难而上！ 积极查阅资料，寻求帮助，解决问题。

● 保持耐心，循序渐进！ 目标检测是一个复杂的领域，需要不断学习和积累。

● 注重原理，举一反三！ 理解算法的底层原理，才能灵活应用和创新。





