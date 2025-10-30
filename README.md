Cloning paper :
SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH
50X FEWER PARAMETERS AND <0.5MB MODEL SIZE

By :
Forrest N. Iandola1, Song Han2, Matthew W. Moskewicz1, Khalid Ashraf1, William J. Dally2, Kurt Keutzer1

The SqueezeNet architecture is available for download here:
https://github.com/DeepScale/SqueezeNet

## 🧩 Paper Summary: _SqueezeNet — AlexNet-Level Accuracy with 50× Fewer Parameters_

**Reference:**  
_Forrest N. Iandola et al., “SqueezeNet: AlexNet-Level Accuracy with 50× Fewer Parameters and <0.5MB Model Size,” ICLR 2017._

**Overview:**  
SqueezeNet is a lightweight convolutional neural network that achieves **AlexNet-level accuracy** on ImageNet with **50× fewer parameters** and a **model size under 0.5 MB** when combined with compression.

It introduces the **Fire module**, built from:

- **Squeeze layer** — uses 1×1 filters to reduce input channels.
- **Expand layer** — uses a mix of 1×1 and 3×3 filters to regain spatial richness.

**Key Design Strategies:**

1. Replace most 3×3 filters with 1×1 filters (reduces parameters 9×).
2. Decrease the number of input channels to 3×3 filters using squeeze layers.
3. Downsample later in the network to preserve large activation maps for better accuracy.
4. Add **bypass (residual) connections** to enhance information flow and improve accuracy without increasing model size.

**Results:**

- Achieves **AlexNet accuracy (≈57.5% Top-1, 80.3% Top-5)** with only **4.8 MB**.
- When combined with **Deep Compression**, model size shrinks to **0.47 MB (510× smaller)**.
- **Simple residual connections** improved Top-1 accuracy by **+2.9%** with no increase in parameters.

---

## 🧠 Insights Gained from Implementation

Through running and experimenting with SqueezeNet, I gained deeper understanding of **architecture–efficiency trade-offs** in CNN design:

- **Squeeze Ratio (SR):** Controls how aggressively features are reduced before expansion.  
  A low SR gives smaller models, while moderate SR improves accuracy up to a saturation point.
- **Filter Ratio (1×1 vs 3×3):** More 1×1 filters save parameters, but a balance is needed to retain spatial detail.
- **Late Downsampling:** Preserving spatial resolution in early layers helps maintain accuracy with fewer filters.
- **Bypass Connections:** Even in lightweight models, simple residual paths stabilize gradients and improve learning.

These insights highlight that **careful architectural structuring** can achieve both compactness and performance — without relying solely on post-training compression.

---

## 🏗️ Structural Understanding Gained in Designing CNN Architectures

Implementing and analyzing SqueezeNet helped develop a clearer **framework for designing efficient neural networks**:

| Design Aspect             | Guiding Principle                                      | Example in SqueezeNet                             |
| ------------------------- | ------------------------------------------------------ | ------------------------------------------------- |
| **Microarchitecture**     | Optimize inside modules (filter sizes, channel counts) | Fire module with 1×1 + 3×3 filters                |
| **Macroarchitecture**     | Organize modules and residual connections effectively  | Simple bypass paths improve gradient flow         |
| **Parameter Efficiency**  | Minimize redundant filters and channels                | 1×1 filters + squeeze layers                      |
| **Computation Placement** | Delay pooling and downsampling                         | Maintain larger activation maps longer            |
| **Scalability**           | Use tunable metaparameters (SR, pct3×3, etc.)          | Easy to scale up or shrink model                  |
| **Hardware Awareness**    | Design for low memory and bandwidth                    | Fully CNN (no FC layers), 0.5 MB deployable model |

---

### 💬 Final Takeaway

> _SqueezeNet taught me that efficient deep learning isn’t just about pruning or quantization — it starts with disciplined architectural design. Compact, well-structured networks can achieve competitive accuracy while staying lightweight, fast, and hardware-friendly._

### Experiment for SqueezeNet

Num classes: 10, Num epochs: 20, LR: 0.001
Model architecture:
SqueezeNet(
(setm): Sequential(
(0): Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2))
(1): ReLU(inplace=True)
)
(fire_modules): Sequential(
(maxpool_1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
(fire_2): FireModule(
(squeeze): Conv2d(96, 16, kernel_size=(1, 1), stride=(1, 1))
(squeeze_activation): ReLU(inplace=True)
(expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
(expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(expand_activation): ReLU(inplace=True)
)
(fire_3): FireModule(
(squeeze): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
(squeeze_activation): ReLU(inplace=True)
(expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
(expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(expand_activation): ReLU(inplace=True)
)
(fire_4): FireModule(
(squeeze): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
(squeeze_activation): ReLU(inplace=True)
(expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
(expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(expand_activation): ReLU(inplace=True)
)
(maxpool_4): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
(fire_5): FireModule(
(squeeze): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
(squeeze_activation): ReLU(inplace=True)
(expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
(expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(expand_activation): ReLU(inplace=True)
)
(fire_6): FireModule(
(squeeze): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
(squeeze_activation): ReLU(inplace=True)
(expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
(expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(expand_activation): ReLU(inplace=True)
)
(fire_7): FireModule(
(squeeze): Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))
(squeeze_activation): ReLU(inplace=True)
(expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
(expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(expand_activation): ReLU(inplace=True)
)
(fire_8): FireModule(
(squeeze): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1))
(squeeze_activation): ReLU(inplace=True)
(expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
(expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(expand_activation): ReLU(inplace=True)
)
(maxpool_8): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
(fire_9): FireModule(
(squeeze): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
(squeeze_activation): ReLU(inplace=True)
(expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
(expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(expand_activation): ReLU(inplace=True)
)
)
(final_conv): Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
(classifier): Sequential(
(0): Dropout(p=0.5, inplace=False)
(1): Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
(2): ReLU(inplace=True)
(3): AdaptiveAvgPool2d(output_size=(1, 1))
)
)

==========================================================================================
Layer (type:depth-idx) Output Shape Param #
==========================================================================================
SqueezeNet [1, 10] --
├─Sequential: 1-1 [1, 96, 109, 109] --
│ └─Conv2d: 2-1 [1, 96, 109, 109] 14,208
│ └─ReLU: 2-2 [1, 96, 109, 109] --
├─Sequential: 1-2 [1, 512, 13, 13] --
│ └─MaxPool2d: 2-3 [1, 96, 54, 54] --
│ └─FireModule: 2-4 [1, 128, 54, 54] --
│ │ └─Conv2d: 3-1 [1, 16, 54, 54] 1,552
│ │ └─ReLU: 3-2 [1, 16, 54, 54] --
│ │ └─Conv2d: 3-3 [1, 64, 54, 54] 1,088
│ │ └─Conv2d: 3-4 [1, 64, 54, 54] 9,280
│ │ └─ReLU: 3-5 [1, 128, 54, 54] --
│ └─FireModule: 2-5 [1, 128, 54, 54] --
│ │ └─Conv2d: 3-6 [1, 16, 54, 54] 2,064
│ │ └─ReLU: 3-7 [1, 16, 54, 54] --
│ │ └─Conv2d: 3-8 [1, 64, 54, 54] 1,088
│ │ └─Conv2d: 3-9 [1, 64, 54, 54] 9,280
│ │ └─ReLU: 3-10 [1, 128, 54, 54] --
│ └─FireModule: 2-6 [1, 256, 54, 54] --
│ │ └─Conv2d: 3-11 [1, 32, 54, 54] 4,128
│ │ └─ReLU: 3-12 [1, 32, 54, 54] --
│ │ └─Conv2d: 3-13 [1, 128, 54, 54] 4,224
│ │ └─Conv2d: 3-14 [1, 128, 54, 54] 36,992
│ │ └─ReLU: 3-15 [1, 256, 54, 54] --
│ └─MaxPool2d: 2-7 [1, 256, 27, 27] --
│ └─FireModule: 2-8 [1, 256, 27, 27] --
│ │ └─Conv2d: 3-16 [1, 32, 27, 27] 8,224
│ │ └─ReLU: 3-17 [1, 32, 27, 27] --
│ │ └─Conv2d: 3-18 [1, 128, 27, 27] 4,224
│ │ └─Conv2d: 3-19 [1, 128, 27, 27] 36,992
│ │ └─ReLU: 3-20 [1, 256, 27, 27] --
│ └─FireModule: 2-9 [1, 384, 27, 27] --
│ │ └─Conv2d: 3-21 [1, 48, 27, 27] 12,336
│ │ └─ReLU: 3-22 [1, 48, 27, 27] --
│ │ └─Conv2d: 3-23 [1, 192, 27, 27] 9,408
│ │ └─Conv2d: 3-24 [1, 192, 27, 27] 83,136
│ │ └─ReLU: 3-25 [1, 384, 27, 27] --
│ └─FireModule: 2-10 [1, 384, 27, 27] --
│ │ └─Conv2d: 3-26 [1, 48, 27, 27] 18,480
│ │ └─ReLU: 3-27 [1, 48, 27, 27] --
│ │ └─Conv2d: 3-28 [1, 192, 27, 27] 9,408
│ │ └─Conv2d: 3-29 [1, 192, 27, 27] 83,136
│ │ └─ReLU: 3-30 [1, 384, 27, 27] --
│ └─FireModule: 2-11 [1, 512, 27, 27] --
│ │ └─Conv2d: 3-31 [1, 64, 27, 27] 24,640
│ │ └─ReLU: 3-32 [1, 64, 27, 27] --
│ │ └─Conv2d: 3-33 [1, 256, 27, 27] 16,640
│ │ └─Conv2d: 3-34 [1, 256, 27, 27] 147,712
│ │ └─ReLU: 3-35 [1, 512, 27, 27] --
│ └─MaxPool2d: 2-12 [1, 512, 13, 13] --
│ └─FireModule: 2-13 [1, 512, 13, 13] --
│ │ └─Conv2d: 3-36 [1, 64, 13, 13] 32,832
│ │ └─ReLU: 3-37 [1, 64, 13, 13] --
│ │ └─Conv2d: 3-38 [1, 256, 13, 13] 16,640
│ │ └─Conv2d: 3-39 [1, 256, 13, 13] 147,712
│ │ └─ReLU: 3-40 [1, 512, 13, 13] --
├─Sequential: 1-3 [1, 10, 1, 1] --
│ └─Dropout: 2-14 [1, 512, 13, 13] --
│ └─Conv2d: 2-15 [1, 10, 13, 13] 5,130
│ └─ReLU: 2-16 [1, 10, 13, 13] --
│ └─AdaptiveAvgPool2d: 2-17 [1, 10, 1, 1] --
==========================================================================================
Total params: 740,554
Trainable params: 740,554
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 737.44
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 33.43
Params size (MB): 2.96
Estimated Total Size (MB): 37.00
==========================================================================================
