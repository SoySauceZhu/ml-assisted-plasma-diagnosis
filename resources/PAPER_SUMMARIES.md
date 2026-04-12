# Research Paper Summaries

> All PDFs are from `/resources/` directory. Papers marked with **🔴 RED** were highlighted/red-tagged in macOS Finder.

---

## 🔴 RED — Machine Learning-Enabled Prediction of Optical Emission Spectra in Scalable Non-Thermal Ar Atmospheric Pressure Plasma Multi Jet

**JPhysD-140473_Proof_hi-2(1).pdf** — Journal of Physics D: Applied Physics, 2025

### 文章做了什么

设计并制作了一个 Ar-multi-Non-Thermal Atmospheric Pressure Plasma Jet (Ar-multi-NTAPPJ) 系统，探索其在大容量处理中的可扩展性。通过改变气体流速、电压、占空比和空气混合比，进行光学发射光谱（OES）测量，分析了 Ar I、OH、N₂ 第二正系（ SPS）和原子氧的发射强度分布。使用机器学习（ML）方法对光谱数据进行降维和监督学习，预测不同操作条件下的发射强度，并进一步使用差分进化优化策略寻找最大化特定光谱特征（如 OH(A–X) 发射）的最佳参数组合。

### 使用了什么模型

- **Principal Component Analysis (PCA)** — 降维
- **Random Forest (RF)** — 监督回归
- **Deep Neural Network (DNN)** — 监督回归
- **Differential Evolution (DE)** — 优化策略

### 模型的输入和输出

**输入：** 气体流速、电压、占空比、空气混合比等操作参数（Operating Parameters, OPs）
**输出：** 各发射谱线强度（Emission Intensities），特别是 OH(A–X) 发射强度；以及通过 DE 优化得到的最优参数组合

### 该研究解决了什么问题，有什么好处

解决了传统 OES 分析耗时、难以实时化的问题。通过 ML 快速预测不同操作条件下的光谱发射强度，结合优化算法实现实时参数优化。加速了等离子体诊断过程，为大面积等离子体处理提供了一种高效的诊断与优化方案。

### 一句话总结

本文通过 PCA 降维结合 RF/DNN 模型，实现了对可扩展 Ar 多射流等离子体系统光谱的实时预测与参数优化。

### BibTeX Reference

```bibtex
@article{Srikar2025MLOES,
  title={Machine Learning-Enabled Prediction of Optical Emission Spectra in Scalable Non-Thermal Ar Atmospheric Pressure Plasma Multi Jet},
  author={Srikar, P.S.N.S.R. and Suresh, Indhu and Gangwar, R.K.},
  journal={Journal of Physics D: Applied Physics},
  volume={58},
  pages={415204},
  year={2025},
  doi={10.1088/1361-6463/ae0b81}
}
```

---

## Machine Learning Prediction of Pyrolytic Products of Lignocellulosic Biomass Based on Physicochemical Characteristics and Pyrolysis Conditions

**Machine learning prediction of pyrolytic products of lignocellulosic biomass based on physicochemical characteristics and pyrolysis conditions.pdf** — Bioresource Technology 367 (2023) 128182

### 文章做了什么

基于生物质的物理化学特性和热解条件，使用多种机器学习算法预测生物质热解产物（生物油、生物炭、热解气）的产率。比较分析了 Random Forest (RF)、Gradient Boosting Decision Tree (GBDT)、XGBoost 和 Adaboost 四种集成算法，识别了各特征对不同目标变量的重要性，并进行了偏相关分析以深入理解热解过程。

### 使用了什么模型

- **Random Forest (RF)** — 最佳模型
- **Gradient Boosting Decision Tree (GBDT)**
- **XGBoost (eXtreme Gradient Boosting)**
- **Adaboost (Adaptive Boosting)**

### 模型的输入和输出

**输入：** 生物质物理化学特性（水分含量、碳含量、氢含量、挥发分等）+ 热解条件（最终加热温度、升温速率等）
**输出：** 三种热解产物产率 — 生物油（Bio-oil）产率、生物炭（Biochar）产率、热解气（Pyrolytic gas）产率

### 该研究解决了什么问题，有什么好处

解决了传统经验模型难以准确预测复杂热解产物分布的问题。RF 算法表现出最佳的预测能力，能同时准确预测三种产物产率。研究发现水分含量、碳含量和最终加热温度是最重要的预测因素，生物质特性比热解条件更重要。这为工程应用中的热解过程优化提供了数据驱动的新见解。

### 一句话总结

本文通过比较多种集成学习算法，确立了 Random Forest 为预测生物质热解产物产率的最优模型，并揭示了水分含量、碳含量和最终加热温度的关键作用。

### BibTeX Reference

```bibtex
@article{Dong2023MLBiomass,
  title={Machine learning prediction of pyrolytic products of lignocellulosic biomass based on physicochemical characteristics and pyrolysis conditions},
  author={Dong, Zixun and Bai, Xiaopeng and Xu, Daochun and Li, Wenbin},
  journal={Bioresource Technology},
  volume={367},
  pages={128182},
  year={2023},
  doi={10.1016/j.biortech.2022.128182}
}
```

---

## Accelerated Real-Time Plasma Diagnostics: Integrating Argon Collisional-Radiative Model with Machine Learning Methods

**1-s2.0-S0584854724000533-main.pdf** — Spectrochimica Acta Part B: Atomic Spectroscopy 215 (2024) 106909

### 文章做了什么

将两种机器学习技术（Random Forest 和 Deep Neural Network）与光学发射光谱（OES）及碰撞辐射（CR）模型结合，实现了对非热大气压氩等离子体射流电子温度（Te）的实时预测。通过随机搜索策略优化超参数，使用大规模数据集训练和测试模型。

### 使用了什么模型

- **Random Forest (RF)** — 回归
- **Deep Neural Network (DNN)** — 回归
- **Collisional-Radiative (CR) Model** — 物理模型

### 模型的输入和输出

**输入：** OES 光谱数据（来自氩原子发射谱线）
**输出：** 电子温度 Te 预测值；DNN R²=0.9964，RF R²=0.9869

### 该研究解决了什么问题，有什么好处

解决了传统等离子体诊断方法（如 Boltzmann 斜率法、Saha-Boltzmann 法）耗时且需要专业知识的缺点。ML 方法实现了电子温度的快速（实时）预测，精度高且非侵入式，为等离子体过程实时监控提供了新途径。

### 一句话总结

本文通过 RF 和 DNN 机器学习模型与碰撞辐射模型结合，实现了对大气压氩等离子体射流电子温度的高精度实时预测（R²>0.98）。

### BibTeX Reference

```bibtex
@article{Srikar2024Accelerated,
  title={Accelerated real-time plasma diagnostics: Integrating argon collisional-radiative model with machine learning methods},
  author={Srikar, P.S.N.S.R. and Suresh, Indhu and Gangwar, R.K.},
  journal={Spectrochimica Acta Part B: Atomic Spectroscopy},
  volume={215},
  pages={106909},
  year={2024},
  doi={10.1016/j.sab.2024.106909}
}
```

---

## 🔴 RED — Machine Learning Assisted Optical Emission Spectroscopy to Determine Electron Density and Electron Temperature in a Cascaded Arc Plasma

**1-s2.0-S2352179125001346-main.pdf** — Nuclear Materials and Energy 45 (2025) 101992

### 文章做了什么

针对级联弧等离子体开发了基于支持向量机（SVM）的机器学习模型，结合激光 Thomson 散射（LTS）的高精度和 OES 的空间灵活性，预测电子密度（ne）和电子温度（Te）。使用四对"双峰"光谱线，无需复杂校准即可进行等离子体诊断。

### 使用了什么模型

- **Support Vector Machine (SVM)** — 回归，带网格搜索优化（Grid Search）

### 模型的输入和输出

**Case 1 输入：** 放电条件 + 线强度比（LIRs）
**Case 2 输入：** 仅线强度比（LIRs）
**输出：** 电子密度 ne（Case1 R²≈0.97，Case2 R²≈0.90）和电子温度 Te（Case1 R²≈0.92，Case2 R²≈0.80）

### 该研究解决了什么问题，有什么好处

解决了 LTS 设备复杂、空间灵活性差，以及简单 OES 精度受限的问题。ML 模型融合 LTS 和 OES 的优势，实现了免校准、灵活选择光谱线的等离子体诊断，可推广至其他线性等离子体装置。

### 一句话总结

本文通过 SVM 模型融合激光 Thomson 散射和光学发射光谱的优势，实现了对级联弧等离子体电子密度和温度的高精度免校准预测。

### BibTeX Reference

```bibtex
@article{Wang2025MLOESCascaded,
  title={Machine learning assisted optical emission spectroscopy to determine electron density and electron temperature in a cascaded arc plasma},
  author={Wang, Yong and Zhou, Lina and Li, Cong and Feng, Chunlei and Ding, Hongbin},
  journal={Nuclear Materials and Energy},
  volume={45},
  pages={101992},
  year={2025},
  doi={10.1016/j.nme.2025.101992}
}
```

---

## Prediction and Evaluation of Plasma Arc Reforming of Naphthalene Using a Hybrid Machine Learning Model

**2021 JHM Wang Tar ML.pdf** — Journal of Hazardous Materials 404 (2021) 123965

### 文章做了什么

针对生物质气化中焦油（以萘为模型化合物）的等离子体重整过程，开发了混合机器学习模型用于预测和优化滑弧等离子体焦油重整性能。结合 ANN、SVR 和 DT 三种算法，用遗传算法（GA）优化超参数，分析了蒸汽碳比（S/C）和放电功率等参数对焦油转化、碳平衡和能量效率的影响。

### 使用了什么模型

- **Artificial Neural Network (ANN)** — 混合组件
- **Support Vector Regression (SVR)** — 混合组件
- **Decision Tree (DT)** — 混合组件
- **Genetic Algorithm (GA)** — 超参数优化

### 模型的输入和输出

**输入：** 蒸汽碳比（S/C）、放电功率、气体停留时间等操作参数
**输出：** 焦油转化率（67.2%）、碳平衡（81.7%）、能量效率（7.8 g/kWh）；S/C 比率是最关键参数（相对重要性 38%），放电功率对能量效率影响最大（58%）

### 该研究解决了什么问题，有什么好处

解决了等离子体焦油重整过程多尺度、复杂、非线性难以用传统模型预测的问题。混合 ML 模型在小样本实验数据下实现了高预测精度，为等离子体重整过程的实时优化提供了工具。

### 一句话总结

本文通过 GA 优化的 ANN+SVR+DT 混合模型，实现了生物质气化滑弧等离子体焦油重整过程的多目标优化，S/C 比为主要控制参数。

### BibTeX Reference

```bibtex
@article{Wang2021PlasmaTarML,
  title={Prediction and evaluation of plasma arc reforming of naphthalene using a hybrid machine learning model},
  author={Wang, Yaolin and Liao, Zinan and Mathieu, St{\'e}phanie and Bin, Feng and Tu, Xin},
  journal={Journal of Hazardous Materials},
  volume={404},
  pages={123965},
  year={2021},
  doi={10.1016/j.jhazmat.2020.123965}
}
```

---

## Machine Learning-Driven Optimization of Plasma-Catalytic Dry Reforming of Methane

**2024 Journal of Energy Chemistry.pdf** — Journal of Energy Chemistry 96 (2024) 153–163

### 文章做了什么

针对 Ni/Al₂O₃ 催化剂上介质阻挡放电（DBD）等离子体催化干式重整甲烷（DRM）过程，开发了混合机器学习模型。集成回归树、SVR 和 ANN 三种算法，用遗传算法优化超参数，在有限实验数据下实现过程预测和优化。

### 使用了什么模型

- **Regression Trees** — 混合组件
- **Support Vector Regression (SVR)** — 混合组件
- **Artificial Neural Network (ANN)** — 混合组件
- **Genetic Algorithm (GA)** — 超参数优化

### 模型的输入和输出

**输入：** 放电功率、CO₂/CH₄ 摩尔比、Ni 负载量、总流量等操作参数
**输出：** 各目标变量（转化率、能量产率等）；最优条件：放电功率 20W、CO₂/CH₄=1.5、Ni 负载 7.8wt%、总流量 51mL/min；总流量对各输出影响最大（>35%），Ni 负载影响最小

### 该研究解决了什么问题，有什么好处

解决了 DRM 过程传统热催化需要 >800°C 高温、能耗高的问题。非热等离子体结合催化降低反应温度，ML 模型在有限数据下实现过程优化，推动了 DRM 技术的商业化可行性。

### 一句话总结

本文通过 GA 优化的混合 ML 模型（回归树+SVR+ANN），在有限实验数据下实现了等离子体催化 DRM 过程的预测与优化，确定了最优操作条件。

### BibTeX Reference

```bibtex
@article{Cai2024MLDRM,
  title={Machine learning-driven optimization of plasma-catalytic dry reforming of methane},
  author={Cai, Yuxiang and Mei, Danhua and Chen, Yanzhen and Bogaerts, Annemie and Tu, Xin},
  journal={Journal of Energy Chemistry},
  volume={96},
  pages={153--163},
  year={2024},
  doi={10.1016/j.jechem.2024.01.001}
}
```

---

## 🔴 RED — Machine Learning Assisted Optical Diagnostics on a Cylindrical Surface Dielectric Barrier Discharge

**2404.06817v2.pdf** — (arXiv:2404.06817, 2025)

### 文章做了什么

将机器学习算法与标准光学诊断（时间积分发射光谱和成像）结合，精确预测圆筒表面介质阻挡放电（SDBD）的操作条件并评估发射均匀性。利用无监督（PCA）和监督（MLP）算法，基于光谱/成像数据对施加的电压波形和幅度进行分类和预测，并分析四个不同区域的发射均匀性。

### 使用了什么模型

- **Principal Component Analysis (PCA)** — 无监督降维/分类
- **Multilayer Perceptron (MLP)** — 监督回归/预测

### 模型的输入和输出

**输入：** 时间积分发射光谱和成像数据（SDBD 四个不同区域）
**输出：** (1) 电压波形（AC/pulsed）和幅度的分类/预测；(2) 放电发射均匀性评估

### 该研究解决了什么问题，有什么好处

解决了大气压等离子体瞬态特性带来的诊断挑战。PCA 有效分类电压波形，MLP 实现准确的幅度预测，为等离子体过程的实时控制、监控和优化开辟了新途径，特别是对流控制等应用。

### 一句话总结

本文通过 PCA 降维结合 MLP 神经网络，实现了圆筒表面 DBD 等离子体电压波形和幅度的分类预测，以及发射均匀性评估。

### BibTeX Reference

```bibtex
@article{Stefas2025MLSDBD,
  title={Machine learning assisted optical diagnostics on a cylindrical surface dielectric barrier discharge},
  author={Stefas, D. and Giotis, K. and Invernizzi, L. and H{\"o}ft, H. and Hassouni, K. and Prasanna, S. and Svarnas, P. and Lombardi, G. and Gazeli, K.},
  journal={arXiv:2404.06817},
  year={2025}
}
```

---

## 🔴 RED — Machine Learning for Real-Time Diagnostics of Cold Atmospheric Plasma Sources

**IEEE_Plasma_2019_1.pdf** — IEEE Transactions on Radiation and Plasma Medical Sciences, Vol. 3, No. 5, September 2019

### 文章做了什么

展示了多种机器学习方法利用信息丰富的光学发射光谱（OES）和电声发射信号，对冷大气压等离子体（CAP）源进行实时诊断。演示了旋转温度、振动温度和基底特性等操作相关参数的实时估计方法。

### 使用了什么模型

- **Gaussian Process (GP)**
- **K-means Clustering**
- **Linear Regression**
- **Principle Component Analysis (PCA)**
- 以及其他 ML 方法

### 模型的输入和输出

**输入：** 光学发射光谱（OES）数据 + 电声发射（Electro-acoustic emission）数据
**输出：** 旋转温度、振动温度、基底特性等操作相关参数的实时估计

### 该研究解决了什么问题，有什么好处

解决了 CAP 源传统诊断方法（如 LIF、质谱、拉曼散射）需要昂贵设备和复杂分析、无法实时化的问题。ML 方法利用易于获取的光谱和电声信号，实现实时等离子体诊断，推动了等离子体医学的实际应用。

### 一句话总结

本文证明了基于 ML 的数据分析可用于冷大气压等离子体源的实时诊断，仅利用 OES 和电声信号即可实时估计旋转/振动温度等关键参数。

### BibTeX Reference

```bibtex
@article{Gidon2019MLCAP,
  title={Machine Learning for Real-Time Diagnostics of Cold Atmospheric Plasma Sources},
  author={Gidon, Dogan and Pei, Xuekai and Bonzanini, Angelo D. and Graves, David B. and Mesbah, Ali},
  journal={IEEE Transactions on Radiation and Plasma Medical Sciences},
  volume={3},
  number={5},
  pages={597--607},
  year={2019},
  doi={10.1109/TRPMS.2019.2919582}
}
```

---

## Machine Learning Prediction of Electron Density and Temperature from Optical Emission Spectroscopy in Nitrogen Plasma

**coatings-11-01221-v2.pdf** — Coatings 11 (2021) 1221

### 文章做了什么

提出了一种非侵入式方法，通过光学发射光谱（OES）结合多变量数据分析，监测射频（RF）等离子体渗氮装置中的电子温度（Te）和电子密度（ne）。建立了基于同时 OES 和其他诊断方法的经验相关性，开发了基于 ML 的虚拟计量模型，使用原位 OES 传感器实现实时 Te 和 ne 监测。

### 使用了什么模型

- **Machine Learning (ML)** — 虚拟计量模型（具体算法未在摘要中明确，疑似 ANN 或类似多变量回归模型）

### 模型的输入和输出

**输入：** OES 光谱数据（原位 OES 传感器）
**输出：** 电子密度 ne（预测精度 97%）、电子温度 Te（预测精度 90%）

### 该研究解决了什么问题，有什么好处

解决了工业等离子体加工中直接实时监测等离子体参数困难的问题。虚拟计量方法无需干扰等离子体或干涉工艺过程即可实现原位实时分析，对等离子体加工过程的控制和质量保证具有重要意义。

### 一句话总结

本文通过 OES 结合 ML 虚拟计量模型，实现了射频等离子体渗氮过程中电子密度（97% 精度）和电子温度（90% 精度）的实时非侵入式监测。

### BibTeX Reference

```bibtex
@article{Park2021MLOESNitrogen,
  title={Machine Learning Prediction of Electron Density and Temperature from Optical Emission Spectroscopy in Nitrogen Plasma},
  author={Park, Jun-Hyoung and Cho, Ji-Ho and Yoon, Jung-Sik and Song, Jung-Ho},
  journal={Coatings},
  volume={11},
  pages={1221},
  year={2021},
  doi={10.3390/coatings11101221}
}
```

---

## Development and Testing of an Efficient Data Acquisition Platform for Machine Learning of Optical Emission Spectroscopy of Plasmas in Aqueous Solution

**Wang_2019_Plasma_Sources_Sci._Technol._28_105013.pdf** — Plasma Sources Science and Technology 28 (2019) 105013

### 文章做了什么

开发了一种高效的光谱数据采集平台，用于在溶液中等离子体的光学发射光谱（OES）采集，并测试了多种机器学习算法进行等离子体光谱的多变量分析。在 15 分钟内可采集最多 10k 条光谱（含不同 pH 或电导率条件），共采集 40k 条光谱，测试了 PCA 和 ANN 对溶液电导率的预测能力。

### 使用了什么模型

- **Principal Component Analysis (PCA)** — 多变量分析
- **Artificial Neural Network (ANN)** — 深度 ANN（Deep ANN）
- **Dropout + Early Stopping** — 正则化技术

### 模型的输入和输出

**输入：** 溶液中等离子体 OES 光谱（pH 2.2–5.2，不同电导率条件）
**输出：** 溶液电导率；PCA 在 PC1-PC2 得分图中重叠无法区分，深度 ANN 相比单发射线方法 MSE 降低三个数量级

### 该研究解决了什么问题，有什么好处

解决了溶液中等离子体光谱采集效率低、数据量不足以训练 ML 模型的问题。高效采集平台提供了充足的数据集，深度 ANN 显著优于 PCA 和单发射线方法，证明了深度学习在等离子体光谱分析中的优势。

### 一句话总结

本文开发了高效 OES 数据采集平台，证明了深度 ANN 可将溶液电导率预测的 MSE 降低三个数量级，为等离子体光谱的 ML 分析提供了可靠数据基础。

### BibTeX Reference

```bibtex
@article{Wang2019MLOESAqueous,
  title={Development and testing of an efficient data acquisition platform for machine learning of optical emission spectroscopy of plasmas in aqueous solution},
  author={Wang, Ching-Yu and Hsu, Cheng-Che},
  journal={Plasma Sources Science and Technology},
  volume={28},
  pages={105013},
  year={2019},
  doi={10.1088/1361-6595/ab45e5}
}
```

---

## Machine Learning Assisted Classification of Aluminum Nitride Thin Film Stress via In-Situ Optical Emission Spectroscopy Data

**materials-14-04445-v2.pdf** — Materials 14 (2021) 4445

### 文章做了什么

在脉冲直流磁控溅射沉积氮化铝（AlN）薄膜过程中，通过 OES 采集原位数据，建立 OES 数据与薄膜残余应力（张应力/压应力）之间的相关性。使用 PCA 降维后结合人工神经网络（ANN）进行应力类型分类。

### 使用了什么模型

- **Principal Component Analysis (PCA)** — 降维
- **Artificial Neural Network (ANN)** — 分类

### 模型的输入和输出

**输入：** 溅射功率、氮气流速等工艺参数对应的 OES 光谱数据
**输出：** AlN 薄膜残余应力类型（张应力 vs. 压应力）分类 + 晶体状态（通过 XRD + TEM 验证）

### 该研究解决了什么问题，有什么好处

解决了薄膜应力无法在线实时监测的问题。通过 OES 数据的实时分析，可在不中断工艺的情况下判断薄膜应力类型，有助于提高半导体工艺的良率和效率。

### 一句话总结

本文通过 PCA 降维结合 ANN，对脉冲直流磁控溅射 AlN 薄膜的应力类型（张应力/压应力）实现了基于 OES 数据的在线分类。

### BibTeX Reference

```bibtex
@article{Yang2021MLAlN,
  title={Machine Learning Assisted Classification of Aluminum Nitride Thin Film Stress via In-Situ Optical Emission Spectroscopy Data},
  author={Yang, Yu-Pu and Lu, Te-Yun and Lo, Hsiao-Han and Chen, Wei-Lun and Wang, Peter J. and Lai, Walter and Fuh, Yiin-Kuen and Li, Tomi T.},
  journal={Materials},
  volume={14},
  pages={4445},
  year={2021},
  doi={10.3390/ma14164445}
}
```

---

## A Comparison Review of Transfer Learning and Self-Supervised Learning: Definitions, Applications, Advantages and Limitations

**1-s2.0-S0957417423033092-main.pdf** — Expert Systems with Applications 242 (2024) 122807

### 文章做了什么

综述了迁移学习（Transfer Learning）和自监督学习（Self-Supervised Learning）两种主流预训练方法在深度学习中的应用。阐述了定义、优势、局限和应用场景，对比分析了近年来这两种方法在医学图像、视频识别、自然语言处理等领域的最新进展。

### 使用了什么模型

综述类论文（非实验性），涵盖：
- **Transfer Learning** — 预训练模型微调
- **Self-Supervised Learning** — 对比学习、掩码自编码器等
- **CNN / Vision Transformer** 等 backbone

### 模型的输入和输出

综述性质，无具体输入输出，但讨论了：
- Transfer Learning：源域大数据预训练 → 目标域小数据微调
- Self-Supervised Learning：大规模无标签数据预训练（ pretext tasks）→ 下游任务微调

### 该研究解决了什么问题，有什么好处

解决了深度学习标注数据稀缺的问题。综述为研究者和工程师选择合适的预训练方法提供了全面指导，有助于针对特定应用场景（医学图像、 NLP、CV 等）做出最优决策。

### 一句话总结

本文全面综述了迁移学习和自监督学习两大预训练范式的定义、应用、优缺点，为解决深度学习数据稀缺问题提供了方法选择指南。

### BibTeX Reference

```bibtex
@article{Zhao2024TransferSSL,
  title={A comparison review of transfer learning and self-supervised learning: Definitions, applications, advantages and limitations},
  author={Zhao, Zehui and Alzubaidi, Laith and Zhang, Jinglan and Duan, Ye and Gu, Yuantong},
  journal={Expert Systems with Applications},
  volume={242},
  pages={122807},
  year={2024},
  doi={10.1016/j.eswa.2023.122807}
}
```

---

## 🔴 RED — Deep Unsupervised Domain Adaptation: A Review of Recent Advances and Perspectives

**2208.07422v1.pdf** — APSIPA Transactions on Signal and Information Processing (2022)

### 文章做了什么

综述了深度无监督域适应（UDA）的最新进展和未来展望。介绍了 UDA 如何利用有标签源域数据和无标签目标域数据，使模型能够适应目标域。覆盖了自然图像、医学图像、语音识别等领域的 UDA 方法。

### 使用了什么模型

综述类论文，涵盖多种深度 UDA 方法：
- Domain-Adversarial Neural Networks (DANN)
- Cycle-Consistent UDA
- Contrastive Learning for UDA
- Self-training UDA
- 等

### 模型的输入和输出

综述性质，讨论了：
- **输入：** 有标签源域数据 + 无标签目标域数据
- **输出：** 可在目标域执行任务的深度模型

### 该研究解决了什么问题，有什么好处

解决了深度学习模型在目标域性能下降的问题（域偏移）。UDA 使得模型能够从源域泛化到目标域，减少了昂贵的目标域标注需求，对医学图像分析、跨语言 NLP 等场景有重要价值。

### 一句话总结

本文综述了深度无监督域适应（UDA）的最新方法，解决了跨域深度学习模型泛化问题，为减少目标域标注需求提供了系统指导。

### BibTeX Reference

```bibtex
@article{Liu2022UDA,
  title={Deep Unsupervised Domain Adaptation: A Review of Recent Advances and Perspectives},
  author={Liu, Xiaofeng and Yoo, Chaehwa and Xing, Fangxu and Oh, Hyejin and El Fakhri, Georges and Kang, Je-Won and Woo, Jonghye},
  journal={APSIPA Transactions on Signal and Information Processing},
  volume={1},
  pages={1--48},
  year={2022},
  doi={10.1155/2022/8065794}
}
```

---

## 🔴 RED — Optical Diagnostics of Atmospheric Pressure Air Plasmas

**725.pdf** — Plasma Sources Science and Technology 12 (2003) 125–138

### 文章做了什么

综述了大气压空气等离子体的光学诊断技术，包括发射光谱学（OES）和腔体衰减光谱学（CRDS）。评估了在热平衡到热化学非平衡条件下的大气压等离子体中温度和浓度测量的准确性，挑战了大气压空气等离子体处于局部热力学平衡（LTE）的常见假设。

### 使用了什么模型

综述型论文，无具体机器学习模型。使用物理建模方法：
- Boltzmann 斜率法
- Saha-Boltzmann 法
- OES 和 CRDS 实验诊断

### 该研究解决了什么问题，有什么好处

澄清了大气压空气等离子体非 LTE 特性的重要问题。证明了流动等离子体中的速度梯度和放电产生的升高电子温度会导致热力学和化学平衡的重大偏离，对等离子体诊断和建模有重要指导意义。

### 一句话总结

本文综述了大气压空气等离子体的光学诊断技术，证明这类等离子体经常偏离局部热力学平衡，对传统诊断假设提出了挑战。

### BibTeX Reference

```bibtex
@article{Laux2003OESAir,
  title={Optical diagnostics of atmospheric pressure air plasmas},
  author={Laux, C O and Spence, T G and Kruger, C H and Zare, R N},
  journal={Plasma Sources Science and Technology},
  volume={12},
  pages={125--138},
  year={2003},
  doi={10.1088/0963-0252/12/2/301}
}
```

---

## 🔴 RED — Real-Time Monitoring of the Plasma Density Distribution in Low-Pressure Plasmas Using a Flat-Cutoff Array Sensor

**114103_1_online.pdf** — Applied Physics Letters 122 (2023) 114103

### 文章做了什么

开发了一种平切截止阵列传感器（flat-cutoff array sensor），用于低气压等离子体中等离子体密度分布的实时监测。通过微波传输光谱分析，实现了对等离子体密度空间分布的高精度实时测量。

### 使用了什么模型

实验/工程类论文，无具体机器学习模型。使用微波截止探针技术结合阵列传感器设计。

### 该研究解决了什么问题，有什么好处

解决了低气压等离子体密度分布难以实时监测的问题。该传感器具有空间灵活性，可实时监测等离子体密度分布，对等离子体加工过程的在线质量控制具有重要价值。

### 一句话总结

本文通过平切截止阵列传感器结合微波传输光谱，实现了低气压等离子体密度分布的实时空间监测。

### BibTeX Reference

```bibtex
@article{Yeom2023PlasmaDensity,
  title={Real-time monitoring of the plasma density distribution in low-pressure plasmas using a flat-cutoff array sensor},
  author={Yeom, H.J. and Yoon, M.Y. and Chae, G.S. and Kim, J.H. and You, S.J. and Lee, H.C.},
  journal={Applied Physics Letters},
  volume={122},
  pages={114103},
  year={2023},
  doi={10.1063/5.0129790}
}
```

---

## Safety and Efficiency Evaluation of an Innovative Plasma Jet Array in Argon Using Gas Switching Technology

**Balazinski_2025_J._Phys._D__Appl._Phys._58_295202.pdf** — Journal of Physics D: Applied Physics 58 (2025) 295202

### 文章做了什么

评估了一种基于气体切换技术的创新氩气等离子体射流阵列的安全性和效率。研究了气体切换对等离子体射流性能的影响，分析了等离子体射流阵列在不同操作条件下的稳定性和均匀性。

### 使用了什么模型

实验/工程类论文，无具体机器学习模型。

### 该研究解决了什么问题，有什么好处

解决了等离子体射流阵列在大面积应用中安全性不足和效率低下的问题。气体切换技术提供了一种改善等离子体射流阵列性能的方法，拓展了等离子体在材料处理和生物医学领域的应用。

### 一句话总结

本文评估了基于气体切换技术的氩气等离子体射流阵列，证明了其在安全性和效率方面的优势。

### BibTeX Reference

```bibtex
@article{Balazinski2025PlasmaJet,
  title={Safety and efficiency evaluation of an innovative plasma jet array in argon using gas switching technology},
  author={Balazinski, Martina and Wagner, Robert and Horn, Stefan and Timm, Michael and Hahn, Veronika and Turski, Philipp and L{\"u}ttjohann, Paul and Glitsch, Sven and von Woedtke, Thomas and Weltmann, Klaus-Dieter and Gerling, Torsten},
  journal={Journal of Physics D: Applied Physics},
  volume={58},
  pages={295202},
  year={2025},
  doi={10.1088/1361-6463/ade264}
}
```

---

## 🔴 RED — Efficient Synthesis of CO and H₂ via Nanosecond Pulsed CO₂ Bubble Discharge

**Gao_2024_J._Phys._D__Appl._Phys._57_375204.pdf** — Journal of Physics D: Applied Physics 57 (2024) 375204

### 文章做了什么

通过纳秒脉冲 CO₂ 气泡放电高效合成 CO 和 H₂。研究了脉冲参数（频率、脉宽、上升时间等）对放电特性和合成产物的影响，揭示了纳秒脉冲在 CO₂ 利用和 H₂ 生产中的优势。

### 使用了什么模型

实验/工程类论文，无具体机器学习模型。使用光谱诊断（OES）、电学测量等方法。

### 该研究解决了什么问题，有什么好处

解决了 CO₂ 转化和 H₂ 生产中能耗高的问题。纳秒脉冲放电在 CO₂ 活化和分解方面具有高效率，为温室气体利用和清洁氢生产提供了新途径。

### 一句话总结

本文通过纳秒脉冲 CO₂ 气泡放电实现了 CO 和 H₂ 的高效合成，揭示了脉冲参数对产物分布的影响机制。

### BibTeX Reference

```bibtex
@article{Gao2024NSCO2Discharge,
  title={Efficient synthesis of CO and H2 via nanosecond pulsed CO2 bubble discharge},
  author={Gao, Yuting and Zhou, Renwu and Hong, Longfei and Chen, Bohan and Sun, Jing and Zhou, Rusen and Liu, Zhijie},
  journal={Journal of Physics D: Applied Physics},
  volume={57},
  pages={375204},
  year={2024},
  doi={10.1088/1361-6463/ad5569}
}
```

---

## 🔴 RED — Atmospheric-Pressure Pulsed Discharges and Plasmas: Mechanism, Characteristics and Applications

**High Voltage - 2018 - Shao - Atmospheric‐pressure pulsed discharges and plasmas.pdf** — High Voltage (2018)

### 文章做了什么

综述了大气压脉冲放电等离子体及其应用的研究进展。介绍了基于逃逸电子的纳秒脉冲气体放电机制，以及直接驱动脉冲放电、脉冲介质阻挡放电和脉冲等离子体射流三种典型放电的特性。涵盖了脉冲等离子体在表面改性、医学应用等领域的典型应用。

### 使用了什么模型

综述类论文，无具体机器学习模型。涵盖了纳秒脉冲放电理论、逃逸电子测量、等离子体特性表征和多种应用案例。

### 该研究解决了什么问题，有什么好处

系统总结了脉冲放电等离子体领域的研究进展，为该领域的深入研究和应用开发提供了全面的理论基础，推动了脉冲等离子体技术在民用领域的转化。

### 一句话总结

本文综述了大气压脉冲放电等离子体的机制、特性及应用，为纳秒脉冲放电研究和等离子体应用开发提供了全面的参考。

### BibTeX Reference

```bibtex
@article{Shao2018PulsedDischarges,
  title={Atmospheric-pressure pulsed discharges and plasmas: mechanism, characteristics and applications},
  author={Shao, Tao and Wang, Ruixue and Zhang, Cheng and Yan, Ping},
  journal={High Voltage},
  volume={3},
  pages={14--20},
  year={2018},
  doi={10.1049/hve.2016.0014}
}
```

---

## Global Plasma Modeling of a Magnetized High-Frequency Plasma Source in Low-Pressure Nitrogen and Oxygen for Air-Breathing Electric Propulsion Applications

**Mrózek_2021_Plasma_Sources_Sci._Technol._30_125007.pdf** — Plasma Sources Science and Technology 30 (2021) 125007

### 文章做了什么

对低气压氮气和氧气中磁化高频等离子体源进行了全局等离子体建模（Global Plasma Modeling），应用于空气呼吸电推进（Air-Breathing Electric Propulsion, ABEP）。研究了电子密度、电子温度和重粒子密度分布，为 ABEP 装置的设计和优化提供了理论指导。

### 使用了什么模型

物理建模/数值模拟论文（Global Plasma Model），无具体机器学习模型。

### 该研究解决了什么问题，有什么好处

解决了空气呼吸电推进中等离子体源设计和优化缺乏理论指导的问题。建模结果有助于理解和预测不同气体成分和操作条件下的等离子体行为，为空间推进系统的工程化应用提供了支撑。

### 一句话总结

本文通过全局等离子体建模，研究了低气压氮/氧混合气中磁化高频等离子体源的特性，为空气呼吸电推进系统提供了理论优化依据。

### BibTeX Reference

```bibtex
@article{Mrozek2021GlobalPlasma,
  title={Global plasma modeling of a magnetized high-frequency plasma source in low-pressure nitrogen and oxygen for air-breathing electric propulsion applications},
  author={Mroźek, Kry{\v{s}}tof and Dytrych, Tom{\'a}{\v{s}} and Mo{\l}isz, Pavel and D{\'a}niel, Vladim{\'\i}r and Obrusn{\'\i}k, Adam},
  journal={Plasma Sources Science and Technology},
  volume={30},
  pages={125007},
  year={2021},
  doi={10.1088/1361-6595/ac36ac}
}
```

---

## Diagnostics of Atmospheric-Pressure Pulsed-DC Discharge with Metal and Liquid Anodes by Multiple Laser-Aided Methods

**Urabe_2016_Plasma_Sources_Sci._Technol._25_045004.pdf** — Plasma Sources Science and Technology 25 (2016) 045004

### 文章做了什么

使用多种激光辅助方法（激光诱导荧光、Thomson 散射、Millikan 粒子成像等）对大气压脉冲直流放电（金属阳极和液体阳极）进行诊断研究。测量了电子温度、离子温度和等离子体密度等关键参数。

### 使用了什么模型

实验诊断论文，无具体机器学习模型。

### 该研究解决了什么问题，有什么好处

解决了大气压放电诊断困难的问题，特别是金属和液体阳极条件下的诊断。多激光辅助方法提供了全面的等离子体参数信息，有助于理解电极材料对放电特性的影响。

### 一句话总结

本文通过多种激光辅助诊断方法，对大气压脉冲直流放电（金属和液体阳极）等离子体进行了全面参数诊断。

### BibTeX Reference

```bibtex
@article{Urabe2016PulsedDC,
  title={Diagnostics of atmospheric-pressure pulsed-dc discharge with metal and liquid anodes by multiple laser-aided methods},
  author={Urabe, Keiichiro and Shirai, Naoki and Tomita, Kentaro and Akiyama, Tsuyoshi and Murakami, Tomoyuki},
  journal={Plasma Sources Science and Technology},
  volume={25},
  pages={045004},
  year={2016},
  doi={10.1088/0963-0252/25/4/045004}
}
```

---

## Intensity Ratio of Spectral Bands of Nitrogen as a Measure of Electric Field Strength in Plasmas

**Paris_2005_J._Phys._D__Appl._Phys._38_3894.pdf** — Journal of Physics D: Applied Physics 38 (2005) 3894–3899

### 文章做了什么

研究了氮气等离子体中不同光谱带强度比作为等离子体电场强度测量指标的应用。利用氮分子第二正系（SPS，C³Πu→B³Πg）和第一负系（FNS, B²Σu⁺→X²Σg⁺）等光谱带的强度比，建立了一种测量等离子体电场的方法。

### 使用了什么模型

实验/物理建模论文，无具体机器学习模型。

### 该研究解决了什么问题，有什么好处

提供了一种利用光谱强度比测量等离子体电场的方法，无需复杂设备即可估计电场强度，对等离子体诊断和过程控制有实际应用价值。

### 一句话总结

本文利用氮气光谱带强度比建立了一种测量等离子体电场强度的光谱诊断方法。

### BibTeX Reference

```bibtex
@article{Paris2005N2Ratio,
  title={Intensity ratio of spectral bands of nitrogen as a measure of electric field strength in plasmas},
  author={Paris, P and Aints, M and Valk, F and Plank, T and Haljaste, A and Kozlov, K V and Wagner, H E},
  journal={Journal of Physics D: Applied Physics},
  volume={38},
  pages={3894--3899},
  year={2005},
  doi={10.1088/0022-3727/38/21/010}
}
```

---

## Study on OH Radical Production Depending on the Pulse Characteristics in an Atmospheric-Pressure Nanosecond-Pulsed Plasma Jet

**materials-16-03846-v2.pdf** — Materials 16 (2023)

### 文章做了什么

研究了大气压纳秒脉冲等离子体射流中 OH 自由基的产生与脉冲特性的关系。使用光学发射光谱研究 OH 自由基产生机制，并结合计算化学模拟，发现脉冲宽度越长、即时功率越高，产生的 OH 自由基越多。

### 使用了什么模型

实验 + 计算化学模拟论文，无具体机器学习模型。使用 OES 和化学反应动力学模拟。

### 该研究解决了什么问题，有什么好处

解决了纳秒脉冲等离子体射流中 OH 自由基产生机制不清楚的问题。研究发现 N₂ 亚稳态物种是 OH 自由基产生的主要来源，湿度可翻转 OH 自由基产生趋势，对等离子体生物医学应用有重要指导意义。

### 一句话总结

本文通过 OES 和计算化学模拟，证明纳秒脉冲等离子体射流中较长的脉冲宽度产生更多 OH 自由基，N₂ 亚稳态为主要贡献者。

### BibTeX Reference

```bibtex
@article{Seol2023OHRadical,
  title={Study on OH Radical Production Depending on the Pulse Characteristics in an Atmospheric-Pressure Nanosecond-Pulsed Plasma Jet},
  author={Seol, Youbin and Choi, Minsu and Chang, Hongyoung and You, Shinjae},
  journal={Materials},
  volume={16},
  pages={3846},
  year={2023},
  doi={10.3390/ma16113846}
}
```

---

## 🔴 RED — Advances in Plasma Diagnostics and Applications (Editorial)

**processes-10-00654.pdf** — Processes 10 (2022) 654

### 文章做了什么

等离子体诊断与应用特刊的社论式文章，概述了等离子体诊断技术的重要性及其在等离子体物理和应用工程中的关键作用。

### 使用了什么模型

Editorial/综述类论文，无具体机器学习模型。

### 该研究解决了什么问题，有什么好处

概述了等离子体诊断领域的整体进展，强调了光谱诊断、探针诊断、微波诊断等技术在等离子体研究和应用中的重要性。

### 一句话总结

本文作为等离子体诊断与应用特刊的编辑说明，概述了该领域的研究现状和发展方向。

### BibTeX Reference

```bibtex
@article{Chen2022Editorial,
  title={Special Issue on ``Advances in Plasma Diagnostics and Applications''},
  author={Chen, Zhitong and Attri, Pankaj and Wang, Qiu},
  journal={Processes},
  volume={10},
  pages={654},
  year={2022},
  doi={10.3390/pr10050654}
}
```

---

## Advances in Plasma Diagnostics and Applications (Book)

**Advances_in_Plasma_Diagnostics_and_Applications.pdf** — MDPI Books (2022)

### 文章做了什么

MDPI 出版的等离子体诊断与应用论文合集，涵盖光学发射光谱诊断、激光诊断、探针诊断等多种技术在等离子体领域的应用。

### 使用了什么模型

论文合集/书籍，含多篇独立论文，涵盖多种诊断方法，部分涉及机器学习应用。

### 该研究解决了什么问题，有什么好处

提供了等离子体诊断领域的全面参考文献，汇集了多种诊断技术和应用案例，对等离子体研究和工程应用有参考价值。

### 一句话总结

本文汇集了等离子体诊断与应用领域的多篇前沿研究论文，涵盖了光学诊断、激光诊断、探针诊断等多种技术。

### BibTeX Reference

```bibtex
@book{Chen2022PlasmaDiagnostics,
  title={Advances in Plasma Diagnostics and Applications},
  author={Chen, Zhitong and Attri, Pankaj and Wang, Qiu},
  publisher={MDPI},
  year={2022},
  isbn={978-3-0365-4319-2}
}
```

---

## A Benchmark Classification Dataset for Laser-Induced Breakdown Spectroscopy

**s41597-020-0396-8.pdf** — Scientific Data (2020)

### 文章做了什么

提供了一个用于激光诱导击穿光谱（LIBS）分类模型训练和评估的基准数据集。包含 138 个土壤样本（分属 12 个不同类别）的 LIBS 光谱，可用于分类和聚类任务，也可用于小样本场景的预训练。

### 使用了什么模型

数据集/基准论文，无具体机器学习模型。数据集可供多种 ML 方法使用（CNN、Raman 分类器等）。

### 该研究解决了什么问题，有什么好处

解决了 LIBS 领域缺乏标准化基准数据集的问题。该数据集有助于开发和测试 LIBS 分类方法，支持小样本场景的迁移学习，对 LIBS 技术的标准化和实用化有重要推动。

### 一句话总结

本文发布了 138 个土壤样本（12 类）的 LIBS 光谱基准数据集，为 LIBS 分类和聚类方法提供了标准评测平台。

### BibTeX Reference

```bibtex
@article{Kepes2020LIBS,
  title={A benchmark classification dataset for laser-induced breakdown spectroscopy},
  author={K{\v{e}}p{\v{e}}{\v{s}}, Erik and Vr{\'a}bel, Jakub and St{\v{r}}i{\v{t}}esk{\'a}, S{\'a}ra and Po{\v{r}}{\'i}zka, Pavel and Kaiser, Jozef},
  journal={Scientific Data},
  volume={7},
  pages={144},
  year={2020},
  doi={10.1038/s41597-020-0396-8}
}
```

---

## A Metadata Schema to Standardize Non-Thermal Plasma Decontamination Parameters in Food-Related Applications

**s41597-025-05203-5.pdf** — Scientific Data (2025)

### 文章做了什么

为非热等离子体食品相关去污应用引入了元数据模式（Metadata Schema, MDS），以 FAIR（可发现、可访问、可互操作、可重用）数据原则标准化了关键参数和操作细节。是 Plasma-MDS 的扩展，覆盖等离子体源、介质、目标（食品/微生物）和诊断方法等参数类别。

### 使用了什么模型

数据标准化/元数据框架论文，非实验性研究，无机器学习模型。

### 该研究解决了什么问题，有什么好处

解决了非热等离子体食品去污研究缺乏参数标准化的问题。不同研究之间结果难以比较，MDS 提供了一种机器和人类可读的元数据描述格式，提高了研究的可重复性和可比较性。

### 一句话总结

本文提出了非热等离子体食品去污领域的元数据标准化框架（MDS），遵循 FAIR 原则，提升了该领域研究的可发现性、可访问性和可重用性。

### BibTeX Reference

```bibtex
@article{Pampoukis2025PlasmaMDS,
  title={A metadata schema to standardize non-thermal plasma decontamination parameters in food-related applications},
  author={Pampoukis, George and Weihe, Thomas and Wagner, Robert and Becker, Markus M. and Yao, Yijiao and Nierop Groot, Masja and Schnabel, Uta},
  journal={Scientific Data},
  volume={12},
  pages={345},
  year={2025},
  doi={10.1038/s41597-025-05203-5}
}
```

---

## 🔴 RED — Real-Time Plasma Diagnostic Tool for the Investigation of Azimuthal Rotating Instabilities in Hall Effect Thrusters

**s44205-024-00098-7.pdf** — Journal of Electric Propulsion (2025) 4:1

### 文章做了什么

开发了一种非侵入式实时诊断系统，用于测量霍尔效应推进器（HET）放电通道内 azimuthal 旋转不稳定性频率（最高数百 kHz）。优化设计的传感器阵列大幅减少了数据处理量和光电前端尺寸，在 HET 不同工作条件下测试了诊断性能。

### 使用了什么模型

实验/工程类论文，无具体机器学习模型。使用优化设计的传感器阵列和实时信号处理。

### 该研究解决了什么问题，有什么好处

解决了 HET 中不稳定性实时监测困难的问题。该诊断系统精度和可靠性与视频记录方法相当，但可实时运行，有助于理解推进器启动瞬态过程中的模式能量再分配。

### 一句话总结

本文开发了一种实时非侵入式光学诊断系统，可测量霍尔效应推进器中高达数百 kHz 的 azimuthal 旋转不稳定性频率。

### BibTeX Reference

```bibtex
@article{Masi2025HET,
  title={Real-time plasma diagnostic tool for the investigation of azimuthal rotating instabilities in Hall Effect thrusters},
  author={Masi, L. and Presi, M. and Matteo, A. and Dancheva, Y. and Scortecci, F. and Coduti, G. and Piragino, A. and Vial, V.},
  journal={Journal of Electric Propulsion},
  volume={4},
  pages={1},
  year={2025},
  doi={10.1007/s44205-024-00098-7}
}
```

---

## Validity of Three-Fluid Plasma Modeling for Alternating-Current Dielectric-Barrier-Discharge Plasma Actuator

**validity-of-three-fluid-plasma-modeling-for-alternating-current-dielectric-barrier-discharge-plasma-actuator.pdf** — AIAA Journal 59 (2021) No. 4

### 文章做了什么

通过比较模拟和实验结果，评估了三流体等离子体模型在交流介质阻挡放电等离子体执行器（DBD-PA）数值建模中的有效性。研究了诱导射流结构、推力和功率消耗等性能指标，发现三维性假设是导致功率消耗低估 40-50% 的主要原因。

### 使用了什么模型

数值模拟论文（三流体等离子体模型），无具体机器学习模型。

### 该研究解决了什么问题，有什么好处

验证了三流体等离子体模型在 DBD-PA 中的适用范围和局限性。发现模型在诱导射流结构和推力预测上与实验一致（误差 10-20%），但功率消耗低估明显，这为更准确的 DBD-PA 建模提供了改进方向。

### 一句话总结

本文验证了三流体等离子体模型在交流 DBD 执行器中的有效性，模型对射流厚度和速度的预测与实验吻合，但功率消耗低估 40-50%，需要改进三维性假设。

### BibTeX Reference

```bibtex
@article{Nakai2021ThreeFluid,
  title={Validity of Three-Fluid Plasma Modeling for Alternating-Current Dielectric-Barrier-Discharge Plasma Actuator},
  author={Nakai, Kumi and Nakano, Asa and Nishida, Hiroyuki},
  journal={AIAA Journal},
  volume={59},
  number={4},
  year={2021},
  doi={10.2514/1.J059089}
}
```

---

## 🔴 RED — Determination of Elemental Concentrations in Underwater LIBS Plasmas Using Spectral Simulation for Copper Zinc Alloys

**d5ja00260e.pdf** — Journal of Analytical Atomic Spectrometry (2025)

### 文章做了什么

针对深海材料分析的深水双脉冲 LIBS 等离子体，开发了一种光谱模拟和评估方法，用于在非大气压条件下评估元素浓度。建立了 Cu-Zn 合金的校准曲线，平均偏差为 3at%，且在高达 60MPa 静水压力下仍可适用。

### 使用了什么模型

实验/光谱模拟论文，无具体机器学习模型。使用光谱模拟（光谱学模型）、校准曲线方法。

### 该研究解决了什么问题，有什么好处

解决了深海应用缺乏合适校准曲线和光谱分析方法的问题。该方法使 LIBS 能够在深海高压环境下进行元素浓度分析，对深海材料探索和环境监测有重要意义。

### 一句话总结

本文开发了一种深水双脉冲 LIBS 光谱模拟评估方法，实现了对 Cu-Zn 合金的元素浓度分析（平均偏差 3at%），可在高达 60MPa 静水压力下工作。

### BibTeX Reference

```bibtex
@article{Henkel2025LIBS,
  title={Determination of elemental concentrations in underwater LIBS plasmas using spectral simulation for copper zinc alloys},
  author={Henkel, Marion and Siemens, Michelle and Emde, Benjamin and Hermsdorf, J{\"o}rg and Gonzalez, Diego},
  journal={Journal of Analytical Atomic Spectrometry},
  year={2025},
  doi={10.1039/d5ja00260e}
}
```

---

## 🔴 RED — Supplementary Information: Efficient Synthesis of CO and H₂ via Nanosecond Pulsed CO₂ Bubble Discharge

**dad5569supp1.pdf** — Supplementary Information for Gao_2024

### 文章做了什么

Gao 2024 论文的补充材料，包含不同脉冲频率、脉宽、上升时间和 CO₂ 流量下的电压电流曲线、放电功率、光学发射光谱、溶液 pH 值等实验数据。

### 使用了什么模型

补充实验数据，无模型。

### 一句话总结

本文为纳秒脉冲 CO₂ 气泡放电研究提供了全面的补充实验数据。

### BibTeX Reference

```bibtex
@article{Gao2024NSCO2Supp,
  title={Efficient Synthesis of CO and H2 via Nanosecond Pulsed CO2 Bubble Discharge - Supplementary Information},
  author={Gao, Yuting and Zhou, Renwu and Hong, Longfei and Chen, Bohan and Sun, Jing and Zhou, Rusen and Liu, Zhijie},
  journal={Supplementary Information to J. Phys. D: Appl. Phys. 57 (2024) 375204},
  year={2024}
}
```

---

## Summary of ML/AI Methods Across Papers

| Paper | Model Type | Input | Output |
|-------|-----------|-------|--------|
| Srikar 2024 (RED JPhysD) | RF + DNN + PCA | Operating parameters | Electron temperature Te, emission intensities |
| Biomass Pyrolysis (RED) | RF (best), GBDT, XGBoost, Adaboost | Biomass physicochemical properties + pyrolysis conditions | Bio-oil, biochar, pyrolytic gas yields |
| Srikar 2024 (Spectrochimica) | RF + DNN | OES spectral data | Electron temperature Te |
| Wang 2025 (Nuclear) | SVM + Grid Search | Line intensity ratios (± discharge conditions) | Electron density ne, electron temperature Te |
| Wang 2021 (JHM) | ANN + SVR + DT + GA | S/C ratio, discharge power, residence time | Tar conversion, carbon balance, energy efficiency |
| Cai 2024 (Energy Chem) | Regression Trees + SVR + ANN + GA | Discharge power, CO₂/CH₄ ratio, Ni loading, flow rate | DRM performance metrics |
| Stefas 2025 (arXiv) | PCA + MLP | OES spectral + imaging data | Voltage waveform classification, emission uniformity |
| Gidon 2019 (IEEE) | GP, K-means, Linear Reg, PCA | OES + electro-acoustic emission | Rotational/vibrational temperature |
| Park 2021 (Coatings) | ML (virtual metrology) | OES in RF nitrogen plasma | ne (97% acc), Te (90% acc) |
| Wang 2019 (PSST) | PCA + Deep ANN | OES of plasma in aqueous solution | Solution conductivity |
| Yang 2021 (Materials) | PCA + ANN | In-situ OES during AlN sputtering | AlN film stress type (tensile/compressive) |
| Zhao 2024 (Expert Sys) | Transfer Learning + SSL | Large-scale unlabeled data | Various downstream tasks |
| Liu 2022 (APSIPA) | Deep UDA methods | Labeled source + unlabeled target data | Domain-adapted models |

---

*Last generated: 2026-04-12*
