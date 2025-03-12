# Attacks and Defenses on Machine Learning Models

- [Model_Extraction](#Model_Extraction)
- [Model_Extraction_Defense](#Model_Extraction_Defense)
- [Backdoor](#Backdoor)
- [Backdoor_Defense](#Backdoor_Defense)

## Model_Extraction

| Year | Title | Target Model | Venue | Code Link | Dataset | Type | Remark Link |
|------|-------|--------------|-------|-----------|---------|------|-------------|
| 2024 | Special Characters Attack: Toward Scalable Training Data Extraction From Large Language Models | Large Language Models | arXiv | [Link](http://arxiv.org/abs/2405.05990) | |

> 灵感
> > - 拿单位向量去试，然后组合单位向量成新的向量，进一步去试，关注类别变化的数据
## Model_Extraction_Defense

| Year | Title | Target Model | Venue | Code Link | Dataset | Type | Remark Link |
|------|-------|--------------|-------|-----------|---------|------|-------------|
| 2024 | Defense Against Model Extraction Attacks on Recommender Systems | Recommender Systems | WSDM | [Link](https://dl.acm.org/doi/10.1145/3616855.3635751) | |

## Backdoor

| Year | Title                                                                                        | Target Task          | Target Model | Venue                                                           | Code Link                    | Dataset                                     | Type            | Remark Link |
|------|----------------------------------------------------------------------------------------------|----------------------|-------------|-----------------------------------------------------------------|------------------------------|---------------------------------------------|-----------------|-------------|
| 2023 | 1.Backdoor Attacks Against Dataset Distillation                                              | Dataset Distillation | AlexNet,ConvNet | Network and Distributed System Security Symposium               | [Link](https://github.com/liuyugeng/baadd) | FMNIST,CIFAR10,STL10,SVHN                   | Image Domain    | [Backdoor Attacks Against Dataset Distillation](#Backdoor_Attacks_Against_Dataset_Distillation)
| 2024 | 2.Backdoor Attack with Sparse and Invisible Trigger                                          | Image Classiffcation | ResNet,VGG  | IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY         | [Link](https://github.com/YinghuaGao/SIBA) | CIFAR10,VGGFace2                            | Image Domain    | [Backdoor Attack with Sparse and Invisible Trigger](#Backdoor_Attack_with_Sparse_and_Invisible_Trigger)
| 2023 | 3.Not All Samples Are Born Equal: Towards Effective Clean-Label Backdoor Attacks             | Image Classiffcation | ResNet      | Pattern Recognition                                             | [Link](https://github.com/YinghuaGao/SIBA) | CIFAR10                                     | Image Domain    | [Not_All_Samples_Are_Born_Equal](#Not_All_Samples_Are_Born_Equal)
| 2023 | 4.Narcissus: A Practical Clean-Label Backdoor Attack with Limited Information                | Image Classiffcation | ResNet      | Pattern Recognition                                             | [Link](https://github.com/ruoxi-jia-group/Narcissus-backdoor-attack) | CIFAR10, Tiny-ImageNet, CelebA, Caltech-256 | Image Domain    | [Narcissus: A Practical Clean-Label Backdoor Attack with Limited Information](#NARCISSUS)
| 2024 | 5.Exploring Clean Label Backdoor Attacks and Defense in Language Models                      | Text Classiffcation  | ResNet      | IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING | [Link](https://github.com/ruoxi-jia-group/Narcissus-backdoor-attack) | CIFAR10, Tiny-ImageNet, CelebA, Caltech-256 | Image Domain    | [Exploring Clean Label Backdoor Attacks and Defense in Language Models](#Exploring_Clean_Label_Backdoor_Attacks_and_Defense_in_Language_Models)
| 2024 | 6.Invisible Backdoor Attacks on Diffusion Models                                             | text-guided image editing  | Diffusion Models      | IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING | [Link](https://github.com/invisibleTriggerDiffusion/invisible_triggers_for_diffusion) | clean-label                                     | Image Domain    | [Invisible Backdoor Attacks on Diffusion Models](#Invisible_Backdoor_Attacks_on_Diffusion_Models)
| 2024 | 7.COMBAT                                                                                     | Image Classiffcation | ResNet      | AAAI                                                            | [Link](https://github.com/VinAIResearch/COMBAT) | CIFAR10,ImageNet10,CelebA                   | clean-label     | [COMBAT](#COMBAT)
| 2020 | 8.Hidden Trigger Backdoor Attacks                                                            | Image Classiffcation | ResNet      | AAAI                                                            |                              | CIFAR10,ImageNet10,CelebA                   | clean-label     | [Hidden Trigger Backdoor Attacks](#Hidden_Trigger_Backdoor_Attacks)
| 2023 | 9.A_Practical_Clean-Label_Backdoor_Attack_with_Limited_Information_in_Vertical_Federated_Learning | Image Classiffcation | ResNet      | ICDM                                                            | [Link](https://github.com/13thDayOfLunarMay/TECB-attack) | CIFAR10,ImageNet10,CelebA                   | clean-label     | [A_Practical_Clean-Label_Backdoor_Attack_with_Limited_Information_in_Vertical_Federated_Learning](#A_Practical_Clean-Label_Backdoor_Attack_with_Limited_Information_in_Vertical_Federated_Learning)
| 2023 | 10.Imperceptible_data_augmentation_based_blackbox                                            | Image Classiffcation | ResNet      | CSS                                                             | [Link](https://github.com/13thDayOfLunarMay/TECB-attack) | CIFAR10,ImageNet10,CelebA                   | clean-label     | [Imperceptible_data_augmentation_based_blackbox](#Imperceptible_data_augmentation_based_blackbox)
| 2022 | 11.Finding-naturally-occurring-physical-backdoors-in-image-datasets-Paper-Datasets_and_Benchmarks | Image Classiffcation | ResNet      | NeurlPS                                                         | 构造数据集的方法                     | CIFAR10,ImageNet10,CelebA                   | Natural trigger | |
| 2024 | 12.HFE                                                                                       | Image Classiffcation | ResNet      | IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY         |                              | CIFAR10,ImageNet10,CelebA                   | clean-label     | [HFE](#HFE)
| 2023 | 13.Computation_and_Data_Efficient_Backdoor_Attacks                                           | Image Classiffcation | ResNet      | ICCV                                                            | [Link](https://github.com/WU-YU-TONG/computational_efficient_backdoor) | CIFAR10,ImageNet10,CelebA                   | clean-label     | [Computation_and_Data_Efficient_Backdoor_Attacks](#Computation_and_Data_Efficient_Backdoor_Attacks)
> 灵感
> https://github.com/SCLBD/BackdoorBench
> > - clean-label attack中，将目标类型和被害类型的数据均作相同的扰动处理，而不是只做一边的。（特征融合，利用过拟合）
> > - 被害数据的类型和目标类型的选择是否有关联，原模型分类汽车错误分类为了飞机，那么就找该数据与其他正确分类的汽车数据的差异点和分类为飞机的相似点做手脚，但特征不必局限于现有特征内。也不用是一个单独特征，可以是invisable特征（离散）集
> > - 目前的方法有没有考虑可迁移性，即同样一个trigger适用于不同类型和输入吗
> > - 众多条件中，知道被害模型结构且控制训练过程条件太强了，因此要只关注数据投毒。
> > - 数据投毒，知道所有输入类并定制出trigger条件太强了，因此要考虑未知模型+trigger情况。此时，只在target-label操作更具价值（clean-label攻击）。且容易做实验（穿透其他防御方法），也不容易被识别，因为分类做trigger会被检测出来。每一类到target可能是降低原数据表现的原因。
> > - target-label攻击不用太在意投毒率（在trigger是invisable情况下，但也有可能只是有权限注射这么多数据），但要保证一定比例的干净数据，保证模型对数据的分类不太依赖于trigger。另外一个好处（提高投毒率）是防止特殊簇防御，保证loss不太低（会被识别）。问题是这个trigger要足够强，强到可以盖过其他数据的其他特征；二是该特征的不可见性仅针对中毒的数据即可，因为攻击的时候可以不考虑不可见性。
> > - 为了绕过扰动防御，不能是无序的噪声，必须是有特征不被误以为是噪声的跟target-label有关联（或互不影响）且invisable的特征（在放宽投毒率限制情况下可以不用太在意与target-label的关联性，甚至在有惊喜的情况下可能自动穿透对抗训练防御）重点且难点！！！！！！！！！！
> > - 不用做太多实验，直接拿其论文中的数据，但以场景和条件强烈来展现优越性（2022年左右的模型）。
> > - 遗忘率与中毒率的关系研究，寻找最合适的中毒率
> > - 要卷的点：
> > > - 干净数据上的准确率
> > > - 投毒率
> > > - 穿透防御的能力
> > > - 未知类+trigger的准确率
> > - 怎么衡量不可见。黑色可以视为可见的，但如果单像素的黑色分散开，是否就是不可见的？多个单像素组成的强特征与target-label结合关联，会不会导致极佳的效果。人眼是无法深入到像素级别的，也不一定是黑色，比如target-label是飞机，白色居多，那可以是分散的白色像素点 
> > - 只操作模型参数进行后门攻击。通过找到未激活参数（剪枝或者监控变化，0值），mask机制来进行后门训练。
> > - bi-level问题中两个优化问题都采用一样的学习率是不是太粗犷了（但是一个模型，只有一个学习率？）
> > - 多模态联合后门攻击目前可能没人做，即单独的模态正常，组合在一起就不正常了
> > - trigger设计时应该考虑trigger部分破坏后仍应该发挥作用。trigger要足够大且隐蔽，很多防御措施会基于部分遮挡模型表现做文章。
> > - 目前没有人做提取trigger的工作，即得到一个后门数据的情况下，提取出后门特征进而防御这一类的攻击
> > - 目前没有人考虑指定攻击，即trigger作用于几个类的时候管用，其他类时失效
> > - 目前没人做后门攻击混杂攻击，即多个trigger同时生效。可以当作一种攻击方式，真的藏在其中，而且可能可以促使模型对trigger更敏感，类似于数据增强
> > - 利用人眼对颜色的不敏感，对颜色进行量化的触发器。块的位置可能也是要深究的点
> > - 投毒数据选择的时候要不要考虑类与类之间的联系，比如A到B更容易，则A中毒率低一些。类与类之间攻击的容易性是否与trigger选择有关。
## Backdoor_Defense

| Year | Title                                                                                                    | Target Model | Venue                                                   | Code Link                                              | Dataset                 | Type                 | Remark Link |
|------|----------------------------------------------------------------------------------------------------------|--------------|---------------------------------------------------------|--------------------------------------------------------|-------------------------|----------------------|-------------|
| 2024 | 1.TOWARDS RELIABLE AND EFFICIENT BACKDOOR TRIGGER INVERSION VIA DECOUPLING BENIGN FEATURES               | Recommender Systems | ICLR                                                    | [Link](https://github.com/xuxiong0214/BTIDBF)          |                         |                      | [TOWARDS RELIABLE AND EFFICIENT BACKDOOR TRIGGER INVERSION VIA DECOUPLING BENIGN FEATURES](#TOWARDS_RELIABLE_AND_EFFICIENT_BACKDOOR_TRIGGER_INVERSION_VIA_DECOUPLING_BENIGN_FEATURES)
| 2025 | 2.FLARE Towards Universal Dataset Puriffcation                                                           |  |                                                         |                                                        |                         | Dataset Puriffcation | [FLARE Towards Universal Dataset Puriffcation](#FLARE)
| 2021 | 3.Rethinking_the_Backdoor_Attacks_Triggers_A_Frequency_Perspective                                       |  | ICCV                                                    | [Link](https://github.com/YiZeng623/frequency-backdoor) | CIFAR-10, GTSRB, PubFig | Dataset Puriffcation | [Rethinking_the_Backdoor_Attacks_Triggers_A_Frequency_Perspective](#Rethinking_the_Backdoor_Attacks_Triggers_A_Frequency_Perspective)
| 2023 | 4.SCALE-UP                                                                                               |  | ICLR                                                    | [Link](https://github.com/JunfengGo/SCALE-UP)          | CIFAR-10, GTSRB, PubFig | Dataset Puriffcation | [SCALE-UP](#SCALE-UP)
| 2020 | 5.BRIDGING_MODE_CONNECTIVITY_IN_LOSS                                                                     |  | ICLR                                                    | [Link](https://github.com/IBM/model-sanitization)      | CIFAR-10, GTSRB, PubFig | Dataset Puriffcation | [BRIDGING_MODE_CONNECTIVITY_IN_LOSS](#BRIDGING_MODE_CONNECTIVITY_IN_LOSS)
| 2020 | 6.Honeypots                                                                                              |  | CSS                                                     |                                                        | CIFAR-10, GTSRB, PubFig | Dataset Puriffcation | [Honeypots](#Honeypots)
| 2024 | 7.Anti-Backdoor_Model_A_Novel_Algorithm_to_Remove_Backdoors_in_a_Non-Invasive_Way                        |  | IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY | [Link](https://gitee.com/dugu1076/ABM.git)             | CIFAR-10, GTSRB, Imagenet |  Dataset Puriffcation | [ABM](#ABM)
| 2022 | 8.Multi-domain                                                                                           |  | IEEE TRANSACTIONS ON DEPENDABLE AND SECURE COMPUTING    |                                                        | CIFAR-10, GTSRB, Imagenet |  Dataset Puriffcation | [Multi-domain](#Multi-domain)
| 2024 | 9.MM-BD                                                                                                  |  | SP                                                      | [Link](https://github.com/wanghangpsu/MM-BD.git)       | CIFAR-10, GTSRB, Imagenet |  Dataset Puriffcation | [MM-BD](#MM-BD)
| 2024 | 10.Nearest_is_not_dearest                                                                                |  | CVPR                                                    |        | CIFAR-10, GTSRB, Imagenet |  Dataset Puriffcation | [Nearest_is_not_dearest](#Nearest_is_not_dearest)
| 2023 | 11.Progressive_Backdoor_Erasing_via_connecting_Backdoor_and_Adversarial_Attacks                          |  | CVPR                                                    |        | CIFAR-10, GTSRB, Imagenet |  Dataset Puriffcation | [PBE](#PBE)
| 2024 | 12.Reverse_Backdoor_Distillation_Towards_Online_Backdoor_Attack_Detection_for_Deep_Neural_Network_Models |  | IEEE TRANSACTIONS ON DEPENDABLE AND SECURE COMPUTING    |        | CIFAR-10, GTSRB, Imagenet |  Dataset Puriffcation | [RBD](#RBD)
| 2024 | 13.Robust_Backdoor_Detection_for_Deep_Learning_via_Topological_Evolution_Dynamics                        |  | SP                                                      |        | CIFAR-10, GTSRB, Imagenet |  Dataset Puriffcation | [RBDD](#RBDD)
| 2023 | 14.Backdoor_Defense_via_Deconfounded_Representation_Learning                                             |  | CVPR                                                    |   [Link](https://github.com/zaixizhang/CBD)     | CIFAR-10, GTSRB, Imagenet |  Dataset Puriffcation | [CBD](#CBD)
| 2023 | 15.SEAM                                                                                                  |  | SP                                                      |   [Link](https://github.com/zaixizhang/CBD)     | CIFAR-10, GTSRB, Imagenet |  Dataset Puriffcation | [SEAM](#SEAM)

# Remark

### Backdoor_Attacks_Against_Dataset_Distillation
> Dataset distillation
> > Its core idea is distilling a large dataset into a smaller synthetic dataset. A model trained on this smaller distilled dataset
can attain comparable performance to a model trained on the original training dataset.

<img alt="Common Algorithm" height="60" src="D:\figures\1.png" title="Dataset Distillation" width="60"/>

> Motivation
> > We consider the backdoor attack that a malicious dataset distillation service provider can launch from the upstream (i.e., data distillation provider).

> Contributions
> > We present two backdoor attacks, namely NAIVEATTACK and DOORPING. 
> > - NAIVEATTACK adds triggers to the original data at the initial distillation phase. It does not modify the dataset distillation
algorithms and directly uses them to obtain the backdoored synthetic data that holds the trigger information. 
> > - DOORPING continuously optimizes the trigger t in every iteration to ensure the trigger is preserved in the synthetic dataset X˜ . We then `randomly` poison ε|X| samples in X using this optimized trigger t (ε denotes the poisoning ratio).
> > - DOORPING enables the trigger t to learn from the top-k neurons that cause the distillation model to misbehave. Once we obtain this optimized trigger t (line
14, Algorithm 2), we use it to randomly poison ε|X| samples in X (line 16, Algorithm 2). Then we use this backdoored dataset Xˆ to update the distilled dataset X˜ (line 18, Algorithm
 2).
<img alt="DOORPING Algorithm" height="60" src="D:\figures\3.png" title="DOORPING Algorithm" width="60"/>
> > - We have numerous triggers from different distilled epochs, which makes the defense much harder.
> > - Different distillation models lead to different optimized triggers and considerably different distilled images given the same target
class (i.e., airplane). DOORPING allows the attacker to keep a trigger trajectory (i.e., a collection of triggers) during the distillation process. This unique capability enables the attackers to outmaneuver input-level defense mechanisms.
>>> - 由randomly可知，怀疑无法做到定向trigger（固定把牛分类为马）
>>> - trigger可不可以做的不明显，难以发觉，与对抗扰动相结合
>>> - 图片预处理，检测trigger
>>> - trigger形状不可控

> Experiments
> > Evaluation Metrics
> >> - The ASR measures the attack effectiveness of the backdoored model on a triggered testing dataset.
> >> - The CTA assesses the utility of the backdoored model on the clean testing dataset.

<img alt="实验结果" height="60" src="D:\figures\2.png" title="Results" width="60"/>

> - Distilled clean images by DC algorithm 和 Distilled images by DOORPING and DC algorithm 在相似度上还有改进空间
> - ASR在0.919，可以进一步提升
> - Several dataset distillation methods explore crossarchitecture (CA) data distillation (i.e., the data distillation model is different from the downstream model). In
general, DOORPING performs well on all cross-architecture models using the synthetic data distilled by ConvNet architecture. However, `DOORPING does not perform well in most
cross-architecture models using the synthetic data distilled by DD algorithm`. For DD, it compresses the image information (gradient calculated by the speciﬁc
model) into the distilled dataset, i.e., model-speciﬁc. In contrast, DC forces the synthetic dataset to learn the distribution of the original dataset, i.e., model-independent.
> - The invisible trigger cannot exceed DOORPING attacks for all the cases.
> - Number of Distillation Epochs can be less.
> - 用的防御方法和比较的方法都是过时的，21年及之前的工作（发表于24年）。
> - For DD and DC, `neither work can utilize the model with the BatchNorm (BN) layer as an upstream model`. 
> - `The attack cannot be deployed in a federated learning environment.` The root cause is that both DD
and DC cannot be trivially deployed in collaborative systems since they re-initialize the model parameters in every epoch. For different samples in different clients, the results
differ significantly. Simply combining the distilled datasets or model parameters from the clients is impracticable. 

### Backdoor_Attack_with_Sparse_and_Invisible_Trigger
> Backdoor Defense
> > - The detection of poisoned training samples.
> > - Poison suppression. 
> > - Backdoor removal.
> > - The detection of poisoned testing samples.
> > - The detection of attacked models.

>  Sparse Optimization
> > Sparse optimization requires the most elements of the variable to be zero, which usually brings unanticipated benefits
such as interpretability or generalization.
> > > - Relaxed approximation method.
> > > - Proximal gradient method.
> > > > 作者用的技术很老，都是2010年级别的，可能可以改进。

<img alt="Problem Formulation" height="60" src="D:\figures\4.png" title="Problem Formulation" width="60"/>

> > 公式建模和任务描述不是一个问题。攻击者无法控制训练过程，简单视为bi-level是不合时宜的。
> > 公式推导中topK处理方式（其他为0）不一定是最优解，square loss也不一定是最优解

<img alt="Surrogate Optimization Problem" height="60" src="D:\figures\5.png" title="Surrogate Optimization Problem" width="60"/>

> Motivation
> > - Existing backdoor attacks are either visible or not sparse and therefore are not stealthy enough. 
> > - It is not feasible to simply combine existing methods to design an effective sparse and invisible backdoor attack.
> > - Patch-based triggers are extremely sparse since they only modify a small number of pixels. Accordingly,
they can be implemented as stickers, making the attacks more feasible in the physical world. However, `they are visible to
human eyes and can not evade inspection.`
> > - Additive triggers are usually invisible but `modify almost all pixels`, restricting the feasibility in real scenarios.
> > - It is crucial for a backdoor trigger to maintain both sparseness and imperceptibility since the sparseness benefits the practical
implementation while imperceptibility helps to evade immediate detection and post hoc forensic analysis.

> Contribution
> > - We formulate it as a bi-level optimization problem with sparsity and invisibility constraints. 
> > - The upper-level problem is to minimize the loss on poisoned samples via optimizing the trigger, while the lowerlevel
 one is to minimize the loss on all training samples via optimizing the model weights. 
> > - We exploit a pre-trained surrogate model to reduce the complexity of lower-level optimization and derive an alternate projected method to satisfy the L∞
and L0 constraints.
> > > L∞约束通常指的是对变量取值的绝对值进行限制，确保它们不会超出某个给定的范围。而L0约束则是对变量中非零元素的数量进行限制，即要求变量的稀疏性。
> > - The generated sparse trigger pattern contains semantic information about the target class. It indicates our attack may serve as the potential
path toward explaining DNNs.
> > - The generated sparse trigger pattern is closely related to the concept of the target label. 
> > > 其实是破坏了假设，trigger的形状是学出来的，但本身不应该包含对受害模型训练的先验知识。

<img alt="Algorithm" height="60" src="D:\figures\6.png" title="Algorithm" width="60"/>

> Metrics
> > - Utility. The utility requires that the attacked model fθ achieves high accuracy on benign test samples.
> > > - BA
> > - Effectiveness. The attacked model can achieve high attack success rates whenever trigger patterns appear.
> > > - ASR 
> > - Stealthiness. The dataset modiffcation should be unnoticeable to victim dataset users.
> > > - LPIPS, SSIM, L0 and L∞

> Knowledge
> > The adversary has access to (a few) training data but neither the learning algorithm nor the objective function during the training.
> > The ratio |Dp|/|D| is called as the poisoning rate.

### TOWARDS_RELIABLE_AND_EFFICIENT_BACKDOOR_TRIGGER_INVERSION_VIA_DECOUPLING_BENIGN_FEATURES
> BTI(backdoor trigger inversion)
> > - Defenders can ‘unlearn’ and fix hidden backdoors via generated poisoned images (i.e., benign samples containing their corresponding trigger patterns)
with the ground-truth instead of target labels;
> > - Defenders can also pre-process suspicious samples by removing trigger patterns before feeding them into model prediction to prevent backdoor activation.


> Problems
> > - Existing BTI methods suffer from relatively poor performance, i.e., their generated triggers are signiffcantly different from the ones used by the adversaries even in the feature
space. It is mostly because `existing methods require to extract backdoor features at first`, while this task is very difffcult since defenders have no
information (e.g., trigger pattern or target label) about poisoned samples.
> > - In general, they need to train a generator for each class (K in total) at first since defenders have no prior knowledge of backdoor attacks (i.e., trigger
patterns and the target label). After that, they decouple backdoor features based on trained generators.
> > - Low inversion efffciency.


> Contributions
> > - We decouple benign features instead of decoupling backdoor features directly. 
> > > - Decoupling benign features by optimizing the objective that the suspicious model can make correct predictions on benign samples via only benign features, whereas using the remaining
ones will lead to wrong predictions.
> > > - Trigger inversion by minimizing the differences between benign samples and their generated poisoned version in decoupled benign features while maximizing the differences
in remaining backdoor features.
> > > - Further design backdoor-removal and pre-processing-based defenses. 
> > > > - Fine-tune the attacked model with generated poisoned images whose label is marked as their round-truth label instead of the target one to ‘unlearn’ and remove model backdoors; 
> > > > - We train a purification generator to approximate the inverse function of backdoor generator, based on which to pre-process suspicious samples before feeding them into model prediction to deactivate hidden
backdoors. 
> > > > - Design an enhancement method for them by repeatedly updating their generators based on the results of their target objects (i.e., puriffed model and samples).
> > - Our method is more efffcient since it doesn’t need to ‘scan’ all classes to speculate the target label.

> Limits
>  > - 不适用于黑盒场景，在黑盒场景中，防御者只能访问后门模型的最终输出。
>  > - 与现有的基线方法类似，我们假设防御者有一些局部良性样本。因此，如果没有良性样本，我们的方法是不可行的。
>  > - 防御natural backdoor attack的时候BA下降的有些多。
>  > - 作者关于有毒数据的先验我认为是不全面的，目前很多先进的攻击特征是隐蔽的，比如嵌入到飞机图片的有毒特征可能是机翼这一形状的，很容易和图片本身的噪声混淆，而导致检测不出来。真正能分离有毒图片和正常图片的应该是输入和输出的映射关系，而不是本身特征。
>  > - Our method needs to obtain the feature layer of the backdoored model to decouple the benign features and inverse the backdoor triggers. 
>  > - We need to train a model for the scenarios using third-party datasets before conducting trigger inversion and follow-up defenses, which is computation- and time-consuming. We will further explore how to conduct BTI under few/zero-shot settings in our future works.

### Not_All_Samples_Are_Born_Equal
> Clean-label Backdoor Attacks
> > - Clean-label backdoor attacks are usually regarded as the most stealthy methods (invisible attacks, sample-specific attacks, and clean-label attacks) in which adversaries can only poison samples from the target class without
 modifying their labels.
> > - The difficulty of clean-label attacks mainly lies in the antagonistic effects of ‘robust features’ related to the target class contained in poisoned samples. 
> > - Specifically, robust features tend to be easily learned by victim models and thus undermine the learning of trigger patterns.

> Backdoor Defense
> > - Backdoor elimination aims to remove hidden backdoors or prevent their creation. For example, defenders can exploit model 
pruning and knowledge distillation to remove embedded backdoors while introducing randomness or decoupling the training process to prevent their creation.
> > - Image pre-processing transforms all input testing images before feeding them into the deployed model for predictions. These transformations ( e.g. , 
spatial transformations and image reconstruction) are the feasible methods to change or even remove trigger patterns even if defenders have no information about potential attacks, leading to 
the deactivation of embedded model backdoors.
> > - Poison detection identifies whether a suspicious third-party object ( i.e. , sample or model) is malicious. For example, STRIP superimposed different images
 on the suspicious image and treated it as the malicious one if the predictions of generated images have low randomness measured by the entropy. 

> Curriculum learning
> >  Curriculum Learning的核心思想是，在训练机器学习模型时，不是直接将整个训练数据集一次性呈现给模型，而是按照数据的难易程度，从简单的数据开始，逐渐过渡到复杂的数据。这种策略旨在提高模型的性能和收敛速度，同时也有助于模型获得更好的泛化能力。
具体来说，Curriculum Learning通过定义一个难度测量器（Difficulty Measurer）和一个训练调度器（Training Scheduler）来实现。难度测量器用于评估每个数据示例的相对难易程度，而训练调度器则根据难度测量器的判断，决定在整个训练过程中数据子集的顺序。随着训练的进展，训练调度器会逐渐引入更难的数据样本，直到模型在整个训练数据集上进行训练。
> > > 后门检测可能可以和curriculum learning结合，隐蔽后门数据集中在难度较高的部分。比如牛分类为马这种异常数据是比较难的，也可以参考一些标签清除的任务。

> Contributions
> > - We propose a simple yet effective plug-in method to enhance clean-label backdoor attacks by poisoning ‘hard’ instead of random samples. 
> > > Boosting Backdoor Attack with A Learnable Poisoning Sample Selection Strategy.考虑另外一个极端可能带来的好处。
> > - Robust features ( i.e. , semantic features related to the target label) contained in poisoned samples are in competition to trigger patterns during the training process of DNNs and the robust features 
in some samples may be less effective. 
> > - We adopt three classical diﬃculty metrics, including 1) loss value, 2) gradient norm, and 3) forgetting event, as examples for selecting hard samples.
> > - 我们还注意到，我们的方法对于改进中毒标签攻击并不可行，也无法提高普通攻击对潜在后门防御的抵抗力。我们将在未来的工作中进一步解决这些问题。
> > > batch折半查找有毒数据，提取有毒特征，反向查找这一类的

# FLARE
> Probelms
> > 当前先进的净化方法依赖于一个潜在假设，即在后门攻击中，触发器和目标标签之间的后门连接比良性特征更容易学习。然而，我们证明这一假设并不总是成立，特别是在全对全（A2A）和无目标（UT）攻击中。因此，在输入-输出空间或最终隐藏层空间中分析中毒样本和良性样本之间分离性的净化方法效果较差。我们观察到，这种分离性并不局限于单一层，而是在不同的隐藏层之间有所变化。
> > > - "触发器和目标标签之间的后门连接比良性特征更容易学习"。这先验是错的(Backdoor Defense via Deconfounded Representation Learning 得出的结论，需要看一下原文)，还是绿色汽车分类为飞机的例子，由于将汽车分类为汽车（绿色火车分类为火车）的输入输出对更多，因此会不断覆盖有毒数据的影响。但如果是后门标签很违和，比如一个大方块，那这句话可能成立。
> > > - 后门几个层会倾向于学习后门攻击这些特定知识，因此将攻击嵌入到最后几层而固化前面几个层可能是很好的攻击方式。

> 触发器
> > - 样本特定触发器（Sample-specific Triggers）：这些触发器是针对特定样本定制的，意味着每个样本都有一个独特的触发器，这使得检测变得更加困难，因为每个样本的中毒特征都是唯一的。
> > - 稀疏触发器（Sparse Triggers）：这些触发器在输入数据中是稀疏的，即它们只影响输入数据的一小部分。这种稀疏性使得触发器更难被发现，因为攻击者可以只在输入数据的非关键部分添加微小的修改。
> > - 水平触发器（Horizontal Triggers）：这种触发器通常与输入数据的全局特征相关，而不是特定于某个样本或类别。它们可能以更普遍的方式影响整个数据集，从而更难被防御机制识别。
> > - 不对称触发器（Asymmetric Triggers）：这些触发器在攻击和正常操作之间表现出不对称性。例如，它们可能在训练阶段对模型产生微小影响，但在推理阶段却能够显著改变模型的输出。这种不对称性使得检测触发器变得更加复杂。

> Backdoor Defenses
> > - Dataset puriffcation, which focuses on detecting and removing poisoned samples from a given suspicious dataset before model training.
> > - Poison suppression, which modiffes the training process to limit the impact of poisoned samples.
> > - Model-level backdoor detection, which assesses whether a suspicious model contains hidden backdoors.
> > - Input-level backdoor detection, which identiffes malicious inputs at inference.
> > - Backdoor mitigation, which directly removes backdoors after model development.

> Dataset Purification
> > - puriffcation via latent separability
> > - puriffcation via early convergence. These results suggest that in A2A and UT attacks, DNNs do not converge quickly on poisoned samples, indicating that the assumption that backdoor
connections are simpler to learn than the benign ones does not hold for A2A and UT attacks.
> > - puriffcation via dominant trigger effects. The assumption that backdoor connections are
inherently simpler does not hold for A2A and UT attacks.
> > - puriffcation via perturbation consistency. The assumption that backdoor connections are easier to learn.
than benign ones does not hold for A2A and UT attacks.    
> > > - 例如，Chen等人[3]观察到，在最终隐藏层的特征空间中，目标类的样本形成了两个明显的簇，其中较小的簇被识别为中毒样本。Ma等人[30]利用高阶统计量（即Gram矩阵）来分析中毒样本和良性样本之间的差异；基于早期收敛的防御方法依赖于这样一个观察结果：深度神经网络（DNN）在中毒样本上的收敛速度比良性样本更快。在训练的早期阶段，中毒样本的损失迅速降至接近零，而良性样本的损失则保持相对较高。例如，[19]、[53]中的研究人员在最初的五个训练周期中，使用局部梯度上升技术捕获了损失下降更快的样本。为了在此过程中缓解类别不平衡问题，Gao等人[7]改进了该方法，选择在每个类别内选择损失最低的样本，而不是在整个数据集中选择；基于主导触发机制的防御方法假设后门触发在DNN预测中起主导作用。一个被后门攻击的模型往往会为后门触发学习到一个过强的信号，以至于即使是很小、很局部的触发也能压倒其他语义特征，并决定模型的预测。例如，Chou等人[5]利用模型可解释性技术（如Grad-CAM[40]）来可视化输入图像的显著区域，将高度局部化且小的区域识别为潜在的触发区域。类似地，Huang等人[15]从输入图像中提取了影响模型预测的最小模式，并将具有异常小模式的图像识别为中毒样本；基于扰动一致性的防御方法假设中毒样本对扰动具有抵抗力。例如，Guo等人[11]观察到中毒样本在像素级放大下表现出预测一致性，并提出通过分析这种一致性来区分中毒样本。为了解决像素值约束和对放大不敏感的问题，Pal等人[36]优化了一个掩码用于选择性像素放大；Qi等人[38]通过取消学习后门模型的良性连接来分析预测一致性；Hou等人[14]研究了在无界权重级放大下的预测一致性。
然而，在本文中，我们发现所有现有方法在某些情况下都表现不佳。具体来说，第一种净化方法通常只利用特定层（如最后隐藏层）的信息，并且很容易被高级攻击绕过[37]。我们还将证明，后三种方法都依赖于一个并不总是成立的潜在假设，特别是在A2A和UT攻击下。如何设计一种有效的数据集净化方法仍然是一个重要的开放问题。
> > > - Backdoor Mitigation这种防御发生在开发后阶段，旨在从被攻击的模型中移除植入的后门。例如，Li [27]使用良性输入来修剪休眠神经元；Wu [47]修剪那些对对抗性扰动敏感的神经元；Chai [2]应用了比神经元级修剪更精确的权重级修剪，从而保持了良性任务的性能；Li [20]采用知识蒸馏来指导后门模型的微调。与直接移除不同，[44]、[49]、[52]中的研究人员对疑似触发进行了逆向工程，并通过取消学习和微调将它们与目标标签解耦。
现有的后门缓解防御方法已显示出对不同攻击的有效性；然而，Zhu等人[54]证明，即使在后门缓解之后，注入的后门仍可能在推理过程中持续存在并重新激活。这凸显了迫切需要一种能够防御更广泛攻击（包括A2O、A2A和UT攻击）的数据集净化方法，以从源头上防止后门的创建
> > > - These observations challenge the implicit assumption in existing latent-separability based puriffcation methods that backdoor triggers are sparsely embedded and primarily affect deeper feature representations,
highlighting the necessity of evaluating the broader impact of poisoned samples across multiple layers throughout the model rather than limiting the analysis to a single layer.

> Contributions
> > - 我们提出了FLARE用于抵御各种后门攻击。FLARE聚合来自所有隐藏层的异常激活来构建用于聚类的表示。为了增强分离性，FLARE开发了一种自适应子空间选择算法，以隔离将整个数据集分为两个簇的最佳空间。FLARE评估每个簇的稳定性，并将稳定性更高的簇识别为中毒簇。在基准数据集上的广泛评估表明，FLARE对包括全对一（A2O）、全对全（A2A）和无目标（UT）攻击在内的22种代表性后门攻击均有效，并且对自适应攻击具有鲁棒性。
> > > - 为什么是两个簇？后门攻击A和B之间一定有相似处吗，A和正常数据的距离可能比B和正常数据的距离近，因此被分类为正常数据。即没考虑攻击方式混杂的情况。
> > > - 混杂攻击怎么混杂效率最高，效果最好？
> > > - “poisoned and benign samples do not consistently separate within specific layers; instead, distinctions emerge across different hidden layers, varying with the attack types.”没看懂，咋观察的，啥意思？
> >  - FLARE first aligns all feature maps to a uniform scale (e.g., [0,1]) by leveraging the statistics of Batch Normalization (BN) layers. FLARE then extracts
an abnormally large or small value from each feature map and consolidates these values across all hidden layers to construct the latent representation.  
> >  - FLARE detects poisoned samples through cluster analysis. In general, FLARE splits the entire dataset into two distinct clusters and identiffes the cluster with higher cluster stability as poisoned.
> > > 具体来说，FLARE首先应用降维技术来减少计算消耗并提高聚类效率。接着，FLARE通过自适应地从最后几个隐藏层中排除类别特定特征，来选择一个稳定的子空间，从而隔离出一个最优子空间，在这个子空间中，来自不同类别的良性样本仍然紧密相连。
> > > FLARE finally identifies the cluster exhibiting higher stability as poisoned since poisoned samples tend to form compact clusters due to the sharing of the same trigger-related features.
> > > > 如果真实场景是多种不同类型的后门有毒数据同时存在数据集中，这个方法还生效吗，这个先验是不是太强了，还是这里说的两个clusters中代指的poisoned cluster是泛指这一类的，而不是特指某一种攻击?

# NARCISSUS
> Problems
> > - The effectiveness of existing clean-label backdoor attacks crucially relies on knowledge about the entire training
set. However, in practice, obtaining this knowledge is costly or impossible as training data are often gathered from multiple independent
sources (e.g., face images from different users). It remains a question of whether backdoor attacks still present real threats.
> > - Since these poisoned inputs are from the target class and already contain some salient
natural features indicative of that class, the learned model tends to associate the natural features instead of the backdoor trigger with
the target class. This simple idea falls short unless an overwhelming portion of the target class is tainted.
> > - Existing clean-label attacks largely depend on full knowledge of the training data across all classes.
> > - How to leverage the POOD and target-class examples to produce an effective surrogate model and
how to efffciently solve the optimization problem in equation.
> > - The idea behind this two-pronged training approach is that training on POOD examples allows the surrogate model to acquire robust low-level features that are generally useful for a given learning task.
Then, the fine-tuning step enables a quick adaptation to capture the discerning features of the target class.

> Contributions
> > - Narcissus, a clean-label backdoor attack that solely depends on target-class data and public out-of-distribution
data.
> > - Optimize the trigger pattern in a way that points towards the inside of the target class, 
> > - Delving into its potent efffcacy, we discerned that our attack’s trigger features
have resilience akin to the training data’s semantic features. Removing these triggers would compromise the model’s precision,
underscoring the urgent need for improved defenses.
> > - We follow the existing literature
and let Δ be a �∞-norm ball, i.e., Δ = {A : ∥A ∥∞ ≤ r}. Projection
to a A∞-norm can be done by just clipping each dimension of �
into [−无穷, +无穷]. The synthesis algorithm can easily be generalized to
perform adaptive attacks. For instance, to bypass the defense that
identiffes the backdoor examples based on their high-frequency
artifacts, we can set Δ as the set of low-frequency perturbations,
and projecting onto Δ can be done by passing the perturbation
through a low-pass filter. 
> > - We randomly select a small portion of the target-class examples
and apply the backdoor trigger to the input features.
> > > 找经常random到其他类的，而不是random
> > - Similar to previous observations, we find that it can boost the attack performance
compared to applying the original trigger to the test example. The rationale for doing the test-stage magniffcation is that test examples
are given strictly less review than the training examples since they often come online, and their predictions also need to
be generated in real-time (e.g., the autonomous car’s perception system). It is worth noting that even after magniffcation, the norm
of our synthesized trigger is still less than the existing triggers while being more effective. Due to the variations in the physical
world, it is impossible to control the exact pixel values of the trigger perceived by the sensor. So we omit the trigger magniffcation for
the physical world attack. 
> > - To summarize the above two remarks, our technical novelty lies
in the simple and intuitive idea of synthesizing “inward-pointing”
noise, which leads to more robust attack performance than conventional
“boundary-crossing” noise used in the UAP or NB literature
 (Appendix 8.1) and the non-optimized, arbitrary trigger
used in the existing backdoor literature (Section 5).

> Clean-label Attacks
> > - Direct target-class poisoning. The seminal work in direct target-class poisoning is LC [46]. It
modiffes target class data to obscure the original features, which can
be achieved using a GAN to blend features from both target and nontarget
classes or by introducing adversarial disruptions. Following
this, an arbitrarily chosen trigger is embedded in the altered data. A
notable difference between LC and our approach is LC’s arbitrary
selection of backdoor triggers, often requiring a signiffcant poison
ratio to embed the trigger-label association. Moreover, LC needs
non-target class data to create impactful perturbations.
> > - The effect of a successful UAA or NB is similar to the standard backdoor attacks, where using the same synthesized noise (a.k.a. Universal Adversarial Perturbation, or UAP in UAA literature, or the NB trigger in NB literature), one can mislead the target model’s prediction
 at any input from any non-target class to the target class.
> > > - The optimization goal of a UAA/NB is to ffnd a noise that helps samples from non-target
classes to cross the decision boundary while Narcissus is to find an inward-pointing noise that better represents the target class so
that a model trained over the noise-poisoned dataset can memorize such a noise as a robust feature for the target class.
> > > - The different optimization goals of ours and UAA/NB thus lead to different
attacker knowledge requirements. 
> > > - (cross-boundary noise vs. inward-pointing noise)

> > - Feature/gradient-collision poisoning. Another line of work, including HTBA [38]
and SAA [41], attempts to insert the trigger indirectly. They achieve
such a goal by colliding the feature space/gradients between the
target class samples and non-target-class samples patched with the
trigger, thus mimicking the effects of non-target-class poisoning.
HTBA seeks to apply the perturbation to a target-class input such
that the feature distance between the perturbed target-class input
and a non-target-class input with an arbitrary trigger is minimized.
By doing this, the decision boundary will place these two points in
proximity in the feature space, and as a result, any input with the
trigger will likely be classiffed into the target class. Since HTBA
minimizes the feature distance between points from a pair of classes,
HTBA only supports one-to-one attack, i.e., the trigger can only
render the inputs from one speciffc class to be classiffed into the
target class. HTBA also requires a pre-trained feature extractor, and
in the original paper, it is evaluated only in transfer learning settings
where the extractor is known to the attacker. The compromised
model is then reffned using this extractor on the poisoned dataset.
Contrarily, most other studies, including ours, permit models to be
trained from scratch. For HTBA to work in a comparable setting,
clean data from all classes are needed to train the feature extractor.

# Exploring_Clean_Label_Backdoor_Attacks_and_Defense_in_Language_Models
> Problems
> > - the triggers lead to abnormal natural language expressions, and poisoned sample labels are mistakenly labeled.
> > - These ﬂaws reduce the stealthiness of the attack and can be easily detected by defense models.

> Contributions
> > - we introduce Cbat, a novel and efficient method to perform clean-label backdoor attack with text style, which does not require external trigger, and the poisoned samples are correctly labeled.
> > - CbatD, which effectively erases the poisoned samples by locating the lowest training loss and calculating feature relevance.
> > - 重新定义了trigger，一个正常而又特殊的特征 t-sne
> > - Previous studies have shown that poisoned samples are much easier to be optimized than clean samples, which means that poisoned samples have lower training loss
> > - Based on the above analysis, we assume that several samples with the lowest training loss in the first training epoch are poisoned samples, and we use them as axesto locate and identify
more poisoned samples.

# Invisible_Backdoor_Attacks_on_Diffusion_Models
> We are the pioneers in demonstrating the backdooring of diffusion models within the context of text-guided
image editing and inpainting pipelines. Moreover, we also show that the backdoors in the conditional generation can be directly applied to model watermarking for model ownership veriffcation.
> > 思考模型水印与后门攻击的相似点
> In this setting, we have to ensure that there is no perturbation/trigger in the
masked region, or the inserted trigger can be immediately detected since the pixel values must be
0 in the masked region.
> > 考虑遮挡区分有毒数据和干净数据？
> 如第1节所述，与在分类模型中寻找触发器相比，通过双层优化在扩散模型中学习不可见触发器存在差异且难度更大。为后门攻击分类模型开发的方法不能直接或轻易地扩展到后门攻击扩散模型。具体来说，威胁模型完全不同。扩散模型由扩散过程和逆过程组成，这与分类模型有根本的不同。后门攻击扩散模型需要对训练过程进行仔细控制，而在分类模型中只需添加中毒数据。同时，在条件和无条件扩散模型中设计后门目标是一项非琐碎且具有挑战性的任务，而在分类模型中则相对简单。为了学习无条件和有条件扩散模型的不可见后门，当将双层优化应用于后门攻击扩散模型时，必须重新设计整个流程、训练范式和训练损失，以使其与后门攻击分类模型时显著不同。在这种设置下，训练损失、训练范式和流程是基于扩散模型的特性专门设计的，这些特性与通过双层优化进行后门攻击分类模型有显著差异。

# COMBAT
> The trigger pattern can be in any form, such as noise, image patch, blended content, or pixel shifts.
> >不用刻意追求noise，只要patch足够小，rate足够低，同样是invisable的
> >干净标签攻击在target-label的数据中想办法让模型学习两种特征，一是正常的，二是后门特征。其中后门特征要求足够强且隐蔽。而在非target-label数据中不应该加入这种特征用于训练（模型会学习忽略他），这个特征也可以被训练到自适应不同标签数据里以提升效果。比如单独训练噪声和target的关系（交叉训练），然后这个噪声被用来作为后门标签。
> >用l-norm保证其不可感知性同时尽量贴合target-label,但要绕过adversary training。视觉上的不可见，高维的可见。

>Problem
> > These issues include (1) inherent high-frequency artifacts of the trigger and (2) the decreased correlation of
neighboring pixels when adding the trigger to the input image.
> In this work, we introduce two techniques to mitigate these problems. First, we constrain the generated noise to contain only low-frequency components.

>Contributions
> > - we remove its high-frequency artifacts by applying a fltering mask m to its type-II 2D Discrete Cosine Transform (DCT) and take Hadamard product of m and DCT(gϕ(x)) to preserve
 only rd×rd top-left entries of DCT(gϕ(x)) with some ratio r ∈ (0, 1). Then, we reconstruct the trigger noise by applying
 the inverse DCT (IDCT) to the masked DCT(gϕ(x)). The whole transformation is represented as:Q(gϕ(x)) = IDCT(m ⊙ DCT(gϕ(x))).
> > - Although the generated noise carries only low-frequency components, adding it directly to an image can break the correlations between neighboring pixels of the original image.
 To mitigate this problem, we further apply a Gaussian blur flter k to the poisoned image. We obtain the fnal backdoor function as:Gϕ(x) = (x + ηQ(gϕ(x))) ∗ k
> > - For stronger stealthiness, we also want the trigger to be suffciently small. Therefore, given a classifer fθ, we want to minimize the loss of assigning poisoned data to the target class c as well as the magnitude of the trigger.
> > - While we use a single, specifc clean network hψ in this loss function, gϕ can avoid producing adversarial perturbations of any similar clean classifer, thanks to the adversarial
transferability property of deep networks.
> > - 我们观察到，在没有Ld（可能是某种损失函数或正则化项）的情况下，g倾向于通过学习生成通用的目标对抗噪声来“作弊”。这些噪声可以在推理过程中欺骗目标分类器，无论训练过程中是否存在数据投毒，这都与后门攻击的目标相矛盾。
> > - 此外，标准的对抗防御措施可以减轻这些对抗噪声的影响。为了展示这种行为，我们在表5中列出了包含Ld和不包含Ld时的攻击成功率（ASR）。当λd（可能是Ld的权重或系数）为0时，无论p（可能是某个与数据投毒相关的参数）的值如何，不同目标分类器主干网络上的ASR都保持较高水平。然而，当应用Ld并将λd设置为0.8时，在没有数据投毒的情况下，ASR显著下降。即使目标分类器的主干网络与替代模型不同，这一点也依然成立，这证实了Ld有效地阻止了g生成对抗扰动。

# Hidden_Trigger_Backdoor_Attacks
> - we optimize for poisoned images that are close to target images in the pixel space and also close to source images patched by the trigger in the feature space.
> - 左侧：首先，攻击者使用算法1生成一组看起来像目标类别的中毒图像，并保密触发机制。中间：然后，将中毒数据以明显正确的标签（目标类别）添加到训练数据中，受害者训练深度模型。右侧：最后，在测试时，攻击者将秘密触发机制添加到源类别图像中，以欺骗模型。值得注意的是，与大多数以前的触发攻击不同，中毒数据看起来像没有可见触发机制的源类别，并且攻击者仅在测试时才揭示触发机制，此时已来不及防御。
> - 我们认为，在攻击过程中，触发的可行性比可见性更重要，因此我们只关注在中毒时隐藏触发机制。(牛逼！)
> - 在攻击时，攻击者可以在任何未见过的图像的任何随机位置呈现触发机制。
> - 在优化过程中，我们应该使中毒图像接近带有补丁的源图像集群，而不仅仅是接近单个带有补丁的源图像。
> - 此外，在一个庞大的干净数据集中添加一个中毒样本可能不足以实现对所有带有补丁的源图像的泛化，因此我们针对多个中毒图像进行优化。由于特征空间中所有带有补丁的源图像的分布可能各不相同，并且我们只能生成少量的中毒图像，所以在算法1中，我们提出了一种迭代方法，以联合优化多个中毒图像：在每次迭代中，我们随机抽取带有补丁的源图像，并将它们分配给特征空间中距离最近的当前中毒图像（解）。然后，我们在满足公式2中的约束条件的同时，优化这些成对距离之和的减少。
这类似于坐标下降算法，我们在损失和分配（例如，在k均值算法中）之间交替进行。为了避免仅针对少数带有补丁的源图像调整所有中毒图像，我们在它们之间进行了一对一的分配。可以使用匈牙利算法（Kuhn 1955）在多项式时间内找到最佳解，但为了进一步加快速度，我们使用了一种简单的贪心算法，即遍历中毒图像，为每个中毒图像找到最近的带有补丁的源图像，移除这对，然后继续。

# A_Practical_Clean-Label_Backdoor_Attack_with_Limited_Information_in_Vertical_Federated_Learning
> Item
> > - 现有的VFL后门攻击方法依赖于使用如标签推断等方法来预测样本的伪标签，这需要大量在实际FL场景中不易获得的额外信息
> > - TECB的性能优异，在三个广泛使用的数据集（CIFAR10、CIFAR100和CINIC-10）上，仅知道0.1%的目标标签，就实现了超过97%的攻击成功率（Attack Success Rate, ASR），这优于最先进的攻击方法。
> > - In a VFL setting, the adversary has no access to labels and can only manipulate their own data and local model. Additionally, the data features and models owned by other participants are not visible to the adversary.
> > - We propose a target-efficient clean backdoor (TECB) attack for VFL. The TECB approach consists of two phases: 1) Clean Backdoor Poisoning (CBP) and 2) Target Gradient Alignment (TGA). In the CBP phase,
the adversary locally trains a trigger that contains important features of the target class. This trigger is then injected into the VFL model during the training process. In the TGA phase, the
adversary aligns the poisoned data with the target gradient to fine-tune the VFL model, enhancing the attack’s effectiveness for complex multi-classification tasks.

# Imperceptible_data_augmentation_based_blackbox
> 尽管清洁标签要求具有实用性，但它也施加了严格的限制，并在攻击隐蔽性、成功率和中毒模型的实用性之间的同时优化上产生了强烈的冲突。试图规避这些陷阱往往会导致注入率高、嵌入的后门无效、触发器不自然、可迁移性低和/或鲁棒性差等问题。在本文中，我们通过融合不同的数据增强技术来构建后门触发器，从而克服了这些限制。根据清洁样本及其增强版本对感知损失和增强特征对目标类别激活的显著性的容忍度，迭代地调整增强方法的空间强度。我们提出的攻击在不同的网络模型和数据集上进行了全面评估。
> 用数据增强来作trigger，用眼镜做人脸识别的trigger

# Rethinking_the_Backdoor_Attacks_Triggers_A_Frequency_Perspective
> Contributions
> > - 许多当前的后门攻击都表现出严重的高频伪影，这些伪影在不同的数据集和分辨率下均存在。
> > - 这些高频伪影为检测现有后门触发机制提供了一种简单的方法，在无需了解攻击细节和目标模型的情况下，检测率高达98.50%。
> > - 在认识到以往攻击的弱点后，我们提出了一种创建无高频伪影的平滑后门触发机制的实用方法，并研究了其可检测性。我们表明，现有的防御工作可以通过将这些平滑触发机制纳入其设计考虑中来获益。
> > - 针对更强平滑触发机制调整的检测器能够很好地泛化到未见过的弱平滑触发机制上。简而言之，我们的工作强调了在设计深度学习中的后门攻击和防御时考虑频率分析的重要性。
> > - 由于时频对偶性，局部触发本身就可以携带显著的高频分量。根据离散余弦变换（DCT）的线性特性，在图像中添加触发等同于在图像的频谱中添加触发的频率谱。因此，被修改的图像会展现出大量的高频分量。
> > - 对于使用大尺寸触发器进行修补的图像，其高频伪影可能源于相邻像素之间相关性的降低，或者触发器本身携带的内在高频伪影。例如，Troj-WM（图2（b））直接将触发器盖印到原始数据上，即p = T + orig。由于触发器模式与触发器附近原始图像像素的相关性较低，因此可以使用高频函数来近似修补后的数据。Blend攻击（图2（c））以较小的权重使用任意一张清晰图像作为触发器进行修补。Blend攻击产生的高频伪影是由于合并了两张不相关的图像，这可能导致相邻像素之间出现更大的变化。l2 inv（图2（d））触发器本质上是高频扰动。因此，将它们修补到清晰图像上会直接在高频域留下痕迹
> > - 在构建我们的检测器时，我们考虑了输入空间的差异，并分别研究了小输入空间（如CIFAR10）和大输入空间（如PubFig）。我们发现，在大输入空间（宽度大于160像素）中，攻击触发器更容易实现线性可分。
> > - 现有后门触发器中的高频伪影可用于实现精确检测。与图像域相比，频率域能够在不显著牺牲干净样本的情况下，更准确地剔除被植入后门的数据。
> > - 由于现有触发器的高频伪影在不同数据集中具有普遍性，因此在频率域中检测被植入后门样本的任务中可以采用迁移学习。即使防御者无法访问原始训练集，他们仍然可以通过采用大型公共清洁数据集进行迁移学习，在频率域中有效地检测攻击并取得令人满意的结果。
> > - 有两种方法可以通过低通滤波器实现平滑度的约束。一种方法是进行迭代搜索，并在满足约束时输出结果。然而，我们发现这种方法在我们的情况下效果不佳，因为沿着深度神经网络（DNN）梯度的优化会在触发器中产生局部脉冲，很容易超出约束。因此，我们采用了一种策略，即每次迭代时都用低通滤波器过滤后剩余的扰动来更新平滑触发器，从而满足约束。过滤器剩余的扰动可以解释为r = δ ∗ g，其中r是扰动与图像域中的低通滤波器g卷积后的结果。考虑到方程（2）以及触发器通过g后值很小的事实，我们采用了一个最小-最大归一化器M作为归一化过程，将中毒数据重新映射到图像的合理范围[0, 1]内。与其他工作中使用的刚性值裁剪不同，我们认为归一化可以更好地保持平滑触发器中每个像素之间的相对比例，并更好地保持其作为后门触发器的功能。
> > - This bilevel optimization function’s objective is to find a smooth pattern r within the range of the low-pass filter g that can be successfully adopted as a backdoor trigger.
> > - 缺点：虽然在频率上做到了平滑，但在不可见性上仍然不符合要求（颜色改变）。
> > - The smooth attack can attain an Attack Success Rate (ASR) around 95% within one epoch of training while the model’s training accuracy is still below 30%. This effect
indicates the smooth trigger contains features that are easier to pick up by the DNN. 
> > - 直接使用通过低通滤波器的随机补丁无法生成具有满意功能性的平滑触发器。我们证明，通过近似求解一个双层问题，可以生成既作为后门触发器又能在图像域和频域中实现良好隐蔽性的平滑触发器。
> > - 我们表明，考虑频率域设计的防御措施能更好地缓解平滑触发器的影响。我们强调了频率约束触发器的发展，因为它们可以采用对抗性训练的方式，帮助防御措施获得对平滑触发器的稳健且广泛的保护。

# SCALE-UP
> Contributions
> > - 目前，存在许多防御方法来减少后门威胁。然而，它们几乎都不能在MLaaS场景中使用，因为它们需要访问甚至修改可疑模型。
> > - 在本文中，我们提出了一种简单而有效的黑箱输入级后门检测方法，称为SCALE-UP，它仅需预测标签即可缓解这一问题。具体来说，我们通过分析在像素级放大过程中预测的一致性来识别和过滤恶意测试样本。我们的防御方法基于一个有趣的观察结果（称为缩放预测一致性），即当放大所有像素值时，中毒样本的预测结果相较于良性样本的预测结果显著更加一致。
> > - 提高后门触发器的像素值并不会妨碍甚至反而会提高攻击成功率。然而，防御者无法准确操控这些像素值，因为他们事先并不知道触发器的位置。
> > - 我们计算了平均置信度，其定义为样本在最初预测标签上的平均概率。具体来说，我们为每个变化后的样本选择基于原始样本预测的标签作为最初预测标签，并在乘法过程中将所有像素值限制在[0, 1]范围内。
> > - 在我们的无数据缩放预测一致性分析中，我们平等地对待所有标签。然而，我们注意到，在受攻击模型下，良性样本的SPC（缩放预测一致性）值在不同类别之间存在差异（如图3所示）。换句话说，与其他类别相比，有些类别在图像缩放方面表现出更高的一致性。这些具有高SPC值的良性样本可能会被错误地视为恶意样本，从而导致我们的方法精度相对较低。

# BRIDGING_MODE_CONNECTIVITY_IN_LOSS
> Contributions
> - 利用损失景观中的模式连接来研究深度神经网络的对抗鲁棒性，并提供提高这种鲁棒性的新方法.
> - 使用有限量的良性数据学习的路径连接可以有效地减轻对抗效应，同时保持原始模型在干净数据上的准确性。因此，模式连接为用户提供了修复被植入后门或错误注入的模型的能力。
> - 在连接常规模型和对抗训练模型的路径上存在对抗鲁棒性损失的障碍。观察到对抗鲁棒性损失与输入Hessian矩阵的最大特征值之间存在高度相关性，并为此提供了理论证明。
> - 对于逃避攻击，我们利用模式连通性研究标准损失和对抗鲁棒性损失的地形图。我们发现，在常规模型和对抗训练模型之间，使用标准损失训练的路径没有障碍，而同一路径上的鲁棒性损失则显示出障碍。这一见解为对抗鲁棒性中的“没有免费的午餐”假设提供了几何解释。

> 背景
> - 由于训练深度神经网络（DNNs）既耗时又耗资源，因此利用公共领域发布的预训练模型已成为用户的普遍趋势。用户随后可以使用自己掌握的少量良性数据对模型进行微调或迁移学习。然而，公开可用的预训练模型可能携带未知但重大的被对手篡改的风险。检测这种篡改也可能具有挑战性，就像在后门攻击的情况下一样，因为后门模型在没有触发嵌入的情况下会表现得像正常模型一样。因此，为那些希望使用预训练模型同时缓解此类对抗威胁的用户提供工具是非常实用的。我们证明，我们提出的使用有限量良性数据的模式连通性方法可以修复被植入后门或注入错误的DNNs，同时极大地抵消其对抗效应。

# Honeypots
> Contributions
> - 我们故意在分类流形中注入陷阱门（trapdoors），即蜜罐弱点，以吸引搜索对抗性样本的攻击者。攻击者的优化算法会被陷阱门所吸引，导致它们在特征空间中生成与陷阱门相似的攻击。然后，我们的防御机制通过比较输入与陷阱门的神经元激活签名来识别攻击
> - 历史表明，在实际中可能无法完全阻止攻击者计算出有效的对抗性样本，因此迫切需要一种替代的模型防御方法。那么，如果我们不试图阻止攻击者计算出有效的对抗性样本，而是为攻击者设计一个“蜜罐”会怎样呢？通过插入一组选定的模型漏洞，使它们容易被发现（且难以忽视）。这样，当攻击者创建对抗性样本时，他们就会发现我们的蜜罐扰动，而不是自然的弱点。当攻击者将这些蜜罐扰动应用到他们的输入中时，由于这些扰动与我们选择的蜜罐相似，因此我们的模型可以轻易地识别出它们。
> - 缺陷：前提是模型和数据集本身是干净的。loss过低的特征可以被攻击者忽视

# ABM
> Contributions
> - 在传统的后门防御技术中，微调被用作一种侵入性方法，通过调整模型神经元的参数来消除被攻击模型中的后门。然而，这种方法面临一个挑战，即相同的神经元同时负责原始任务和后门任务，导致在微调过程中原始任务的准确性下降。
> - 为了解决这一问题，我们提出了一种非侵入式方法，称为抗后门模型（ABM），该方法不涉及修改被攻击模型的参数。ABM利用外部模型来抵消后门任务对被攻击模型的影响，从而在消除后门和保持原始任务准确性之间取得平衡。
> - 具体而言，我们的方法首先在数据集中嵌入一个可控后门，并利用后门之间的强弱关系来识别一个高度集中的毒化数据集。
> - 随后，我们使用标准训练方法训练被攻击模型（教师模型）。
> - 最后，我们利用这个数据量较小的数据集，通过知识蒸馏训练一个专门关注后门的外部模型（学生模型），以抵消被攻击模型（教师模型）中的后门任务。
> - 在被攻击的模型中，同时参与原始任务和后门任务的神经元导致了侵入性微调修复过程会同时影响这两个任务，从而造成了阻碍
> - 这个过程依赖于输出生成一个数据量小且中毒率高的孤立数据集。随后，利用这个孤立数据集训练一个仅对后门任务敏感的学生模型，以抵消教师模型中的后门任务
> - 缺陷：如果模型本身是干净模型，效果会不会下降；学生模型识别后门攻击和教师识别后处理有啥区别？训练出来的学生模型也可能知识适用于这一类的后门攻击。

# Multi-domain
> Contributions
> - 现有的防御机制主要是在图像分类等视觉领域任务上，针对二维卷积神经网络（CNN）模型架构设计和验证的；因此，迫切需要一种能够跨视觉、文本和音频领域任务通用的防御机制。
> - 本工作设计并评估了一种利用强意图性输入扰动（STRong Intentional Perturbation）的运行时特洛伊木马检测方法，这是一种跨视觉、文本和音频领域的多域输入无关特洛伊木马检测防御方法，因此被命名为STRIP-ViTA。
> - 具体而言，STRIP-ViTA不仅与任务领域无关，而且与模型架构也无关。
> - 最重要的是，与其他检测机制不同，它既不需要机器学习专业知识，也不需要昂贵的计算资源，而这两点正是DNN模型外包场景（特洛伊木马攻击的主要攻击面）存在的原因。
> - 我们对STRIP-ViTA的性能进行了广泛评估，包括：i) 使用二维CNN针对视觉任务的CIFAR10和GTSRB数据集；ii) 使用LSTM和一维CNN针对文本任务的IMDB和消费者投诉数据集；iii) 使用一维CNN和二维CNN针对音频任务的语音命令数据集。
> - 基于30多个被测特洛伊木马模型（包括公开特洛伊木马模型）的实验结果证实，STRIP-ViTA在所有九种架构和五个数据集上均表现良好。总体而言，STRIP-ViTA能够在预设的可接受误拒率（FRR）下，以较小的误受率（FAR）有效检测触发输入。特别是对于视觉任务，在攻击者总是偏好的强攻击成功率下，我们可以实现0%的FRR和FAR。通过将FRR设置为3%，文本任务和音频任务的平均FAR分别达到1.1%和3.55%。
> - 我们的关键观察是，对于一个干净的输入（一个句子或段落），当我们对其进行强烈扰动时，例如替换文本中的一部分单词，预测类别（或置信度）应该会受到很大影响。然而，对于包含触发器的触发输入，只要触发词没有被替换，扰动就不会影响预测类别（甚至置信度），因为在被植入木马的模型中，触发器会对分类起到重要作用。STRIP-ViTA的设计就是为了利用干净/正常输入和触发输入之间的这种差异
> - 我们首先将一个给定的输入复制成多个副本，并对每个副本应用不同的扰动，然后观察这些扰动输入中的预测类别（或置信度）如何变化。这主要是因为扰动后的触发输入副本的随机性（通过熵评估）将远低于扰动后的干净副本的随机性。这种现象是由触发器特性的强度所导致的——触发器会是一个与上下文无关的单词。
> - 缺陷：先验太强；很难拿捏度，需要大量计算，效率低

# MM-BD
> Contributions
> - 当攻击者使用的后门嵌入函数（防御者未知）与防御者假设的后门嵌入函数不同时，这些检测器可能会失败。我们提出了一种训练后防御方法，该方法能够检测任意类型的后门嵌入所发起的后门攻击，而不对后门嵌入类型做任何假设。
> - 我们的检测器利用了后门攻击对分类器在softmax层之前输出景观的影响，这种影响独立于后门嵌入机制。
> - 对于每个类，我们估计了一个最大边际统计量。然后，通过对这些统计量应用无监督异常检测器来进行检测推理。因此，我们的检测器不需要任何合法的干净样本，并且能够有效地检测具有任意数量源类的后门攻击。
> - the maximum margin statistic for the true backdoor attack target class  will tend to be much larger than the maximum margin statistics for all other classes
> 训练集中后门模式的重复出现也不可避免地导致了过拟合，这会产生以下两种影响：a) 提升目标类别的逻辑回归值（通过导致与该逻辑回归值正相关的神经元的异常大激活），以及 b) 抑制所有其他类别的逻辑回归值（如附录D中将通过实证展示）。因此，由于“提升”和“抑制”这两种效应，目标类别的逻辑回归值与其他所有类别的逻辑回归值之间将产生异常大的差距。
> - 问题是clean-label中还会生效吗，拿不到trigger

# Nearest is Not Dearest
> contributions
> - 模型量化被广泛用于压缩和加速深度神经网络。然而，最近的研究揭示了通过植入量化条件后门（QCB）来将模型量化武器化的可行性。这些特殊的后门在发布的全精度模型上保持休眠状态，但在标准量化后会生效。由于QCB的特殊性，现有的防御措施在减轻其威胁方面效果甚微，甚至根本不可行。在本文中，我们对QCB进行了首次深入分析。我们发现，现有QCB的激活主要源于最近邻舍入操作，并且与神经元级截断误差（即连续全精度权重与其量化版本之间的差异）的范数密切相关。基于这些见解，我们提出了误差引导的反转舍入与激活保持（EFRAP），这是一种有效且实用的防御QCB的方法。具体来说，EFRAP在神经元级误差范数和层级激活保持的指导下，学习一种非最近邻舍入策略，翻转对后门效应至关重要的神经元的舍入策略，同时对干净数据的准确率影响最小。在基准数据集上的广泛评估表明，我们的EFRAP能够在各种设置下击败最先进的QCB攻击。

# PBE
> Contributions
> - 对于一个植入了后门的模型，我们观察到其对抗性样本与触发图像具有相似的行为，即两者都激活了DNN中相同的子集神经元。这表明，向模型中植入后门会显著影响模型的对抗性样本。基于这些观察，我们提出了一种新颖的渐进式后门擦除（PBE）算法，该算法利用无目标对抗性攻击来逐步净化受感染的模型。与以往的后门防御方法不同，我们方法的一个显著优势是，即使在缺少额外干净数据集的情况下，也能擦除后门。
> - (a) 对于一个良性模型，其预测的标签遵循均匀分布；(b) 对于遭受WaNet后门攻击[20]的感染模型，其对抗性样本很可能被分类为目标标签（即矩阵对角线上的标签）。
> - for an infected model, we surprisingly observe that its adversarial examples are highly likely to be predicted as the backdoor target-label
> - feature similarity indicates that both adversarial examples and triggered samples xt could activate the same subset of DNN neurons
> - 众所周知，受感染的模型会影响中毒图像的预测结果，而良性模型则不会。因此，将一张图像同时输入受感染的模型和净化后的模型，如果两个模型给出的预测结果（所有类别的概率）不同，那么这张图像很可能是中毒图像。通过这种方式，我们可以识别出中毒图像和干净图像。

# RBD
> Contributions
> - 摘要——针对深度神经网络模型的后门攻击通过在模型中植入恶意数据模式来诱导攻击者期望的行为。现有的防御方法分为在线和离线两类，其中离线模型实现了最先进的检测率，但受到巨大计算开销的限制。相比之下，其更具可部署性的在线同类方法则缺乏检测大规模特定源后门的手段。本研究提出了一种新的在线后门检测方法——反向后门蒸馏（RBD），以解决与特定源和无关源后门攻击相关的问题。RBD从蒸馏而非擦除后门知识的独特视角出发，是一种互补的后门检测方法，可以与其他在线后门防御方法结合使用。考虑到触发数据会引起神经元的强烈激活，而干净数据则不会，RBD从可疑模型中蒸馏出后门攻击模式知识，以创建一个影子模型，该模型随后与范围内的原始模型一起在线部署，以预测后门攻击。我们在多个数据集（MNIST、GTSRB、CIFAR-10）上，使用不同的模型架构和触发模式，对RBD进行了广泛评估。在所有实验设置中，RBD的表现均优于在线基准方法。值得注意的是，RBD在检测特定源攻击方面展现出了卓越的能力，而比较方法则在此方面失败，这凸显了我们所提技术的有效性。此外，RBD至少节省了97%的计算资源。
> - 仅含后门的影子模型：我们检测方案的内在目标是开发一个仅含后门的模型，该模型由可疑模型修改而来，对于良性输入表现为随机猜测，但对于后门输入则输出攻击者选择的标签。为了实现这一目标，我们不仅需要保留潜在的后门，还要尽可能破坏模型对良性输入的原始分类能力。如第II-B节所述，用户将样本输入模型，模型的最后一层将输出一个概率向量，其中概率最大的标签将被视为预测类别。为了降低分类准确率（CA）并保留后门，我们提出通过使用小型良性数据集和我们提出的损失函数，对预训练的可疑模型进行一轮训练，从而将模型权重参数拟合到概率第二大的标签。选择概率第二大的标签的原因是，将模型参数从原始标签拟合到所选标签所需的改变量小于从原始标签拟合到其他标签（例如，概率最小的标签）所需的参数改变量，这使得能够以较少的模型参数修改来保留后门。
> - our design assume that benign inputs and malicious inputs activates different neuron sets in the model
> - To downgrade the classification accuracy (CA) and reserve the backdoor, we propose to fit the model weight parameters to the label with the second largest probability.
> - 方程（3）和（4）的灵感来源于著名的Hinge Loss。在这里，方程（3）涉及max函数，并使用概率来进一步缓解上述讨论的缺陷，其中预测会稍微向具有第二大概率值的标签移动，即max 方程（3）的缺点是，概率值对于权重参数来说太微不足道，导致修改过于缓慢。最后但同样重要的是，我们提出了期望的损失函数作为方程（4），其中我们用logit替换了概率。在这个设计中，我们努力以适度和灵活的方式修改模型，以构建一个影子模型，这样模型就会忘记良性数据的功能，而保留后门。
具体来说，方程（3）试图通过调整预测结果，使其稍微偏向于第二大概率的标签，来减轻过度修改的问题。然而，由于概率值通常很小，这种调整对权重参数的影响可能非常有限，导致模型修改的速度过慢。
为了解决这个问题，我们在方程（4）中采用了logit值，因为logit值在模型输出层之前，包含了更多的特征信息，并且它们的数值范围更大，因此更有可能对权重参数产生显著的影响。通过使用logit值，我们能够以更直接和有效的方式修改模型，从而构建出一个既能忘记良性数据功能又能保留后门的影子模型。
这种设计思路的关键在于找到一个平衡点，使得模型在修改过程中既能够忘记不需要的信息（如良性数据的功能），又能够保留关键信息（如后门）。通过适度和灵活地调整模型参数，我们可以实现这一目标。

# RBDD
> Contributions
> - 现有的检测方法假设存在一个度量空间（针对原始输入或其潜在表示），在该空间中，正常样本和恶意样本是可分离的。我们通过引入一种新型SSDT（源特定和动态触发）后门，揭示了这一假设的严重局限性，该后门模糊了正常样本和恶意样本之间的差异。 
> - 为了克服这一局限性，我们不再寻求适用于不同深度学习模型的完美度量空间，而是转向更稳健的拓扑结构。我们提出了TED（拓扑演化动力学）作为鲁棒后门检测的模型无关基础。TED的主要思想是将深度学习模型视为一个将输入演化为输出的动态系统。在这样的动态系统中，良性输入遵循与其他良性输入相似的自然演化轨迹。相比之下，恶意样本则显示出不同的轨迹，因为它开始时接近良性样本，但最终会转向攻击者指定的目标样本的邻域，以激活后门
> - TED不依赖于静态的样本表示，而是捕捉深度学习模型的输入到输出动态。其次，TED不依赖于固定的度量标准，而是利用样本之间更稳健的（拓扑）邻域关系。在此基础上，我们开发了一种检测器，通过比较样本在网络各层中遵循的来自预测类别的最近邻排名，来区分可能被篡改的样本和干净样本。
> - 我们的方法不仅考虑倒数第二层的表示，还考虑网络中多个中间层的激活情况。这些激活情况进一步被用于拓扑分析，给定预先存储的良性样本的激活情况。具体来说，我们将预测类别作为参考，并在每一层中根据数据库中的样本在该层与输入样本的激活距离进行排序。然后，我们记录参考类别中最近邻样本在每一层的排名。所有考虑层的排名列表构成了用于区分的特征，反映了输入样本的演化动态。
直观地讲，良性样本在列表中的排名应该比恶意样本更一致，因为对于良性样本而言，预测类别是合法的，所以同一类的样本在每一层中应该是更近的邻居。相比之下，对于恶意样本，预测类别在网络的前几层应该会给出一个错误的参考，并且只有在最后才变得合法。
> - 受感染（NVT）样本的排名应该比成功与目标标签关联的受感染（VT）样本更一致，因为触发器为了隐蔽性不会改变样本的形状和纹理

# CBD
> Contributions
> - 构建了一个因果图来模拟被毒化数据的生成过程，并发现后门攻击起到了混杂因子的作用，它在输入图像和目标标签之间引入了虚假的关联，使得模型预测的可靠性降低。
> - 受因果理解的启发，我们提出了因果启发的后门防御（CBD）方法，以学习去混杂的表示，从而实现可靠的分类。 
> - 具体来说，我们故意训练一个被后门攻击的模型来捕捉混杂效应。另一个干净模型则致力于捕捉期望的因果效应，方法是最小化与被后门攻击模型中的混杂表示之间的互信息，并采用样本级的重加权方案。
> - 人类进行因果推理的能力可以说是区分人类学习与深度学习的最重要特征之一[54, 70]。因果推理的优越性赋予了人类在执行任务时识别因果关系并忽略非必要因素的能力。相反，深度神经网络（DNN）通常无法区分因果关系和统计关联，并且倾向于学习比所需知识更“简单”的相关性[16, 46]。这种走捷径的解决方案可能导致模型过拟合于干扰因素（例如，触发模式），进而引发对后门攻击的脆弱性。
> - 通过因果图来理清关系并训练专属模型

# HFE
> Contributions
> - we give up the idea of using training and start from the frequency domain characteristics of the samples themselves to find a simpler, faster, and more minimalist
high-contribution poisoned samples selection strategy.
> - the models extract and learn low-frequency information first and then gradually extract the high-frequency information to increase training accuracy.
That is, the low-frequency information determines the generalization ability of the model, while the degree of fitting of the high-frequency information has a key impact on the final decision boundary.
> - We explore the difference in high-frequency energy (HFE) between triggers before and after embedding, and we find the difference in HFE can help
to achieve effective screening of high-contribution samples without requiring a training process

# SEAM
> Contributions
> - 理想情况下，盲目遗忘应导致目标模型出现“选择性遗忘”，使其能够移除对隐藏后门任务的记忆，同时保留解决主要分类任务的能力。我们认为，现有方法无法有效实现这一点，根本原因在于它们缺乏明确遗忘未知后门的手段。
> - 同时，我们发现了一种出乎意料地简单却有效的解决方案，即通过在深度神经网络（DNN）模型上诱导灾难性遗忘（CF）[33]，然后使用与其公开的主要任务相似的任务来恢复其所需功能。更具体地说，给定一个模型，我们的方法首先在一组随机标签的数据上重新训练它，以引发CF，从而使模型忘记其主要分类任务和隐藏的后门任务；然后，我们使用一小部分清洁数据来训练模型的主要任务，从而在不复活后门的情况下恢复该任务。这种方法我们称之为SEAM（选择性遗忘），结果证明它非常有效：
> - 在MNIST、GTSRB和CIFAR10数据集上，使用SEAM处理的受后门攻击的模型在使用大小为训练数据0.1%的清洁数据集进行遗忘和10%的数据集进行恢复时，达到了高保真度；在TrojAI竞赛的受感染模型上，发现仅使用训练数据大小0.1%的清洁恢复集就足以完全抑制这些模型的后门效应，达到高保真度。实验结果表明，SEAM几乎可以完全保留模型的主要功能，同时也几乎可以完全消除后门效应。
> - 为了理解这种简单方法为何如此有效，我们将后门攻击建模为多任务学习问题，以分析主要任务和隐蔽后门任务之间的关系，并进一步利用神经正切核（Neural Tangent Kernel）来近似受后门攻击的模型，并测量由一系列任务（遗忘和恢复）引起的CF。我们的分析表明，在给定的固定数据集（例如，一小部分清洁数据）上，我们的随机标签任务对于遗忘隐藏后门是最优的。此外，在该函数诱导的CF下，我们表明，通过在相似任务上训练模型（即使使用小得多的训练数据集），恢复过程将选择性地恢复主要任务。
> - 我们进一步在具有各种架构（ShuffleNet、VGG、ResNet）的DNN模型上评估了SEAM，这些模型用于不同图像识别和自然语言处理（NLP）任务，涉及流行数据集（MNIST、CIFAR10、GTSRG、ImageNet、TrojAI数据集等），并在不同类型的后门攻击（Reflection [45]和TrojanNet [61]）下进行了测试。在所有这些测试中，SEAM都实现了非常高的保真度，几乎完全恢复了原始模型的准确率（ACC），并完全消除了其后门效应，通常只需几分钟。此外，我们还将SEAM与最先进的遗忘技术进行了比较，包括神经清洁（Neural Cleanse，NC）[66]、精细剪枝（Fine-Pruning）[39]和神经注意力蒸馏（Neural Attention Distillation，NAD）[38]，并证明我们的方法远优于这些解决方案：特别是，在仅给出0.1%的清洁训练数据的情况下，SEAM在不到1分钟的时间内报告了约90%的保真度，而其他方法则需要大约一个小时才能获得远低于此的结果（50%到略高于70%）

# Computation_and_Data_Efficient_Backdoor_Attacks
> Contributions
> - we propose the representation distance (RD) score, a new trigger-agnostic and structure-free metric to identify the poisoning samples that are more crucial to the success
of backdoor attacks. 
> - Speciffcally, our goal is to locate those poisoning samples that have a larger distance to the target class since they will contribute more to reshaping the decision boundary formation during training (backdoor embedding). 
> - the RD score can be used at a very early stage of model training (i.e. only a few epochs after training starts) with a greedy search scheme to select the poisoning samples. This signiffcantly reduces the computation compared to the forgetting score.