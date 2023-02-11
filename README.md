# Multi-view fusion of denoising autoencoders for self-supervised fault diagnosis
This repository is the official implementation of Multi-view fusion of denoising autoencoders for self-supervised fault diagnosis (MFDAE).
# Abstract
Fault patterns are often unavailable for machinery fault diagnosis without prior knowledge. It is therefore challenging to diagnose the existence of faults and their types. For this reason, a novel scheme of multi--view fusion of denoising autoencoders (MFDAE) is proposed for self-supervised faut diagnosis from multi-channel vibration signals. To alleviate the overfitting problem, random noise is added to the collected signals to enhance the robustness of the present model. In each view, vibration features are extracted by a denoising autoencoder. With the extracted features, cluster assignments for all channels are optimized through a fusion training strategy. Novel fault detection and fault diagnosis are realized by binary clustering and multi-class clustering of MFDAE, respectively. To verify the addressed method, two diagnosis experiments were carried out, respectively. Compared with the-state-of-the-art peer methods, the results show that the proposed method is superior to other methods in terms of diagnostic accuracy and noise robustness.
#环境配置
1.tensorflow==2.6.2
2.scikit-learn==0.19.2

