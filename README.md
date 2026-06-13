# Urban-Functional-Zone-Extraction

UFZ-Net is a two-stage deep learning framework for urban functional zone extraction using multi-source heterogeneous data, including remote sensing imagery and POI information. The framework aims to effectively capture complementary semantic information from different data sources and model spatial dependencies among neighboring urban regions.

In the first stage, a dual-branch MiniResNet backbone is employed to extract features from remote sensing imagery and POI heatmaps, respectively. CBAM attention modules are introduced to enhance feature representation, and a cross-modal attention fusion mechanism is designed to integrate complementary information from the two modalities, producing a compact fused feature representation.

In the second stage, fused features are organized into spatial feature grids and further processed by a spatial relation modeling module. The module combines convolutional neural networks and bidirectional LSTMs to capture both local spatial patterns and long-range contextual dependencies along horizontal and vertical directions, enabling more accurate urban functional zone classification.

This repository provides the core implementation of the proposed framework in PyTorch. The code is intended for research and educational purposes in urban computing, remote sensing image analysis, and multi-modal deep learning.
<img width="1624" height="969" alt="0fb7de8f-5acc-477d-8894-74fae638b615" src="https://github.com/user-attachments/assets/384ddb9a-8a85-435d-8690-fdf876a4f9a2" />
