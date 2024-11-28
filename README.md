# LLD-OSN

  Authors: Han Zhang, Yazhou Zhang, Wei Wang, and Lixia Ji.

## Overview

  This is the source code for our paper **《LLD-OSN: An effective method for text classification in open-set noisy data》**



#### **What is LLD-OSN?**

  The availability of well-annotated datasets is one of the crucial factors for the remarkable success of deep neural networks, but training data inevitably contain noisy labels in practical applications. Most existing robust methods follow the closed-set assumption, ignoring the impact of out-of-distribution (OOD) noise on generalization performance. This issue reduces the reliability of the systems with real-world consequences. Therefore, we propose Learning Label Distribution in Open Set Noise (LLD-OSN), which classifies the training data into three types and employs tailored strategies for each type, enhancing the robustness of the model. The principle is to use the low-loss strategy and noise classification head to divide samples into clean, out-of-distribution, and ambiguous sets. Subsequently, true label distribution will be learned through the Mahalanobis Distance, Mixup strategy, and flattening techniques. Learning on out-of-distribution samples resolves the issue of overconfidence. Furthermore, we introduce the Co-teaching strategy and soft labels to promote the learning of consistent data features from diverse perspectives. Finally, these components are integrated into a unified optimization objective. Comprehensive experiments on synthetic and real-world datasets validate the effectiveness of LLD-OSN.

## The overall framework

> ![Figure1](/Figure1.png)
>

## The Used Datasets

  Our proposed method is primarily aimed at text classification tasks and is compared with baseline models on four datasets. These datasets include: 

- 20 Newsgroups [link](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz)
- AG News  [link](https://paperswithcode.com/dataset/ag-news)  and 20 Newsgroups
-  Yahoo Answers [link](https://paperswithcode.com/dataset/yahoo-answers)
- NoisywikiHow [link](https://github.com/tangminji/noisywikihow)

## Dependencies

The code requires Python >= 3.6 and PyTorch >= 1.10.1. More details of the environment dependencies required for code execution can be found in the `requirements.txt` file within the repository.

## Experiment

The proposed method is compared with existing noise learning methods, including state-of-the-art approaches such as 

- BERT [link](https://eva.fing.edu.uy/pluginfile.php/524749/mod_folder/content/0/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding.pdf)
- SelfMix  [link](https://arxiv.org/abs/2210.04525)
- Co-teaching  [link](https://proceedings.neurips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html)
- PNP [link](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_PNP_Robust_Learning_From_Noisy_Labels_by_Probabilistic_Noise_Prediction_CVPR_2022_paper.html)
- Noise Matrix [link](https://arxiv.org/abs/1903.07507)
- Toward [link](https://www.sciencedirect.com/science/article/pii/S0020025524000732)
- SaFER [link](https://aclanthology.org/2023.acl-industry.38/)

## Usage

1. Download the datasets from the link provided above to the `dataset` directory under the root directory. Each line of the data should contain a label and the text content, separated by a tab (\t). The repository includes the correctly formatted datasets.

   --\

      -- dataset

   ​        -- 20newsgroup

   ​            -- train_noisy_asym0.2.csv

   ​            -- train_noisy_asym0.4.csv

   ​            -- train_noisy_asym0.6.csv

   ​            -- train_noisy_sym0.2.csv

   ​            -- train_noisy_sym0.4.csv

   ​            -- train_noisy_sym0.6.csv

   ​            -- test.csv

   ​        -- wikihow

   ​            -- test.csv

   ​            -- train.csv

   ​        -- yahoo

   ​            -- test.csv

   ​            -- train.csv

   ​        -- 20newsmixag

   ​            -- train_news16_asym0.2.csv

   ​            -- train_news16_asym0.4.csv

   ​            -- train_news16_asym0.6.csv

   ​            -- train_news16_sym0.2.csv

   ​            -- train_news16_sym0.4.csv

   ​            -- train_news16_sym0.6.csv

   ​            -- test16.csv

2. Modify the config  file under ./config

   The file contains the following parameters: **database **, **dataset**, **n_classes** (number of classes), **pretrainedmodel** (pretrained model directory), **dict_len** (vocabulary size), **trans_n_head** (number of attention heads in the model), **trans_n_layer** (number of layers in the model), **d_model** (feature vector dimension), and **logging** (log information).

   For example, the configuration information for the 20 Newsgroups dataset is already available.

   --\

      -- config

   ​        -- 20newsgroup.cfg

3. Run the baseline model. You can find the paper and repository address for the baseline model through the link provided above. We have provided simple implementations of some methods. `train_bert.py` is the implementation for the BERT model.

4. Run our proposed method. `train.py` contains the specific implementation of our proposed method. Use the config file and dataset directory as input parameters for execution. Some parameters in this file need to be set.Test 

   **config**: Configuration file directory,
   **log_prefix**: Log directory prefix,
   **log_freq**: Log frequency,
   **threshold**: Mahalanobis distance threshold,
   **net**: Network name,
   **stage1**: Pre-warmup epochs.

5. Test results. During the training process, testing will be performed at regular intervals, and the results will be recorded in the log file.

## Experiments

The results compared with the baseline model are as follows:

![](/Figure2.png)

![Figure3](/Figure3.png)



The sensitivity of the hyperparameters is as follows:

![](/Fig3(a).png)

![Fig3(b)](/Fig3(b).png)

The impact of confidence is as follows:

![](/Figure4.png)

