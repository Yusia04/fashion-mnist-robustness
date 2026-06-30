# 軽量CNNによるFashion-MNIST画像分類：データ拡張がロバスト性に与える影響の分析

## テーマ

**Lightweight CNNs for Fashion Image Classification: Analyzing the Effect of Data Augmentation on Robustness**

Fashion-MNISTを用いて、軽量なMLP/CNNの分類性能を比較し、さらにノイズ・回転・遮蔽を加えたテスト画像に対するロバスト性を評価します。

## Research Question

軽量CNNは通常のテスト画像では高精度を出せるが、ノイズ・回転・一部遮蔽などの小さな分布変化に対しても頑健なのか？
また、データ拡張は軽量CNNのロバスト性を改善できるのか？

## 使用データセット

- `torchvision.datasets.FashionMNIST`
- 28x28ピクセルのグレースケール衣類画像
- 10クラス分類
- デフォルト設定:
  - train: 10,000枚
  - validation: 2,000枚
  - test: 10,000枚

## 実験の目的

通常のclean test accuracyだけでなく、加工したテスト画像での性能低下を調べることで、モデルがどのような画像変化に弱いかを分析します。

特に、以下を比較します。

- MLPとSmall CNNの性能差
- データ拡張なしCNNとデータ拡張ありCNNの差
- clean testとcorrupted testの差
- Accuracy低下量
- confusion matrixによるクラス間の混同傾向

## 実装したモデル

### Model 1: MLP

- Flatten
- Linear
- ReLU
- Dropout
- Linear
- ReLU
- Linear

### Model 2: SmallCNN

- Conv2d
- ReLU
- MaxPool2d
- Conv2d
- ReLU
- MaxPool2d
- Flatten
- Linear
- ReLU
- Dropout
- Linear

### Model 3: SmallCNN_Aug

モデル構造はSmallCNNと同じです。
学習データにのみ、以下の軽いデータ拡張を適用します。

- `RandomRotation(10 degrees)`
- `RandomAffine`
- `RandomErasing`
- `Normalize`

validationとtestにはデータ拡張を適用しません。

## 評価条件

以下の4種類のtest dataで評価します。

- `clean`: 通常のFashion-MNIST test data
- `noise`: 軽いGaussian noiseを追加
- `rotation`: 15度回転
- `occlusion`: 中央付近を8x8の黒い四角で遮蔽

評価指標:

- Accuracy
- Macro F1-score
- Confusion matrix

## セットアップ

Python 3.10以上を想定しています。

```bash
cd /home/taki/fashion_mnist_robustness
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

初回実行時にFashion-MNISTが `data/` に自動ダウンロードされます。

## 実行方法

基本実行:

```bash
python run_experiment.py --epochs 5 --train-size 10000 --val-size 2000 --batch-size 128
```

短時間の動作確認:

```bash
python run_experiment.py --epochs 1 --train-size 1000 --val-size 200 --test-size 1000 --batch-size 128
```

特定モデルだけ実行:

```bash
python run_experiment.py --models SmallCNN SmallCNN_Aug --epochs 5
```

GPUが使える場合は自動でCUDAを使います。CPUで固定したい場合:

```bash
python run_experiment.py --device cpu
```

## 出力される結果

```text
results/
├── metrics.csv
├── training_history.csv
├── summary_for_presentation.md
└── figures/
    ├── accuracy_by_model_and_condition.png
    ├── accuracy_drop_from_clean.png
    ├── training_curves_MLP.png
    ├── training_curves_SmallCNN.png
    ├── training_curves_SmallCNN_Aug.png
    ├── confusion_matrix_SmallCNN_clean.png
    ├── confusion_matrix_SmallCNN_Aug_clean.png
    ├── corrupted_test_samples.png
    ├── misclassified_examples_SmallCNN.png
    └── misclassified_examples_SmallCNN_Aug.png
```

`metrics.csv` の例:

| model | test_condition | accuracy | macro_f1 |
|---|---|---:|---:|
| MLP | clean | ... | ... |
| MLP | noise | ... | ... |
| SmallCNN | clean | ... | ... |
| SmallCNN | noise | ... | ... |
| SmallCNN_Aug | clean | ... | ... |

## 発表で使える考察ポイント

- clean testの精度だけでは、モデルの頑健性は判断できない。
- ノイズ・回転・遮蔽により性能がどの程度低下するかを見ると、モデルの弱点を分析できる。
- MLPは画像の局所構造を明示的に使わないため、CNNより画像変化に弱い可能性がある。
- SmallCNNはclean testでは高精度になりやすいが、回転や遮蔽で大きく性能が落ちる可能性がある。
- SmallCNN_Augは、通常精度だけでなく、加工画像に対する性能低下を抑える可能性がある。
- ただし、データ拡張が強すぎるとclean accuracyが少し下がる場合もある。
- T-shirt/top、Shirt、Coat、Pulloverなど、形が似ているクラスは混同しやすい可能性がある。
- confusion matrixを見ることで、単なる正解率だけでは分からない「どのクラスをどのクラスと間違えたか」を説明できる。

## ファイル構成

```text
fashion_mnist_robustness/
├── README.md
├── requirements.txt
├── run_experiment.py
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── transforms.py
│   ├── visualize.py
│   └── utils.py
├── results/
│   ├── summary_for_presentation.md
│   └── figures/
└── data/
```

