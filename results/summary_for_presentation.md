# 発表用サマリ：Fashion-MNISTロバスト性実験

## 実験概要

Fashion-MNISTの衣類画像分類を題材に、軽量なMLP/CNNを学習し、通常のテスト画像だけでなく、ノイズ・回転・遮蔽を加えたテスト画像でも評価する。

目的は、clean test accuracyだけでは分からないモデルのロバスト性を分析すること。

## 使用モデル

- MLP: 比較用ベースライン
- SmallCNN: 軽量CNN
- SmallCNN_Aug: SmallCNNと同じ構造だが、学習時のみデータ拡張を使用

## 評価条件

- clean: 通常のtest data
- noise: Gaussian noiseを追加
- rotation: 15度回転
- occlusion: 中央付近を8x8で遮蔽

## 結果表の見方

`results/metrics.csv` には、各モデル・各テスト条件ごとのAccuracyとMacro F1-scoreが保存される。

見るべき点:

- cleanで最も高いモデルはどれか
- corrupted testで性能が大きく落ちるモデルはどれか
- cleanからnoise/rotation/occlusionへのAccuracy低下量はどれくらいか
- SmallCNN_AugはSmallCNNより低下量を抑えられているか
- AccuracyとMacro F1-scoreの傾向は一致しているか

## 期待される考察

- clean testの精度だけでは、モデルの頑健性は判断できない。
- ノイズ・回転・遮蔽により性能が低下するかを調べることで、モデルの弱点を分析できる。
- データ拡張ありCNNは、通常精度だけでなく、加工画像に対する性能低下を抑える可能性がある。
- 回転に弱い場合は、学習時のRandomRotationが有効に働く可能性がある。
- 遮蔽に弱い場合は、画像の一部に強く依存して分類している可能性がある。
- T-shirt/top、Shirt、Coat、Pulloverなど、形が似ているクラスは混同しやすい可能性がある。
- confusion matrixを使うと、どのクラス同士が混ざりやすいかを具体的に説明できる。

## グループメンバーの役割分担例

- Member A: データ前処理・加工テストデータ作成
- Member B: モデル実装・学習
- Member C: 評価指標・混同行列分析
- Member D: 可視化・スライド作成・発表

## スライド構成例

1. 研究背景とResearch Question
2. Fashion-MNISTと評価条件の説明
3. 実装したモデルの説明
4. clean testの結果
5. corrupted testの結果
6. Accuracy低下量の比較
7. confusion matrixによる誤分類分析
8. 考察とまとめ

