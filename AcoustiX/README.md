# AcoustiX

AcoustiXを用いた多チャンネル音響シミュレーション  

公式の[AcoustiX](https://github.com/penn-waves-lab/AcoustiX)実装をベースとして、軽微なバグの修正及び多チャンネル音響シミュレーションコードの追加を行いました。  

AcoutiXは、NVIDIAの電波用レイトレーシングシミュレータ [Sionna ray tracing](https://github.com/NVlabs/sionna)を音響用に拡張したシミュレータです。

---

## 動作環境

主要なソフトウェア及びそのバージョンは以下の通りになります。

- Python 3.9
- TensorFlow 2.13
- NumPy 1.24
- SciPy 1.13
- Matplotlib 3.9
- Mitsuba 3.4

詳細な依存関係および正確なバージョンについては、
`requirements.txt` を参照してください。

### インストール

以下のコマンドで依存関係をインストールします。

pip install -r requirements.txt

---

## リポジトリ構成

```text
AcoustiX/
├ README.md                   AcoustiXを用いた多チャンネル音響シミュレーションの手順
├ LICENSE                     ライセンスファイル（MIT License）
├ requirements.txt            動作環境情報
├ acoustic_absorptions.json   材料ごとの吸音率設定ファイル
├ simu_input/                 シミュレーション入力データ
├ simu_utils.py               シミュレーション用の共通関数
├ simulation.py               多チャンネル音響のシミュレーション
├ pattern.py                  音源・マイクの指向性パターン
├ check_scene.ipynb           シミュレーション環境（シーン）の可視化
└ sionna/                     Sionna ray tracingモジュール
```

---

## 入出力

### 入力

以下すべての入力が必要です。

| 入力 | 説明 |
|---|---|
| シミュレーション設定ファイル | レイトレーシングの条件や指向性などの設定 |
| シーンファイル | シミュレーション環境（シーン）の情報 |
| 送信機データ | 送信機（スピーカー）の位置・向き |
| 受信機データ | 受信点（マイク）の位置・向き |
| 出力先ディレクトリ | シミュレーション結果の保存先 |

---

#### シミュレーション設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は`basic_config.yml`を参照してください。

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| max_depth | 10 | 音線の最大反射回数 |
| num_samples | 50000 | 放射する音線数 |
| los | true | 直接音を計算するか |
| reflection | true | 反射音を考慮するか |
| diffraction | false | 回折効果を考慮するか |
| scattering | true | 散乱効果を考慮するか |
| scat_prob | 0.00001 | 音線が散乱する確率 |
| attn | 0.001 | 減衰係数 |
| fs | 48000 | サンプリング周波数 [Hz] |
| ir_len | 4800 | インパルス応答長（サンプル数） |
| speed | 343.8 | 音速 [m/s] |
| noise | 0.001 | 付加ノイズレベル |
| tx_pattern | uniform | 音源指向性 |
| rx_pattern | uniform | マイク指向性 |

---

#### シーンファイル

シミュレーション環境（シーン）を表すXMLファイル、plyファイルをBlenderで作成する必要があります。

[論文](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/Challenge-068/2025_03/_article/-char/ja)で使用したシーンファイルは、[Google Drive](https://drive.google.com/drive/folders/1h1R4gZKTwJghD0qsZyB5vbckLi2LphX3)からダウンロードできます。 

Drive内のデータをすべてダウンロードし、そのままのディレクトリ構成でシミュレーション実行環境に配置してください。  

シーンを自作したい場合は、  
[自分でシミュレーション環境を構築する場合](#自分でシミュレーション環境を構築する場合)  
を参照してください。

---

#### 送信機データ

送信機（スピーカー）の位置および向きを定義したJSONファイルを用意します。  
シミュレーション設定で`tx_pattern`を`uniform`（指向性なし）にした場合は、向きによる影響はありません。  
[論文](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/Challenge-068/2025_03/_article/-char/ja)で使用したファイルは、`speaker_data.json`を参照してください。

| key | 型 | shape | 内容 |
|---|---|---|---|
| positions | list | (N_tx, 3) | 送信機位置 [x, y, z] |
| orientations | list | (N_tx, 3) | 送信機の向き [x, y, z] |

---

#### 受信機データ

受信機（マイク）の位置および向きを定義したJSONファイルを用意します。  
シミュレーション設定で`rx_pattern`を`uniform`（指向性なし）にした場合は、向きによる影響はありません。  
[論文](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/Challenge-068/2025_03/_article/-char/ja)で使用したファイルは、`receiver_data.json`を参照してください。

| key | 型 | shape | 内容 |
|---|---|---|---|
| positions | list | (N_rx, 3) | 受信点中心位置 [x, y, z] |
| orientations | list | (N_rx, 3) | 受信点の向き [x, y, z] |

---

### 出力

```text
output_dir/
├ config.yml
├ speaker_data.json
├ receiver_data.json
├ tx_0/                        # 送信機インデックス（0,1,2,...）
│  ├ rx_0/                     # 受信点インデックス（0,1,2,...）
│  │  ├ ir_000000.npz          # チャンネル0
│  │  ├ ir_000001.npz          # チャンネル1
│  │  ├ ...
│  ├ rx_1/
│  │  ├ ir_000000.npz
│  │  ├ ...
│  ├ ...
├ tx_1/
│  └ rx_0/ ...
├ ...
```

シミュレーション設定の記録のため、入力に使用した`config.yml`、`speaker_data.json`、`receiver_data.json`をコピーして出力先に保存します。
各npzファイルの内容は以下のようになります。

| key            | dtype   | shape | 内容                 |
| -------------- | ------- | ----- | ------------------ |
| ir             | ndarray | (ir_len,)  | インパルス応答の波形      |
| position_rx    | ndarray | (3,)  | 受信機位置 [x, y, z]  |
| position_tx    | ndarray | (3,)  | 送信機位置 [x, y, z]    |
| orientation_rx | ndarray | (3,)  | 受信機の向き [x, y, z] |
| orientation_tx | ndarray | (3,)  | 送信機の向き [x, y, z]   |


---

## 使い方

### シミュレーションの実行

以下のコマンドでインパルス応答のシミュレーションを実行します。

python simulation.py

シミュレーション条件は `simu_config/` 以下の設定ファイルで指定します。


---

## 自分でシミュレーション環境を構築する場合

独自の音響環境を構築したい場合は、
Sionna の公式チュートリアルを参照してください。

Create your own scene using Blender (Sionna Official)  
https://www.youtube.com/watch?v=7xHLDxUaQ7c

---

## ライセンス

本リポジトリは MIT License の条件に従って公開されています。

LICENSE ファイルを参照してください。
