# Pyroomacoustics

[Pyroomacoustics](https://pyroomacoustics.readthedocs.io/en/pypi-release/)を用いた多チャンネル音響シミュレーション

---

## 動作環境

主要なソフトウェア及びそのバージョンは以下の通りになります。

- Python 3.9
- NumPy 1.23
- Pyroomacoustics 0.7

詳細な依存関係および正確なバージョンについては、[`requirements.txt`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics/requirements.txt) を参照してください。

---

## リポジトリ構成

```text
Pyroomacoustics/
├ README.md                       ドキュメント
├ requirements.txt                依存関係
├ make_speaker_receiver_data.py   受信機・送信機データファイル生成
├ simu_input/                     シミュレーション入力データ
│  ├ config.yml                     シミュレーション設定ファイル
│  ├ speaker_data.json              送信機データファイル
│  └ receiver_data.json             受信機データファイル
└ simulation.py                   多チャンネル音響のシミュレーション
```

---

## シミュレーション手順

1. リポジトリのクローン

```
git clone https://github.com/KMASAHIRO/multichannel-soundfields
cd multichannel-soundfields/Pyroomacoustics
```

2. 依存関係のインストール

```
pip install -r requirements.txt
```

3. シミュレーションの実行

```
python simulation.py \
  --config simu_input/config.yml \
  --speaker simu_input/speaker_data.json \
  --receiver simu_input/receiver_data.json \
  --output_dir outputs
```

---

## 入出力

### 入力

以下の入力を使ってシミュレーションを実行します。すべての入力が必要です。

| 入力 | 説明 |
|---|---|
| [シミュレーション設定ファイル](#シミュレーション設定ファイル) | シミュレーション条件の設定 |
| [送信機データファイル](#送信機データファイル) | 送信機（スピーカー）の位置 |
| [受信機データファイル](#受信機データファイル) | 受信機（マイクロフォンアレイ）の位置 |
| [出力先ディレクトリ](#出力先ディレクトリ) | シミュレーション結果の保存先 |

---

#### シミュレーション設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は[`config.yml`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics/simu_input/config.yml)を参照してください。

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| room_dim | [6.110, 8.807, 2.7] | 部屋寸法 [m]（x, y, z） |
| max_order | 10 | 鏡像法（Image Source Method）における最大反射回数（反射次数） |
| e_absorption | 0.0055 | 壁面の吸音率 |
| sampling_rate | 16000 | サンプリング周波数 [Hz] |
| ir_len | 1600 | 保存するインパルス応答長（サンプル数） |

---

#### 送信機データファイル

送信機（スピーカー）の位置を定義したJSONファイルを用意します。  
`N_tx`は送信機配置パターンの総数です。各シミュレーションでは送信機は1台のみ配置します。

| key | 型 | shape | 内容 |
|---|---|---|---|
| positions | list | (N_tx, 3) | 送信機位置 [x, y, z] |

[論文](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/Challenge-068/2025_03/_article/-char/ja)で使用した、下図のようなグリッド上の橙点に配置されたスピーカーに対応するファイルは、[`speaker_data.json`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics/simu_input/speaker_data.json)を参照してください。

<img width="500" height="426" alt="room_dim" src="https://github.com/user-attachments/assets/049b55de-3061-4ea8-bdd7-519d04ef4a4a" />


---

#### 受信機データファイル

受信機（マイクロフォンアレイ）の位置を定義したJSONファイルを用意します。  
`N_rx`は受信機（マイクロフォンアレイ）の配置数を表し、各受信機はN_chチャンネルで構成されます。ただし、アレイの中心と送信機位置が重なる受信機は除外してシミュレーションを行います。

| key | 型 | shape | 内容 |
|---|---|---|---|
| positions | list | (N_rx, N_ch, 3) | 受信機位置 [x, y, z] |

[論文](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/Challenge-068/2025_03/_article/-char/ja)で使用した、下図のようなグリッド上に配置された8ch円形マイクロフォンアレイに対応するファイルは、[`receiver_data.json`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics/simu_input/receiver_data.json)を参照してください。

<img width="500" height="426" alt="room_dim" src="https://github.com/user-attachments/assets/049b55de-3061-4ea8-bdd7-519d04ef4a4a" />

---

### 出力

入力時に指定した出力先ディレクトリ`output_dir`に、以下の構成で出力します。

```text
output_dir/
├ config.yml
├ speaker_data.json
├ receiver_data.json
├ tx_0/                        # 送信機のインデックス（0,1,2,...）
│  ├ rx_0.npz                  # 受信機のインデックス（0,1,2,...）
│  ├ rx_1.npz
│  ├ ...
├ tx_1/
│  ├ rx_0.npz
│  ├ ...
├ ...
```

シミュレーション条件を保存するため、入力に使用した`config.yml`、`speaker_data.json`、`receiver_data.json`をコピーして出力先に保存します。
各npzファイルの内容は以下のようになります。

| key            | dtype   | shape | 内容                 |
| -------------- | ------- | ----- | ------------------ |
| ir             | ndarray | (N_ch, ir_len)  | インパルス応答の波形      |
| position_rx    | ndarray | (N_ch, 3)  | 受信機位置 [x, y, z]  |
| position_tx    | ndarray | (3,)  | 送信機位置 [x, y, z]    |
