# NAF

NAFを用いた多チャンネル音場推定  

[NAF](https://github.com/aluo-x/Learning_Neural_Acoustic_Fields)をベースとして、多チャンネル音場への拡張を行いました。  

---

## 動作環境

主要なソフトウェア及びそのバージョンは以下の通りになります。

- Python 3.9
- CUDA 12.6
- PyTorch 2.7
- torchaudio 2.7
- NumPy 2.0
- SciPy 1.13
- Matplotlib 3.9
- Pyroomacoustics 0.8
- librosa 0.11
- scikit-learn 1.6
- Optuna 4.5

詳細な依存関係および正確なバージョンについては、  
[`requirements.txt`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/NAF/requirements.txt) を参照してください。

---

## リポジトリ構成

```text
NAF/
├ README.md                     ドキュメント
├ requirements.txt              依存関係
├ config_files/                 各種設定ファイル（YAML）
│  ├ preprocess_config.yml        前処理の設定ファイル
│  ├ optuna_config.yml            ハイパラチューニングの設定ファイル
│  ├ train_config.yml             学習の設定ファイル
│  └ inference_config.yml              推論の設定ファイル
├ preprocess/                   前処理
│  ├ split_train_val.py           学習データと検証データの分割
│  └ preprocess.py                前処理（STFTなど）
├ model/                        NAFモデル
│  ├ modules.py                   モデル内部で使用するモジュール
│  └ networks.py                  NAFのニューラルネットワーク
├ sound_loader.py               データローダ
├ optuna_tuning.py              Optunaによるハイパラチューニング
├ train.py                      学習
└ inference.py                       推論
```

---

## 入出力

### 前処理

#### 入力

| 入力 | 説明 |
|---|---|
| データセットディレクトリ | 多チャンネルインパルス応答の波形 |
| 前処理設定ファイル | 前処理（STFTなど）の条件 |
| 前処理結果出力先ディレクトリ | 前処理結果の出力先 |

#### データセットディレクトリ

`dataset_dir`に、以下のディレクトリ構成で多チャンネルインパルス応答の波形データを用意します。  
[実データ](https://github.com/KMASAHIRO/multichannel-soundfields/tree/main/real_data)及び[AcoustiX](https://github.com/KMASAHIRO/multichannel-soundfields/tree/main/AcoustiX#output)や[Pyroomacoustics](https://github.com/KMASAHIRO/multichannel-soundfields/tree/main/Pyroomacoustics#出力)によるシミュレーションデータを使う場合は、各出力先ディレクトリをそのまま使用してください。

```text
dataset_dir/
├ tx_0/                        # 送信機のインデックス（0,1,2,...）
│  ├ rx_0.npz                  # 受信機のインデックス（0,1,2,...）
│  ├ rx_1.npz
│  ├ ...
├ tx_1/
│  ├ rx_0.npz
│  ├ ...
├ ...
```

各`rx_*.npz`の内容は以下の通りです。

| key | dtype | shape | 内容 |
|---|---|---|---|
| ir | ndarray | (N_ch, ir_len) | 多チャンネルインパルス応答 |
| position_rx | ndarray | (N_ch, 3) | 各チャンネルの受信機位置 |
| position_tx | ndarray | (3,) | 送信機位置 |

#### 前処理設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は[`preprocess_config.yml`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics/config_files/preprocess_config.yml)を参照してください。  
NAFのモデルタイプにおける`NAF+`は、`NAF`を多チャンネルデータ用に拡張したモデルです。

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| fs | 16000 | サンプリング周波数 [Hz] |
| n_fft | 512 | STFTの窓幅 |
| hop_size | 128 | STFTのhop size |
| window | hann | 窓関数 |
| log_eps | 1e-3 | log計算のエラーを防ぐための微小値 |
| mag_std_eps | 0.1 | 振幅の標準偏差計算時の微小値 |
| model_type | NAF+ | NAFのモデルタイプ |

---

#### 出力

`preprocessed_data_dir`に、以下のディレクトリ構成で前処理結果を出力します。

```text
preprocessed_data_dir/
├ preprocess_config.yml
├ train_val_split.pkl
├ positions.h5
├ magnitudes.h5
└ phases.h5
```

---

### ハイパラチューニング（Optuna）

#### 入力

| 入力 | 説明 |
|---|---|
| 前処理結果出力先ディレクトリ | 前処理済みデータ |
| ハイパラチューニング設定ファイル | ハイパラチューニングの条件 |
| ハイパラチューニング結果出力先ディレクトリ | ハイパラチューニング結果の出力先 |

#### ハイパラチューニング設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は[`optuna_config.yml`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics/config_files/optuna_config.yml)を参照してください。

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| study.study_name | "naf_plus_optuna" | Optunaのstudy名 |
| study.direction | "minimize" | 最適化方向 |
| study.n_trials | 50 | 試行回数 |
| study.sampler | "tpe" | ハイパーパラメータ探索のサンプラ |
| doa_metric.algorithm | NormMUSIC | DoAアルゴリズム |
| doa_metric.fallback_value | 999.0 | 評価値が取得できない場合の代替値 |
| training_fixed.* | — | 固定するハイパーパラメータ（詳細は[学習設定ファイル](#学習設定ファイル)） |
| search_space.* | — | チューニングするハイパーパラメータ（詳細は[学習設定ファイル](#学習設定ファイル)） |
| search_space.*.type | — | 探索タイプの指定（`int` / `float` / `categorical`） |
| search_space.*.low | — | 探索範囲の下限（typeが `int` / `float` の場合） |
| search_space.*.high | — | 探索範囲の上限（typeが `int` / `float` の場合） |
| search_space.*.log | — | 対数スケールで探索するか（typeが `float` の場合） |
| search_space.*.choices | — | 探索候補（typeが `categorical` の場合） |

---

#### 出力

`optuna_output_dir`に、以下のディレクトリ構成でハイパラチューニング結果を出力します。

```text
optuna_output_dir/
└─ <study_name>/
   ├─ optuna_config.yml
   ├─ <study_name>.db                  # Optuna study DB (SQLite)
   └─ trials/
      ├─ trial0001/
      │  ├─ chkpt/
      │  │  ├─ best0001.ckpt
      │  │  ├─ best0002.ckpt
      │  │  └─ ...
      │  ├─ loss/
      │  │  ├─ epoch0001.npz
      │  │  ├─ epoch0002.npz
      │  │  └─ ...
      │  └─ val_results/
      │     ├─ epoch0001.npz
      │     ├─ epoch0002.npz
      │     └─ ...
      ├─ trial0002/
      │  └─ ...
      └─ ...
```

---

### 学習

#### 入力

| 入力 | 説明 |
|---|---|
| 学習設定ファイル | NAFの学習条件 |
| 前処理結果出力先ディレクトリ | 前処理済みデータ |
| 学習結果出力先ディレクトリ | 学習結果の出力先 |

#### 学習設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は[`train_config.yml`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics/config_files/train_config.yml)を参照してください。

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| doa_metric.algorithm | NormMUSIC | DoA評価に用いるアルゴリズム |
| doa_metric.fallback_value | 999.0 | DoA評価が失敗・未計算時に用いる代替値 |
| gpus | 4 | 学習に使用するGPU数 |
| model_type | `NAF+` | NAFのモデルタイプ（`NAF` / `NAF+`） |
| dir_ch | 8 | チャンネル数 |
| epochs | 200 | 学習エポック数 |
| resume | 0 | チェックポイントから再開するかどうか |
| batch_size | 20 | 学習時のバッチサイズ |
| mag_alpha | 1.0 | 振幅（magnitude）損失の重み |
| phase_alpha | 1.0 | 位相（phase）損失の重み |
| lr_init | 1.0e-3 | 初期学習率 |
| lr_decay | 1.0e-1 | 学習率減衰係数 |
| weight_decay_grid | 1.0e-2 | グリッド系パラメータに対するweight decay |
| weight_decay_main | 0.0 | 非グリッド系パラメータに対するweight decay |
| reg_eps | 0.05 | 学習時に位置に加える微小値 |
| pixel_count | 2000 | 1 forward あたりにサンプリングするピクセル数 |
| layers | 8 | MLPの総レイヤ数 |
| layers_residual | 1 | Residualブロック数 |
| features | 256 | 各レイヤの隠れ特徴量次元 |
| grid_features | 64 | グリッド特徴量の次元数 |
| activation_func_name | default | 活性化関数 |
| grid_gap | 0.25 | 初期グリッド点の間隔 |
| bandwith_init | 0.25 | 各グリッド点が持つRBFカーネルの初期的な広がり |
| position_float | 0.1 | データに合わせてグリッド位置を微調整できる範囲 |
| min_bandwidth | 0.1 | カーネル幅の下限 |
| max_bandwidth | 0.5 | カーネル幅の上限 |
| embed_xyz_num_freqs | 10 | 位置座標の埋め込み（positional encoding）における周波数の数 |
| embed_xyz_max_freq | 7 | 位置座標の埋め込みにおける最大周波数 |
| embed_time_num_freqs | 10 | 時刻の埋め込みにおける周波数の数 |
| embed_time_max_freq | 10 | 時刻の埋め込みにおける最大周波数 |
| embed_freq_num_freqs | 10 | 周波数の埋め込みにおける周波数の数 |
| embed_freq_max_freq | 10 | 周波数の埋め込みにおける最大周波数 |
| save_best_k | 10 | 評価指標が良い上位k個のモデルを保存 |

---

#### 出力

`train_output_dir`に、以下のディレクトリ構成でハイパラチューニング結果を出力します。

```text
train_output_dir/
├─ train_config.yml                  # 学習に使用した設定ファイル
├─ chkpt/
│  ├─ best0001.ckpt
│  ├─ best0002.ckpt
│  └─ ...
├─ loss/
│  ├─ epoch0001.npz
│  ├─ epoch0002.npz
│  └─ ...
└─ val_results/
   ├─ epoch0001.npz
   ├─ epoch0002.npz
   └─ ...
```


---

### 推論

#### 入力

| 入力 | 説明 |
|---|---|
| 学習済みモデルの重み | 学習済みチェックポイント |
| 推論設定ファイル | NAFの推論条件 |
| 送信機データファイル | 送信機（スピーカー）の位置 |
| 受信機データファイル | 受信機（マイクロフォンアレイ）の位置 |
| 推論結果出力先ディレクトリ | 推論結果の出力先 |

#### 推論設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は[`inference_config.yml`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics/config_files/inference_config.yml)を参照してください。

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| model_type | `NAF+` | NAFのモデルタイプ（`NAF` / `NAF+`） |
| dir_ch | 8 | チャンネル数 |
| layers | 8 | MLPの総レイヤ数 |
| layers_residual | 1 | Residualブロック数 |
| features | 256 | 各レイヤの隠れ特徴量次元 |
| grid_features | 64 | グリッド特徴量の次元数 |
| batch_norm | none | BatchNorm の使用設定 |
| activation_func_name | default | 活性化関数の種類 |
| grid_gap | 0.25 | 初期グリッド点の間隔 |
| bandwith_init | 0.25 | RBFカーネルの初期幅 |
| position_float | 0.1 | グリッド位置を微調整できる範囲 |
| min_bandwidth | 0.1 | カーネル幅の下限 |
| max_bandwidth | 0.5 | カーネル幅の上限 |
| pixel_count | 2000 | 1 forward あたりに処理する (freq×time) 点数 |
| embed_xyz_num_freqs | 10 | 位置座標埋め込みの周波数数 |
| embed_xyz_max_freq | 7 | 位置座標埋め込みの最大周波数 |
| embed_time_num_freqs | 10 | 時刻埋め込みの周波数数 |
| embed_time_max_freq | 10 | 時刻埋め込みの最大周波数 |
| embed_freq_num_freqs | 10 | 周波数埋め込みの周波数数 |
| embed_freq_max_freq | 10 | 周波数埋め込みの最大周波数 |

#### 送信機データファイル

送信機（スピーカー）の位置を定義したJSONファイルを用意します。  
`N_tx`は送信機配置パターンの総数です。

| key | 型 | shape | 内容 |
|---|---|---|---|
| positions | list | (N_tx, 3) | 送信機位置 [x, y, z] |

#### 受信機データファイル

- `model_type`が`NAF`の場合

受信機の位置を定義したJSONファイルを用意します。  
`N_rx`は受信機（マイクロフォンアレイ）の配置数を表し、各受信機はN_chチャンネルで構成されます。ただし、アレイの中心と送信機位置が重なる受信機は除外してシミュレーションを行います。

| key | 型 | shape | 内容 |
|---|---|---|---|
| positions | list | (N_rx, N_ch, 3) | 受信機位置 [x, y, z] |

- `model_type`が`NAF+`の場合

受信機（マイクロフォンアレイ）の中心位置を定義したJSONファイルを用意します。  
`N_rx`は受信機（マイクロフォンアレイ）の配置数を表し、各受信機はN_chチャンネルで構成されます。ただし、アレイの中心と送信機位置が重なる受信機は除外してシミュレーションを行います。  
`NAF+`の場合は、マイクロフォンアレイの中心位置をJSONファイルに記録します。

| key | 型 | shape | 内容 |
|---|---|---|---|
| positions | list | (N_rx, 3) | 受信機位置 [x, y, z] |

---

#### 出力

`inference_output_dir`に、以下のディレクトリ構成でハイパラチューニング結果を出力します。

```text
inference_output_dir/
├─ inference_config.yml                  # 推論に使用した設定ファイル
└─ results/
   ├─ epoch0001.npz
   ├─ epoch0002.npz
   └─ ...
```


---

## 実行手順

1. リポジトリのクローン

```
git clone https://github.com/KMASAHIRO/multichannel-soundfields  
cd multichannel-soundfields/NAF
```

2. 依存関係のインストール

```
pip install -r requirements.txt
```

3. 学習データと検証データの分割

```
python preprocess/split_train_val.py \
  --dataset_dir dataset_dir \
  --output_dir preprocessed_data_dir
```

4. 前処理

```
python preprocess/preprocess.py \
  --config config_files/preprocess_config.yml \
  --dataset_dir dataset_dir \
  --output_dir preprocessed_data_dir
```

5. Optuna によるハイパラメータ探索

```
python optuna_tuning.py \
  --config config_files/optuna_config.yml \
  --data_dir preprocessed_data_dir \
  --output_dir optuna_output_dir
```

6. 学習

```
python train.py \
  --config config_files/train_config.yml \
  --data_dir preprocessed_data_dir \
  --output_dir train_output_dir
```

7. 推論

```
python inference.py \
  --chkpt checkpoint.chkpt \
  --config config_files/test_config.yml \
  --speaker config_files/speaker_data.json \
  --receiver config_files/receiver_data_NAF_plus.json \
  --output_dir inference_output_dir
```
