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
  --config config_files/test_config.yml \
  --chkpt checkpoint.chkpt \
  --speaker config_files/speaker_data.json \
  --receiver config_files/receiver_data_NAF_plus.json \
  --output_dir inference_output_dir
```

---

## 入出力

### 前処理　入力

| 入力 | 説明 |
|---|---|
| [前処理設定ファイル](#前処理設定ファイル) | 前処理（STFTなど）の条件 |
| [データセットディレクトリ](#データセットディレクトリ) | 多チャンネルインパルス応答の波形 |
| [前処理結果出力先ディレクトリ](#前処理出力) | 前処理結果の出力先 |

#### データセットディレクトリ

`dataset_dir`に、以下のディレクトリ構成で多チャンネルインパルス応答の波形データを用意します。  
[実データ](https://github.com/KMASAHIRO/multichannel-soundfields/tree/main/real_data)及び[AcoustiX](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AcoustiX#出力)や[Pyroomacoustics](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics#出力)によるシミュレーションデータを使う場合は、各「出力先ディレクトリ」をそのまま使用してください。

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

各`rx_*.npz`の内容は以下の通りです。N_chは受信機（マイクロフォンアレイ）のチャンネル数です。

| key | dtype | shape | 内容 |
|---|---|---|---|
| ir | float32 | (N_ch, ir_len) | 多チャンネルインパルス応答 |
| position_rx | float32 | (N_ch, 3) | 各チャンネルの受信機位置 [x, y, z] |
| position_tx | float32 | (3,) | 送信機位置 [x, y, z] |

位置座標はx、yのみ使用するので、zを除いた以下の内容でも構いません。

| key | dtype | shape | 内容 |
|---|---|---|---|
| ir | float32 | (N_ch, ir_len) | 多チャンネルインパルス応答 |
| position_rx | float32 | (N_ch, 2) | 各チャンネルの受信機位置 [x, y] |
| position_tx | float32 | (2,) | 送信機位置 [x, y] |


#### 前処理設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は[`preprocess_config.yml`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics/config_files/preprocess_config.yml)を参照してください。  
NAFのモデルタイプにおける`NAF+`は、`NAF`を多チャンネルデータ用に拡張したモデルです。

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| fs | 16000 | サンプリング周波数 [Hz] |
| n_fft | 512 | STFTの窓幅 |
| hop_size | 128 | STFTのhop size |
| window | hann | 窓関数（`hamming` / `gaussian` / `boxcar` など（[詳細](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html)）） |
| log_eps | 1e-3 | log計算のエラーを防ぐための微小値 |
| mag_std_eps | 0.1 | 振幅の標準偏差計算時の微小値 |
| model_type | NAF+ | モデルタイプ（`NAF` / `NAF+`） |

---

### 前処理　出力

`preprocessed_data_dir`に、以下のディレクトリ構成で前処理結果を出力します。

```text
preprocessed_data_dir/
├ preprocess_config.yml         # 入力の設定ファイルのコピー
├ train_val_split.pkl           # 学習データと検証データの分割
├ positions.h5                  # 送信機・受信機の位置
├ magnitudes.h5                 # 振幅
└ phases.h5                     # 位相
```

---

### ハイパラチューニング　入力

| 入力 | 説明 |
|---|---|
| [ハイパラチューニング設定ファイル](#ハイパラチューニング設定ファイル) | ハイパラチューニングの条件 |
| [前処理結果出力先ディレクトリ](#前処理出力) | 前処理済みデータ |
| [ハイパラチューニング結果出力先ディレクトリ](#ハイパラチューニング出力) | ハイパラチューニング結果の出力先 |

#### ハイパラチューニング設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は[`optuna_config.yml`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics/config_files/optuna_config.yml)を参照してください。

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| study.study_name | naf_plus_optuna | Optunaのstudy名 |
| study.direction | minimize | 最適化方向（`minimize` / `maximize`） |
| study.n_trials | 50 | 試行回数 |
| doa_metric.algorithm | NormMUSIC | DoAアルゴリズム（`MUSIC` / `SRP` など（[詳細](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.doa.html)）） |
| doa_metric.fallback_value | 999.0 | 評価値が取得できない場合の代替値 |
| setting.gpus | 4 | 学習に使用するGPU数 |
| setting.model_type | NAF+ | NAFのモデルタイプ（`NAF` / `NAF+`） |
| setting.dir_ch | 8 | チャンネル数 |
| setting.epochs | 200 | 学習エポック数 |
| setting.resume | False | チェックポイントから再開するかどうか |
| setting.batch_size | 20 | 学習時のバッチサイズ |
| setting.save_best_k | 10 | 評価指標が良い上位k個のモデルを保存 |
| fixed.* | — | 固定するハイパーパラメータ（詳細は[学習設定ファイル](#学習設定ファイル)） |
| search_space.* | — | チューニングするハイパーパラメータ（詳細は[学習設定ファイル](#学習設定ファイル)） |
| search_space.*.type | — | 探索タイプの指定（`int` / `float` / `categorical`） |
| search_space.*.low | — | 探索範囲の下限（typeが `int` / `float` の場合） |
| search_space.*.high | — | 探索範囲の上限（typeが `int` / `float` の場合） |
| search_space.*.log | — | 対数スケールで探索するか（typeが `float` の場合） |
| search_space.*.choices | — | 探索候補（typeが `categorical` の場合） |

---

### ハイパラチューニング　出力

`optuna_output_dir`に、以下のディレクトリ構成でハイパラチューニング結果を出力します。

```text
optuna_output_dir/
└─ <study_name>/
   ├─ optuna_config.yml                # 入力の設定ファイルのコピー
   ├─ <study_name>.db                  # Optuna study DB (SQLiteファイル)
   └─ trials/
      ├─ trial0001/                    # 1回目のパラメータ探索結果
      │  ├─ chkpt/                     # モデルの重み
      │  │  ├─ best0001.chkpt           # 評価指標が1番目に良いときの重み
      │  │  ├─ best0002.chkpt           # 評価指標が2番目に良いときの重み
      │  │  └─ ...
      │  ├─ loss/                      # 損失
      │  │  ├─ epoch0001.npz           # epoch 1 の結果
      │  │  ├─ epoch0002.npz           # epoch 2 の結果
      │  │  └─ ...
      │  └─ val_results/               # 検証データでの推論結果
      │     ├─ epoch0001.npz           # epoch 1 終了時の結果
      │     ├─ epoch0002.npz           # epoch 2 終了時の結果
      │     └─ ...
      ├─ trial0002/
      │  └─ ...
      └─ ...
```

`loss/`以下のnpzファイルの中身は以下のようになります。

| key         | dtype   | shape | 内容                                   |
| ----------- | ------- | ----- | ------------------------------------ |
| epoch       | int32   | ()    | epoch番号                              |
| loss_train  | float32 | ()    | 学習データに対する損失                |
| mag_train   | float32 | ()    | 学習データに対する振幅の損失                   |
| phase_train | float32 | ()    | 学習データに対する位相の損失                 |
| loss_val    | float32 | ()    | 検証データに対する損失                 |
| mag_val     | float32 | ()    | 検証データに対する振幅の損失                     |
| phase_val   | float32 | ()    | 検証データに対する位相の損失                   |

`val_results/`以下のnpzファイルの中身は以下のようになります。N_valは検証データのサンプル数、ir_lenは時間波形の長さです。

| key           | dtype     | shape             | 内容                  |
| ------------- | --------- | ----------------- | ------------------- |
| position_tx   | float32   | (N_val, 2)            | 送信機位置（xy） |
| position_rx   | float32   | (N_val, 2)            | 受信機位置（xy） |
| ir_gt        | float32   | (N_val, N_ch, ir_len) | 正解データの時間波形  |
| ir_pred       | float32   | (N_val, N_ch, ir_len) | 推論結果の時間波形  |
| doa_true_deg   | float32 | (N_val,)  | 物理的な音源方向（`position_tx` - `position_rx` から算出する角度） [°]     |
| doa_gt_deg     | float32 | (N_val,)  | 正解データ波形から推定した音源方向 [°]               |
| doa_pred_deg   | float32 | (N_val,)  | 推論結果波形から推定した音源方向 [°]             |
| metric_angle   | float32 | (N_val, N_ch) | 位相の誤差        |
| metric_amp     | float32 | (N_val, N_ch) | 振幅の誤差    |
| metric_env_pct | float32 | (N_val, N_ch) | 包絡線の誤差 [%] |
| metric_t60_pct | float32 | (N_val, N_ch) | 残響時間T60の誤差 [%]       |
| metric_c50_db  | float32 | (N_val, N_ch) | 明瞭度C50の誤差 [dB]      |
| metric_edt_ms  | float32 | (N_val, N_ch) | 初期残響時間EDTの誤差 [ms]      |

---

### 学習　入力

| 入力 | 説明 |
|---|---|
| [学習設定ファイル](#学習設定ファイル) | NAFの学習条件 |
| [前処理結果出力先ディレクトリ](#前処理出力) | 前処理済みデータ |
| [学習結果出力先ディレクトリ](#学習出力) | 学習結果の出力先 |

#### 学習設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は[`train_config.yml`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics/config_files/train_config.yml)を参照してください。  
`param.`で始まる項目は[ハイパラチューニング](#ハイパラチューニング)でチューニング可能なパラメータです。

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| setting.gpus | 4 | 学習に使用するGPU数 |
| setting.model_type | `NAF+` | NAFのモデルタイプ（`NAF` / `NAF+`） |
| setting.dir_ch | 8 | チャンネル数 |
| setting.epochs | 200 | 学習エポック数 |
| setting.resume | False | チェックポイントから再開するかどうか |
| setting.batch_size | 20 | 学習時のバッチサイズ |
| setting.save_best_k | 10 | 評価指標が良い上位k個のモデルを保存 |
| param.mag_alpha | 1.0 | 振幅の損失項係数 |
| param.phase_alpha | 1.0 | 位相の損失項係数 |
| param.lr_init | 1.0e-3 | 初期学習率 |
| param.lr_decay | 1.0e-1 | 学習率減衰係数 |
| param.weight_decay_grid | 1.0e-2 | グリッドパラメータ（gridが名前に入るパラメータ）に対するweight decay |
| param.weight_decay_main | 0.0 | グリッドパラメータ以外のパラメータに対するweight decay |
| param.reg_eps | 0.05 | 学習時に位置に加える微小値 |
| param.pixel_count | 2000 | 一度に計算するスペクトログラムのピクセル数 |
| param.layers | 8 | MLPの層数 |
| param.layers_residual | 1 | ResidualブロックにおけるMLPの層数 |
| param.features | 256 | MLPの中間層における特徴量次元数 |
| param.grid_features | 64 | グリッド特徴量の次元数 |
| param.activation_func_name | default | 活性化関数 |
| param.grid_gap | 0.25 | 初期グリッド点の間隔 |
| param.bandwith_init | 0.25 | 各グリッド点が持つRBFカーネルの初期的な広がり |
| param.position_float | 0.1 | データに合わせてグリッド位置を微調整できる範囲 |
| param.min_bandwidth | 0.1 | カーネル幅の下限 |
| param.max_bandwidth | 0.5 | カーネル幅の上限 |
| param.embed_xyz_num_freqs | 10 | 位置座標の埋め込みにおける周波数の数 |
| param.embed_xyz_max_freq | 7 | 位置座標の埋め込みにおける最大周波数 |
| param.embed_time_num_freqs | 10 | 時刻の埋め込みにおける周波数の数 |
| param.embed_time_max_freq | 10 | 時刻の埋め込みにおける最大周波数 |
| param.embed_freq_num_freqs | 10 | 周波数の埋め込みにおける周波数の数 |
| param.embed_freq_max_freq | 10 | 周波数の埋め込みにおける最大周波数 |
| doa_metric.algorithm | NormMUSIC | DoAアルゴリズム（`MUSIC` / `SRP` など（[詳細](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.doa.html)）） |

---

### 学習 出力

`train_output_dir`に、以下のディレクトリ構成でハイパラチューニング結果を出力します。

```text
train_output_dir/
├─ train_config.yml                  # 入力の設定ファイルのコピー
├─ chkpt/                            # モデルの重み
│  ├─ best0001.chkpt                  # 評価指標が1番目に良いときの重み
│  ├─ best0002.chkpt                  # 評価指標が2番目に良いときの重み
│  └─ ...
├─ loss/                             # 損失
│  ├─ epoch0001.npz                  # epoch 1 の結果
│  ├─ epoch0002.npz                  # epoch 2 の結果
│  └─ ...
└─ val_results/                      # 検証データでの推論結果
   ├─ epoch0001.npz                  # epoch 1 終了時の結果
   ├─ epoch0002.npz                  # epoch 2 終了時の結果
   └─ ...
```


---

### 推論　入力

| 入力 | 説明 |
|---|---|
| [推論設定ファイル](#推論設定ファイル) | NAFの推論条件 |
| 学習済みモデルの重み | 学習済みチェックポイント |
| [送信機データファイル](#送信機データファイル) | 送信機（スピーカー）の位置 |
| [受信機データファイル](#受信機データファイル) | 受信機（マイクロフォンアレイ）の位置 |
| [推論結果出力先ディレクトリ](#推論出力) | 推論結果の出力先 |

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

### 推論　出力

`inference_output_dir`に、以下のディレクトリ構成でハイパラチューニング結果を出力します。

```text
inference_output_dir/
├─ inference_config.yml                  # 入力の設定ファイルのコピー
└─ results.npz                           # 推論結果
```
