# AVR

AVRを用いた多チャンネル音場推定  

[AVR](https://zitonglan.github.io/project/avr/avr.html)をベースに、多チャンネル音場へ拡張しました。  
AVRに加え、チャンネル埋め込みを入力するAVR+、さらに音源方向推定の誤差を損失に加えるAVR++を実装しています。

---

## 動作環境

主要なソフトウェア及びそのバージョンは以下の通りになります。

- Python 3.9
- CUDA 12.6
- PyTorch 2.2
- torchaudio 2.2
- NumPy 1.26
- SciPy 1.13
- Matplotlib 3.9
- Pyroomacoustics 0.8
- librosa 0.11
- Optuna 4.4
- [Tiny CUDA Neural Networks](https://github.com/NVlabs/tiny-cuda-nn)

詳細な依存関係および正確なバージョンについては、  
[`requirements.txt`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AVR/requirements.txt) を参照してください。

---

## リポジトリ構成

```text
AVR/
├ README.md                     ドキュメント
├ requirements.txt              依存関係
├ config_files/                 各種設定ファイル（YAML）
│  ├ optuna_config.yml            ハイパラチューニングの設定ファイル
│  ├ train_config.yml             学習の設定ファイル
│  └ inference_config.yml         推論の設定ファイル
├ preprocess/                   前処理
│  └ split_train_val.py           学習データと検証データの分割
├ utils/
│  ├ criterion.py                 損失関数
│  └ spatialization.py            マイクの指向性
├ datasets_loader.py            データローダ
├ model.py                      AVRのニューラルネットワーク
├ renderer.py                   ニューラルネットワークを用いたインパルス応答の生成
├ optuna_tuning.py              Optunaによるハイパラチューニング
├ train.py                      学習
└ inference.py                  推論
```

---

## 実行手順

1. リポジトリのクローン
   
    ```
    git clone https://github.com/KMASAHIRO/multichannel-soundfields  
    cd multichannel-soundfields/AVR
    ```
    
2. 依存関係のインストール

    ```
    pip install -r requirements.txt

    # 例：CUDA 12.6 の場合
    module load cuda/12.6
    pip install git+https://github.com/NVlabs/tiny-cuda-nn@48d6989c95def307a40baf176b2d6015dada19f9#subdirectory=bindings/torch
    ```
    
3. データのダウンロード

    a. 実データの場合 
        
        curl -L -o real_wav_data.zip \
          https://github.com/KMASAHIRO/multichannel-soundfields/releases/download/v0.1.0/real_wav_data.zip

        unzip real_wav_data.zip

        python ../real_data/data_formatting.py \
          --data_dir real_wav_data \
          --output_dir dataset_dir
        
   
    b. シミュレーションデータ（AcoustiX）の場合
   
        curl -L -o AcoustiX_data.zip \
          https://github.com/KMASAHIRO/multichannel-soundfields/releases/download/v0.1.0/AcoustiX_data.zip

        unzip AcoustiX_data.zip
        mv AcoustiX_data dataset_dir
   
    c. シミュレーションデータ（Pyroomacoustics）の場合
   
        curl -L -o Pyroomacoustics_data.zip \
          https://github.com/KMASAHIRO/multichannel-soundfields/releases/download/v0.1.0/Pyroomacoustics_data.zip

        unzip Pyroomacoustics_data.zip
        mv Pyroomacoustics_data dataset_dir

4. 学習データと検証データの分割
    ```
    python preprocess/split_train_val.py \
      --dataset_dir dataset_dir
    ```
5. Optunaによるハイパラメータ探索
    ```
    python optuna_tuning.py \
      --config config_files/optuna_config.yml \
      --data_dir dataset_dir \
      --output_dir optuna_output_dir
    ```
6. 学習
    ```
    python train.py \
      --config config_files/train_config.yml \
      --data_dir dataset_dir \
      --output_dir train_output_dir
    ```
7. 推論
    ```
    python inference.py \
      --config config_files/inference_config.yml \
      --ckpt train_output_dir/ckpt/best0001.tar \
      --speaker config_files/speaker_data.json \
      --receiver config_files/receiver_data.json \
      --output_dir inference_output_dir
    ```
---

## 入出力

### ハイパラチューニング

```
python optuna_tuning.py \
  --config config_files/optuna_config.yml \    # ハイパラチューニング設定ファイル
  --data_dir dataset_dir \                     # データセットディレクトリ
  --output_dir optuna_output_dir               # ハイパラチューニング結果出力先ディレクトリ
```

最小化する目的関数は、検証データにおける `doa_pred_deg` と `doa_gt_deg` の角度誤差（循環差分）の平均値です。  

### ハイパラチューニング　入力

| 入力 | 説明 |
|---|---|
| [ハイパラチューニング設定ファイル](#ハイパラチューニング設定ファイル) | ハイパラチューニングの条件 |
| [データセットディレクトリ](#データセットディレクトリ) | 多チャンネルインパルス応答データセット |
| [ハイパラチューニング結果出力先ディレクトリ](#ハイパラチューニング出力) | ハイパラチューニング結果の出力先 |

#### データセットディレクトリ

`dataset_dir`に、以下のディレクトリ構成で多チャンネルインパルス応答の波形データを用意します。  

```text
dataset_dir/
├ tx_0/                        送信機のインデックス（0,1,2,...）
│  ├ rx_0.npz                  受信機のインデックス（0,1,2,...）
│  ├ rx_1.npz
│  ├ ...
├ tx_1/
│  ├ rx_0.npz
│  ├ ...
├ ...
```

各`rx_*.npz`の内容は以下の通りです。ch_numは受信機（マイクロフォンアレイ）のチャンネル数、ir_lenはインパルス応答の時間方向のサンプル数（波形の長さ）です。

| key | dtype | shape | 内容 |
|---|---|---|---|
| ir | float32 | (ch_num, ir_len) | 多チャンネルインパルス応答の波形 |
| position_rx | float32 | (ch_num, 3) | 受信機の各チャンネルの位置 [x, y, z] |
| position_tx | float32 | (3,) | 送信機位置 [x, y, z] |

ここで、以下コマンドでダウンロードしたデータセットをそのまま使用可能です。各データセットの詳細は[Releases](https://github.com/KMASAHIRO/multichannel-soundfields/releases/tag/v0.1.0)を参照してください。また、[AcoustiX](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AcoustiX#出力)や[Pyroomacoustics](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics#出力)によるシミュレーションデータを使用することもできます。

- 実データ

    ```
    curl -L -o real_wav_data.zip \
      https://github.com/KMASAHIRO/multichannel-soundfields/releases/download/v0.1.0/real_wav_data.zip

    unzip real_wav_data.zip

    python ../real_data/data_formatting.py \
      --data_dir real_wav_data \
      --output_dir dataset_dir
    ```

- シミュレーションデータ（AcoustiX）

    ```
    curl -L -o AcoustiX_data.zip \
      https://github.com/KMASAHIRO/multichannel-soundfields/releases/download/v0.1.0/AcoustiX_data.zip

    unzip AcoustiX_data.zip
    mv AcoustiX_data dataset_dir
    ```

- シミュレーションデータ（Pyroomacoustics）

    ```
    curl -L -o Pyroomacoustics_data.zip \
      https://github.com/KMASAHIRO/multichannel-soundfields/releases/download/v0.1.0/Pyroomacoustics_data.zip

    unzip Pyroomacoustics_data.zip
    mv Pyroomacoustics_data dataset_dir
    ```

#### ハイパラチューニング設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は[`optuna_config.yml`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AVR/config_files/optuna_config.yml)を参照してください。

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| study.study_name | avr_optuna | Optunaのstudy名 |
| study.n_trials | 50 | チューニングの試行回数 |
| doa_metric.algorithm | NormMUSIC | 音源方向推定アルゴリズム（`MUSIC` / `SRP` など（[詳細](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.doa.html)）） |
| doa_metric.n_fft | 512 | 音源方向推定時のSTFTの窓幅 |
| doa_metric.fallback_value | 999.0 | 音源方向推定できない場合の代替値 |
| setting.fs | 16000 | サンプリング周波数 [Hz] |
| setting.speed | 343.8 | 音速 [m/s] |
| setting.xyz_min | 0 | 部屋のxyz座標の最小値 [m] |
| setting.xyz_max | 10 | 部屋のxyz座標の最大値 [m] |
| setting.model_type | AVR | AVRのモデルタイプ（`AVR` / `AVR+` / `AVR++`） |
| setting.ch_num | 8 | チャンネル数 |
| setting.T_max | 8300 | 総学習ステップ数 |
| setting.grad_clip_norm | 1 | 勾配クリップの大きさ |
| setting.resume | True | チェックポイントから再開するかどうか |
| setting.batch_size | 8 | 学習時のバッチサイズ |
| setting.save_best_k | 10 | 評価指標が良い上位何個のモデルを保存するか |
| setting.val_freq | 166 | 検証データで評価する間隔（ステップ数） |
| setting.load_workers | 4 | データ読み込み時の並列数 |
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
   ├─ optuna_config.yml            入力の設定ファイルのコピー
   ├─ <study_name>.db              Optuna study DB (SQLiteファイル)
   └─ trials/                      各試行結果
      ├─ trial0001/                  1回目のパラメータ探索結果
      │  ├─ ckpt/                     モデルの重み
      │  │  ├─ best0001.tar            評価指標が1番目に良いときの重み
      │  │  ├─ best0002.tar            評価指標が2番目に良いときの重み
      │  │  └─ ...
      │  ├─ loss/                     損失
      │  │  ├─ epoch0001.npz           epoch 1 の結果
      │  │  ├─ epoch0002.npz           epoch 2 の結果
      │  │  └─ ...
      │  └─ val_results/              検証データでの推論結果
      │     ├─ epoch0001.npz           epoch 1 終了時の結果
      │     ├─ epoch0002.npz           epoch 2 終了時の結果
      │     └─ ...
      ├─ trial0002/
      │  └─ ...
      └─ ...
```

`loss/`以下のnpzファイルの中身は以下のようになります。

| key         | dtype   | shape | 内容                                   |
| ----------- | ------- | ----- | ------------------------------------ |
| epoch       | int32   | ()    | epoch番号 |
| loss_train  | float32 | ()    | 学習データに対する損失 |
| spec_train   | float32 | ()    | 学習データに対する複素スペクトルの損失 |
| amplitude_train   | float32 | ()    | 学習データに対する振幅の損失 |
| angle_train | float32 | ()    | 学習データに対する位相の損失 |
| time_train | float32 | ()    | 学習データに対する時間波形の損失 |
| energy_train | float32 | ()    | 学習データに対するエネルギー減衰特性の損失 |
| multistft_train | float32 | ()    | 学習データに対する複数解像度STFTについての損失 |
| das_train | float32 | ()    | 学習データに対する音源方向推定の損失（`model_type`が`AVR++`のときのみ） |
| loss_val    | float32 | ()    | 検証データに対する損失                 |
| spec_val   | float32 | ()    | 検証データに対する複素スペクトルの損失 |
| amplitude_val   | float32 | ()    | 検証データに対する振幅の損失 |
| angle_val | float32 | ()    | 検証データに対する位相の損失 |
| time_val | float32 | ()    | 検証データに対する時間波形の損失 |
| energy_val | float32 | ()    | 検証データに対するエネルギー減衰特性の損失 |
| multistft_val | float32 | ()    | 検証データに対する複数解像度STFTについての損失 |
| das_val | float32 | ()    | 検証データに対する音源方向推定の損失（`model_type`が`AVR++`のときのみ） |
| doa_err_val | float32 | ()    | 検証データに対する音源方向推定誤差（`doa_pred_deg` と `doa_gt_deg` の誤差の平均） |

`val_results/`以下のnpzファイルの中身は以下のようになります。N_valは検証データのサンプル数、ir_lenはインパルス応答の時間方向のサンプル数（波形の長さ）です。

| key           | dtype     | shape             | 内容                  |
| ------------- | --------- | ----------------- | ------------------- |
| position_tx | float32 | (N_val, 3) | 送信機位置 [x, y, z] |
| position_rx | float32 | (N_val, ch_num, 3) | 各チャンネルの受信機位置 [x, y, z] |
| ir_gt        | float32   | (N_val, ch_num, ir_len) | 正解データの時間波形  |
| ir_pred       | float32   | (N_val, ch_num, ir_len) | 推論結果の時間波形  |
| doa_true_deg   | float32 | (N_val,)  | 物理的な音源方向（`position_tx` - `position_rx` から算出する角度） [°]     |
| doa_gt_deg     | float32 | (N_val,)  | 正解データ波形から推定した音源方向 [°]               |
| doa_pred_deg   | float32 | (N_val,)  | 推論結果波形から推定した音源方向 [°]             |
| metric_angle   | float32 | (N_val, ch_num) | 位相の誤差        |
| metric_amp     | float32 | (N_val, ch_num) | 振幅の誤差    |
| metric_env_pct | float32 | (N_val, ch_num) | 包絡線の誤差 [%] |
| metric_t60_pct | float32 | (N_val, ch_num) | 残響時間T60の誤差 [%]       |
| metric_c50_db  | float32 | (N_val, ch_num) | 明瞭度C50の誤差 [dB]      |
| metric_edt_ms  | float32 | (N_val, ch_num) | 初期残響時間EDTの誤差 [ms]      |

---

### 学習

```
python train.py \
  --config config_files/train_config.yml \   # 学習設定ファイル
  --data_dir dataset_dir \                   # データセットディレクトリ
  --output_dir train_output_dir              # 学習結果出力先ディレクトリ
```

### 学習　入力

| 入力 | 説明 |
|---|---|
| [学習設定ファイル](#学習設定ファイル) | AVRの学習条件 |
| [データセットディレクトリ](#データセットディレクトリ) | 多チャンネルインパルス応答の波形 |
| [学習結果出力先ディレクトリ](#学習出力) | 学習結果の出力先 |

#### 学習設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は[`train_config.yml`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AVR/config_files/train_config.yml)を参照してください。  
`param.`と`model.`で始まる項目（`signal_output_dim`以外）は[ハイパラチューニング](#ハイパラチューニング)でチューニング可能なパラメータです。

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| doa_metric.algorithm | NormMUSIC | 音源方向推定アルゴリズム（`MUSIC` / `SRP` など（[詳細](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.doa.html)）） |
| doa_metric.n_fft | 512 | 音源方向推定時のSTFTの窓幅 |
| doa_metric.fallback_value | 999.0 | 音源方向推定できない場合の代替値 |
| setting.fs | 16000 | サンプリング周波数 [Hz] |
| setting.speed | 343.8 | 音速 [m/s] |
| setting.xyz_min | 0 | 部屋のxyz座標の最小値 [m] |
| setting.xyz_max | 10 | 部屋のxyz座標の最大値 [m] |
| setting.model_type | AVR | AVRのモデルタイプ（`AVR` / `AVR+` / `AVR++`） |
| setting.ch_num | 8 | チャンネル数 |
| setting.T_max | 8300 | 総学習ステップ数 |
| setting.grad_clip_norm | 1 | 勾配クリップの大きさ |
| setting.resume | False | チェックポイントから再開するかどうか |
| setting.batch_size | 8 | 学習時のバッチサイズ |
| setting.save_best_k | 10 | 評価指標が良い上位何個のモデルを保存するか |
| setting.val_freq | 166 | 検証データで評価する間隔（ステップ数） |
| setting.load_workers | 4 | データ読み込み時の並列数 |
| param.near | 0 | r方向積分の下限 |
| param.far | 6 | r方向積分の上限 |
| param.n_samples | 64 | r方向の積分点数 |
| param.n_azi | 64 | 水平角（φ）方向の積分点数 |
| param.n_ele | 32 | 仰角（θ）方向の積分点数 |
| param.pathloss | 1.5 | 距離によるエネルギー減衰係数 |
| param.lr | 5.0e-4 | 初期学習率 |
| param.eta_min | 5.0e-5 | CosineAnnealingにおける最小学習率 |
| param.weight_decay | 0 | AdamのL2正則化係数 |
| param.optimizer.beta1 | 0.9 | Adamにおける一次モーメントの減衰率 |
| param.optimizer.beta2 | 0.999 | Adamにおける二次モーメントの減衰率 |
| param.spec_loss_weight | 2 | 複素スペクトルの損失項係数 |
| param.amplitude_loss_weight | 4 | 振幅の損失項係数 |
| param.angle_loss_weight | 1 | 位相の損失項係数 |
| param.time_loss_weight | 50 | 時間波形の損失項係数 |
| param.energy_loss_weight | 1 | エネルギー減衰特性の損失項係数 |
| param.multistft_loss_weight | 1 | 複数解像度STFTについての損失項係数 |
| param.das_loss_weight | 0 | 音源方向推定の損失項係数（`model_type`が`AVR++`のときのみ） |
| param.softargmax_beta | 100 | 音源方向推定の損失計算に使うsoft-argmaxの温度パラメータ（`model_type`が`AVR++`のときのみ） |
| model.signal_output_dim | 1600 | 出力信号の長さ（サンプル数） |
| model.channel_embed.is_sigma_encoder | True | sigma encoderに多チャンネル埋め込みベクトルを加えるか（`model_type`が`AVR+`または`AVR++`のとき） |
| model.channel_embed.is_sigma_decoder | True | sigma decoderに多チャンネル埋め込みベクトルを加えるか（`model_type`が`AVR+`または`AVR++`のとき） |
| model.channel_embed.is_signal_network | True | signal networkに多チャンネル埋め込みベクトルを加えるか（`model_type`が`AVR+`または`AVR++`のとき） |
| model.sigma_encoder_network.n_hidden_layers | 3 | sigma encoder の隠れ層の数 |
| model.sigma_encoder_network.n_neurons | 128 | sigma encoder の中間層次元数 |
| model.sigma_decoder_network.n_hidden_layers | 3 | sigma decoder の隠れ層の数 |
| model.sigma_decoder_network.n_neurons | 128 | sigma decoder の中間層次元数 |
| model.signal_network.n_hidden_layers | 3 | signal network の隠れ層の数 |
| model.signal_network.n_neurons | 512 | signal network の中間層次元数 |

---

### 学習 出力

`train_output_dir`に、以下のディレクトリ構成で学習結果を出力します。

```text
train_output_dir/
├─ train_config.yml                  入力の設定ファイルのコピー
├─ ckpt/                             モデルの重み
│  ├─ best0001.tar                     評価指標が1番目に良いときの重み
│  ├─ best0002.tar                     評価指標が2番目に良いときの重み
│  └─ ...
├─ loss/                             損失
│  ├─ epoch0001.npz                    epoch 1 の結果
│  ├─ epoch0002.npz                    epoch 2 の結果
│  └─ ...
└─ val_results/                      検証データでの推論結果
   ├─ epoch0001.npz                    epoch 1 終了時の結果
   ├─ epoch0002.npz                    epoch 2 終了時の結果
   └─ ...
```


---

### 推論

```
python inference.py \
  --config config_files/inference_config.yml \   # 推論設定ファイル
  --ckpt train_output_dir/ckpt/best0001.tar \    # 学習済みモデルの重み
  --speaker config_files/speaker_data.json \     # 送信機データファイル
  --receiver config_files/receiver_data.json \   # 受信機データファイル
  --output_dir inference_output_dir              # 推論結果出力先ディレクトリ
```

### 推論　入力

| 入力 | 説明 |
|---|---|
| [推論設定ファイル](#推論設定ファイル) | AVRの推論条件 |
| 学習済みモデルの重み | 学習済みモデルのチェックポイント |
| [送信機データファイル](#送信機データファイル) | 送信機（スピーカー）の位置 |
| [受信機データファイル](#受信機データファイル) | 受信機（マイクロフォンアレイ）の位置 |
| [推論結果出力先ディレクトリ](#推論出力) | 推論結果の出力先 |

#### 推論設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は[`inference_config.yml`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AVR/config_files/inference_config.yml)を参照してください。

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| setting.fs | 16000 | サンプリング周波数 [Hz] |
| setting.speed | 343.8 | 音速 [m/s] |
| setting.xyz_min | 0 | 部屋のxyz座標の最小値 [m] |
| setting.xyz_max | 10 | 部屋のxyz座標の最大値 [m] |
| setting.model_type | AVR | AVRのモデルタイプ（`AVR` / `AVR+` / `AVR++`） |
| setting.ch_num | 8 | チャンネル数 |
| setting.batch_size | 8 | 推論時のバッチサイズ |
| setting.load_workers | 4 | データ読み込み時の並列数（推論時も使用） |
| param.near | 0 | r方向積分の下限 |
| param.far | 6 | r方向積分の上限 |
| param.n_samples | 64 | r方向の積分点数 |
| param.n_azi | 64 | 水平角（φ）方向の積分点数 |
| param.n_ele | 32 | 仰角（θ）方向の積分点数 |
| param.pathloss | 1.5 | 距離によるエネルギー減衰係数 |
| model.signal_output_dim | 1600 | 出力信号の長さ（サンプル数） |
| model.channel_embed.is_sigma_encoder | True | sigma encoderに多チャンネル埋め込みベクトルを加えるか |
| model.channel_embed.is_sigma_decoder | True | sigma decoderに多チャンネル埋め込みベクトルを加えるか |
| model.channel_embed.is_signal_network | True | signal networkに多チャンネル埋め込みベクトルを加えるか |
| model.sigma_encoder_network.n_hidden_layers | 3 | sigma encoder の隠れ層の数 |
| model.sigma_encoder_network.n_neurons | 128 | sigma encoder の中間層次元数 |
| model.sigma_decoder_network.n_hidden_layers | 3 | sigma decoder の隠れ層の数 |
| model.sigma_decoder_network.n_neurons | 128 | sigma decoder の中間層次元数 |
| model.signal_network.n_hidden_layers | 3 | signal network の隠れ層の数 |
| model.signal_network.n_neurons | 512 | signal network の中間層次元数 |

#### 送信機データファイル

送信機（スピーカー）の位置を定義したJSONファイルを用意します。  
`N_tx`は送信機配置パターンの総数です。

| key | 型 | shape | 内容 |
|---|---|---|---|
| positions | list | (N_tx, 3) | 送信機位置 [x, y, z] |

[論文](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/Challenge-068/2025_03/_article/-char/ja)で使用した[`speaker_data.json`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AVR/config_files/speaker_data.json)は、下図のようなグリッド上の橙点にスピーカーを配置したデータです。

<img width="500" height="426" alt="room_dim" src="https://github.com/user-attachments/assets/049b55de-3061-4ea8-bdd7-519d04ef4a4a" />

#### 受信機データファイル

受信機の位置を定義したJSONファイルを用意します。  
`N_rx`は受信機（マイクロフォンアレイ）の配置数を表し、各受信機はch_numチャンネルで構成されます。ただし、アレイの中心と送信機位置が重なる受信機は除外してシミュレーションを行います。

| key | 型 | shape | 内容 |
|---|---|---|---|
| positions | list | (N_rx, ch_num, 3) | 受信機位置 [x, y, z] |

[論文](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/Challenge-068/2025_03/_article/-char/ja)で使用した[`receiver_data.json`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AVR/config_files/receiver_data.json)は、下図のようなグリッド上に配置した8ch円形マイクロフォンアレイのデータです。

<img width="500" height="426" alt="room_dim" src="https://github.com/user-attachments/assets/049b55de-3061-4ea8-bdd7-519d04ef4a4a" />

---

### 推論　出力

`inference_output_dir`に、以下のディレクトリ構成で推論結果を出力します。

```text
inference_output_dir/
├─ inference_config.yml                  入力の設定ファイルのコピー
└─ results.npz                           推論結果
```

`results.npz`の中身は以下のようになります。N_infは推論データのサンプル数、ir_lenはインパルス応答の時間方向のサンプル数（波形の長さ）です。

| key           | dtype     | shape             | 内容                  |
| ------------- | --------- | ----------------- | ------------------- |
| position_tx | float32 | (N_inf, 3) | 送信機位置 [x, y, z] |
| position_rx | float32 | (N_inf, ch_num, 3) | 各チャンネルの受信機位置 [x, y, z] |
| ir_pred       | float32   | (N_inf, ch_num, ir_len) | 推論結果の時間波形  |
