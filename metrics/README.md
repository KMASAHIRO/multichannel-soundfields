# metrics

AVRやNAFの推論結果の評価

---

## 動作環境

主要なソフトウェア及びそのバージョンは以下の通りになります。

- Python 3.9
- PyTorch 2.7
- NumPy 2.0
- SciPy 1.13
- Pyroomacoustics 0.8
- librosa 0.11

詳細な依存関係および正確なバージョンについては、  
[`requirements.txt`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/metrics/requirements.txt) を参照してください。

---

## リポジトリ構成

```text
metrics/
├ README.md                         ドキュメント
├ requirements.txt                  依存関係
├ config_files/                     各種設定ファイル（YAML）
│  ├ metrics_config.yml             評価の設定ファイル
│  └ whitenoise_metrics_config.yml  ホワイトノイズ音源による評価の設定ファイル
├ IR_metrics.py                     IR評価指標関数
├ DoA_metrics.py                    DoA評価指標関数
├ compute_metrics.py                推論結果の評価
└ compute_whitenoise_metrics.py     ホワイトノイズ音源による推論結果の評価
```

---

## 実行手順

[`AVR`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AVR)または[`NAF`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/NAF)で推論まで実行してから、以下のコードを実行してください。

1. 依存関係のインストール

```
pip install -r requirements.txt
```

2. 推論結果の評価

```
python compute_metrics.py \
  --config config_files/metrics_config.yml \
  --input ../NAF/inference_output_dir/results.npz \
  --output_dir metrics_output_dir
```

3. ホワイトノイズ音源による推論結果の評価

python compute_whitenoise_metrics.py \
  --config config_files/whitenoise_metrics_config.yml \
  --input ../NAF/inference_output_dir/results.npz \
  --output_dir whitenoise_metrics_output_dir

---

## 入出力

### 推論結果の評価　入力

| 入力 | 説明 |
|---|---|
| [評価設定ファイル](#評価設定ファイル) | 評価関数のパラメータ |
| [推論結果](#推論結果) | AVRやNAFの推論結果 |
| [出力先ディレクトリ](#推論結果の評価出力) | 評価結果の保存先 |


#### 推論結果

[AVR](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AVR#推論出力)や[NAF](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/NAF#推論出力)の推論結果と正解データの時間波形`ir_gt`を含む`results.npz`を用意します。形式は以下の通りです。N_infは推論データのサンプル数、ch_numは受信機（マイクロフォンアレイ）のチャンネル数、ir_lenは時間波形の長さです。なお、`ir_gt`が含まれていない場合、推論結果波形を用いた音源方向の推定のみ実行されます。

| key           | dtype     | shape             | 内容                  |
| ------------- | --------- | ----------------- | ------------------- |
| position_tx | float32 | (N_inf, 3) or (N_inf, 2) | 送信機位置 [x, y, z] or [x, y] |
| position_rx | float32 | (N_inf, ch_num, 3) or (N_inf, ch_num, 2) | 各チャンネルの受信機位置 [x, y, z] or [x, y] |
| ir_pred       | float32   | (N_inf, ch_num, ir_len) | 推論結果の時間波形  |
| ir_gt       | float32   | (N_inf, ch_num, ir_len) | 正解データの時間波形  |

#### 評価設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は[`metrics_config.yml`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/metrics/config_files/metrics_config.yml)を参照してください。  

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| doa_algorithm | NormMUSIC | 音源方向推定アルゴリズム（`MUSIC` / `SRP` など（[詳細](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.doa.html)）） |
| fs | 16000 | サンプリング周波数 [Hz] |
| n_fft | 512 | STFTの窓幅 |
| hop_size | 128 | STFTのhop size |
| window | hann | 窓関数（`hamming` / `gaussian` / `boxcar` など（[詳細](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html)）） |

---

### 推論結果の評価　出力

`metrics_output_dir`に、以下のディレクトリ構成で評価結果を出力します。

```text
metrics_output_dir/
├─ metrics_config.yml                  入力の設定ファイルのコピー
└─ metrics_results.npz                 評価結果
```

`metrics_results.npz`の中身は以下のようになります。N_infは推論データのサンプル数、ch_numは受信機（マイクロフォンアレイ）のチャンネル数、ir_lenは時間波形の長さです。

| key           | dtype     | shape             | 内容                  |
| ------------- | --------- | ----------------- | ------------------- |
| position_tx | float32 | (N_inf, 3) or (N_inf, 2) | 送信機位置 [x, y, z] or [x, y] |
| position_rx | float32 | (N_inf, ch_num, 3) or (N_inf, ch_num, 2) | 各チャンネルの受信機位置 [x, y, z] or [x, y] |
| ir_gt        | float32   | (N_inf, ch_num, ir_len) | 正解データの時間波形  |
| ir_pred       | float32   | (N_inf, ch_num, ir_len) | 推論結果の時間波形  |
| doa_true_deg   | float32 | (N_inf,)  | 物理的な音源方向（`position_tx` - `position_rx` から算出する角度） [°]     |
| doa_gt_deg     | float32 | (N_inf,)  | 正解データ波形から推定した音源方向 [°]               |
| doa_pred_deg   | float32 | (N_inf,)  | 推論結果波形から推定した音源方向 [°]             |
| metric_angle   | float32 | (N_inf, ch_num) | 位相の誤差        |
| metric_amp     | float32 | (N_inf, ch_num) | 振幅の誤差    |
| metric_env_pct | float32 | (N_inf, ch_num) | 包絡線の誤差 [%] |
| metric_t60_pct | float32 | (N_inf, ch_num) | 残響時間T60の誤差 [%]       |
| metric_c50_db  | float32 | (N_inf, ch_num) | 明瞭度C50の誤差 [dB]      |
| metric_edt_ms  | float32 | (N_inf, ch_num) | 初期残響時間EDTの誤差 [ms]      |

---

### ホワイトノイズ音源による推論結果の評価　入力

| 入力 | 説明 |
|---|---|
| [ホワイトノイズ音源による評価設定ファイル](#ホワイトノイズ音源による評価設定ファイル) | 評価条件 |
| [推論結果](#推論結果) | AVRやNAFの推論結果 |
| [出力先ディレクトリ](#ホワイトノイズ音源による推論結果の評価出力) | 評価結果の保存先 |

#### ホワイトノイズ音源による評価設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は[`whitenoise_metrics_config.yml`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/metrics/config_files/whitenoise_metrics_config.yml)を参照してください。  

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| doa_algorithm | NormMUSIC | 音源方向推定アルゴリズム（`MUSIC` / `SRP` など（[詳細](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.doa.html)）） |
| fs | 16000 | サンプリング周波数 [Hz] |
| n_fft | 1024 | STFTの窓幅 |
| hop_size | 512 | STFTのhop size |
| window | hann | 窓関数（`hamming` / `gaussian` / `boxcar` など（[詳細](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html)）） |
| noise_seconds | 100 | ホワイトノイズ音源の秒数 |
| random_seed | 0 | ランダムシード |
| split_time_frame | 64 | 音源方向推定に使用する時間フレーム数 |

### ホワイトノイズ音源による推論結果の評価　出力

`whitenoise_metrics_output_dir`に、以下のディレクトリ構成で評価結果を出力します。

```text
whitenoise_metrics_output_dir/
├─ whitenoise_metrics_config.yml                  入力の設定ファイルのコピー
└─ whitenoise_metrics_results.npz                 評価結果
```

`whitenoise_metrics_results.npz`の中身は以下のようになります。N_infは推論データのサンプル数、ch_numは受信機（マイクロフォンアレイ）のチャンネル数、ir_lenは時間波形の長さ、N_timeframeはインパルス応答とホワイトノイズ畳み込み後の時間方向の分割数です。

| key           | dtype     | shape             | 内容                  |
| ------------- | --------- | ----------------- | ------------------- |
| position_tx | float32 | (N_inf, 3) or (N_inf, 2) | 送信機位置 [x, y, z] or [x, y] |
| position_rx | float32 | (N_inf, ch_num, 3) or (N_inf, ch_num, 2) | 各チャンネルの受信機位置 [x, y, z] or [x, y] |
| ir_gt        | float32   | (N_inf, ch_num, ir_len) | 正解データの時間波形  |
| ir_pred       | float32   | (N_inf, ch_num, ir_len) | 推論結果の時間波形  |
| doa_true_deg   | float32 | (N_inf, N_timeframe)  | 物理的な音源方向（`position_tx` - `position_rx` から算出する角度） [°]     |
| doa_gt_deg     | float32 | (N_inf, N_timeframe)  | 正解データ波形から推定した音源方向 [°]               |
| doa_pred_deg   | float32 | (N_inf, N_timeframe)  | 推論結果波形から推定した音源方向 [°]             |
