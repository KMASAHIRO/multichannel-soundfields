# AVR

実データのダウンロード、整形

---

## 動作環境

主要なソフトウェア及びそのバージョンは以下の通りになります。

- Python 3.9
- NumPy 2.0

詳細な依存関係および正確なバージョンについては、  
[`requirements.txt`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/real_data/requirements.txt) を参照してください。

---

## リポジトリ構成

```text
NAF/
├ README.md                     ドキュメント
├ requirements.txt              依存関係
└ data_formatting.py            データ整形
```

---

## 実行手順

1. リポジトリのクローン

```
git clone https://github.com/KMASAHIRO/multichannel-soundfields  
cd multichannel-soundfields/real_data
```

2. 依存関係のインストール

```
pip install -r requirements.txt
```

3. 実データのダウンロード

```
curl -L -o real_data_original.zip \
  [URL]

unzip real_data_original.zip -d real_data_original
```

4. データ整形

```
python data_formatting.py \
  --data_dir real_data_original \
  --output_dir real_data
```

---

## 出力

`real_data`に、以下のディレクトリ構成で8チャンネルのインパルス応答の波形データが出力されます。

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

各`rx_*.npz`の内容は以下の通りです。ir_lenは時間波形の長さです。

| key | dtype | shape | 内容 |
|---|---|---|---|
| ir | float32 | (8, ir_len) | 8チャンネルのインパルス応答の波形 |
| position_rx | float32 | (8, 3) | 各チャンネルの受信機位置 [x, y, z] |
| position_tx | float32 | (3,) | 送信機位置 [x, y, z] |
