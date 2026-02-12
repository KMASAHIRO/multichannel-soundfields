# real_data

実データセットのダウンロードと整形を行います。  
[論文](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/Challenge-068/2025_03/_article/-char/ja)で使用した、8ch円形マイクロフォンアレイで測定したインパルス応答データセットです。  
下図のグリッド上 4 × 6 = 24 個の配置点の内、橙点8か所をスピーカー候補位置とし、1回の測定ではそのうち1か所にスピーカーを配置しました。残り23か所にマイクロフォンアレイを配置して記録し、これをスピーカー候補位置8か所それぞれで行いました。こうして作成した、8 × 23 = 184 個の多チャンネルインパルス応答データセットです。
データセットの詳細は[Releases](https://github.com/KMASAHIRO/multichannel-soundfields/releases/tag/v0.1.0)を参照してください。

<img width="500" height="426" alt="room_dim" src="https://github.com/user-attachments/assets/049b55de-3061-4ea8-bdd7-519d04ef4a4a" />

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
real_data/
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
curl -L -o real_wav_data.zip \
  https://github.com/KMASAHIRO/multichannel-soundfields/releases/download/v0.1.0/real_wav_data.zip

unzip real_wav_data.zip
```

4. データ整形

```
python data_formatting.py \
  --data_dir real_wav_data \
  --output_dir dataset_dir
```

---

## 出力

`dataset_dir`に、以下のディレクトリ構成で8チャンネルのインパルス応答の波形データが出力されます。

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

各`rx_*.npz`の内容は以下の通りです。

| key | dtype | shape | 内容 |
|---|---|---|---|
| ir | float32 | (8, 1600) | 8チャンネルのインパルス応答の波形 |
| position_rx | float32 | (8, 3) | 受信機位置 [x, y, z] |
| position_tx | float32 | (3,) | 送信機位置 [x, y, z] |
