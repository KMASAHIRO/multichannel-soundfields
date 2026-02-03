# multichannel-soundfields

「[ニューラル音場推定による仮想音場でのマイクロフォンアレイ音源定位評価](https://www.osaka-kyoiku.ac.jp/~challeng/SIG-Challenge-068/SIG-Challenge-068-03.pdf)」のソースコードです。

ニューラルネットワークを用いた多チャンネル音場推定を実装しています。学習データとなる実データのダウンロードコマンドや、シミュレーションデータの作成コードを含みます。

---

## リポジトリ構成

```text
multichannel-soundfields/
├ README.md          ドキュメント
├ AVR/               音場推定手法：AVR（Acoustic Volume Rendering）
├ AcoustiX/          シミュレーション：AcoustiX
├ NAF/               音場推定手法：NAF（Neural Acoustic Fields）
├ Pyroomacoustics/   シミュレーション：Pyroomacoustics
├ metrics/           推定信号の評価
└ real_data/         実データダウンロード、整形
```

---

## データセット

実データセットのダウンロードコマンドおよびその整形コードを[`real_data`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/real_data)にまとめています。  
シミュレーションデータセットを作成する、AcoustiXおよびPyroomacousticsによるシミュレーションコードをそれぞれ[`AcoustiX`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AcoustiX)、[`Pyroomacoustics`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics)にまとめています。AcoustiXはレイトレーシング、Pyroomacousticsは鏡像法を用いたシミュレータです。

## ニューラルネットワークを用いた音場推定手法

ニューラルネットワークを用いた音場推定手法であるAVR、NAFを多チャンネル音響に拡張した手法をそれぞれ[`AVR`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AVR)、[`NAF`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/NAF)にまとめています。  
AVRについては、チャンネル番号に対応する埋め込みベクトルを入力するAVR+、AVR+の損失関数に音源方向推定の誤差を加えたAVR++を実装しています。  
NAFについては、チャンネル番号に対応する埋め込みベクトルを入力するNAF+を実装しています。

---

## 引用
```
@article{加藤 雅大2025,
  title={ニューラル音場推定による仮想音場でのマイクロフォンアレイ音源定位評価},
  author={加藤 雅大 and 小島 諒介},
  journal={人工知能学会第二種研究会資料},
  volume={2025},
  number={Challenge-068},
  pages={03},
  year={2025},
  doi={10.11517/jsaisigtwo.2025.Challenge-068_03}
}
```
