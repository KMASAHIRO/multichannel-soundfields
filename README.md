# multichannel-soundfields

「[ニューラル音場推定による仮想音場でのマイクロフォンアレイ音源定位評価](https://www.osaka-kyoiku.ac.jp/~challeng/SIG-Challenge-068/SIG-Challenge-068-03.pdf)」のソースコードです。  
多チャンネル音場推定（AVR/NAF）の学習・推論、シミュレーションデータ作成、実データの整形、評価までをまとめています。
実行方法の詳細は、各ディレクトリの`README.md`に記載しています。

---

## リポジトリ構成

```text
multichannel-soundfields/
├ README.md          ドキュメント
├ LICENSE            ライセンスファイル（MIT License）
├ AVR/               音場推定手法：AVR（Acoustic Volume Rendering）
├ AcoustiX/          シミュレーション：AcoustiX（レイトレーシング）
├ NAF/               音場推定手法：NAF（Neural Acoustic Fields）
├ Pyroomacoustics/   シミュレーション：Pyroomacoustics（鏡像法）
├ metrics/           推定信号の評価
└ real_data/         実データのダウンロード・整形
```

---

## データセット

実データのダウンロードと整形は[`real_data`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/real_data)にまとめています。  
シミュレーションデータは[`AcoustiX`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AcoustiX)（レイトレーシング）と[`Pyroomacoustics`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/Pyroomacoustics)（鏡像法）で作成します。

## ニューラルネットワークを用いた音場推定手法

多チャンネル音場推定の手法として、[`AVR`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AVR)と[`NAF`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/NAF)を用意しています。  
[`AVR`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AVR)では、AVRに多チャンネル埋め込みを加えたAVR+、さらに音源方向推定誤差を損失に追加したAVR++を実装しています。  
[`NAF`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/NAF)では、NAFに多チャンネル埋め込みを加えたNAF+を実装しています。

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
