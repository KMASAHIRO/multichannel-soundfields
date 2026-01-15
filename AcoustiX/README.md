# AcoustiX

AcoustiXを用いた多チャンネル音響シミュレーション  

[AcoustiX](https://github.com/penn-waves-lab/AcoustiX)をベースとして、軽微なバグの修正及び多チャンネル音響シミュレーションコードの追加を行いました。  

AcoutiXは、NVIDIAの電波用レイトレーシングシミュレータ [Sionna ray tracing (Sionna RT)](https://github.com/NVlabs/sionna)を音響用に拡張したシミュレータです。

---

## 動作環境

主要なソフトウェア及びそのバージョンは以下の通りになります。

- Python 3.9
- TensorFlow 2.13
- NumPy 1.24
- SciPy 1.13
- Matplotlib 3.9
- Mitsuba 3.4

詳細な依存関係および正確なバージョンについては、[`requirements.txt`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AcoustiX/requirements.txt) を参照してください。

---

## リポジトリ構成

```text
AcoustiX/
├ README.md                       ドキュメント
├ LICENSE                         ライセンスファイル（MIT License）
├ requirements.txt                依存関係
├ acoustic_absorptions.json       材料ごとの吸音率設定ファイル
├ make_speaker_receiver_data.py   受信機・送信機データファイル生成
├ simu_input/                     シミュレーション入力データ
│  ├ config.yml                     シミュレーション設定ファイル
│  ├ speaker_data.json              送信機データファイル
│  └ receiver_data.json             受信機データファイル
├ simu_utils.py                   シミュレーション用の共通関数
├ simulation.py                   多チャンネル音響のシミュレーション
├ pattern.py                      音源・マイクの指向性パターン
├ check_scene.ipynb               シミュレーション環境（シーン）の可視化
└ sionna/                         Sionna ray tracingモジュール
```

---

## 入出力

### 入力

以下の入力を使ってシミュレーションを実行します。すべての入力が必要です。

| 入力 | 説明 |
|---|---|
| シミュレーション設定ファイル | シミュレーション条件の設定 |
| シーンファイル | シミュレーション環境（シーン）の情報 |
| 送信機データファイル | 送信機（スピーカー）の位置・向き・指向性 |
| 受信機データファイル | 受信機（マイクロフォンアレイ）の位置・向き・指向性 |
| 出力先ディレクトリ | シミュレーション結果の保存先 |

---

#### シミュレーション設定ファイル

YAMLファイルで以下の内容を設定します。  
具体的な書き方は[`config.yml`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AcoustiX/simu_input/config.yml)を参照してください。

| 項目 | デフォルト値 | 説明 |
|---|---|---|
| max_depth | 10 | 音線の最大反射回数 |
| num_samples | 50000 | 放射する音線数 |
| los | True | 直接音を計算するか |
| reflection | True | 反射音を考慮するか |
| diffraction | False | 回折効果を考慮するか |
| scattering | True | 散乱効果を考慮するか |
| scat_prob | 0.00001 | 音線が散乱する確率 |
| attn | 0.001 | 減衰係数 |
| fs | 16000 | サンプリング周波数 [Hz] |
| ir_len | 1600 | インパルス応答長（サンプル数） |
| speed | 343.8 | 音速 [m/s] |
| noise | 0.0 | 波形に加えるノイズの大きさ |

---

#### シーンファイル

シミュレーション環境（シーン）を表すXMLファイル、plyファイルをBlenderで作成する必要があります。

[論文](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/Challenge-068/2025_03/_article/-char/ja)で使用した、`6.11×8.807×2.7 [m]`の直方体のシーンファイルは、[Google Drive](https://drive.google.com/drive/folders/1h1R4gZKTwJghD0qsZyB5vbckLi2LphX3)からダウンロードできます。 
Drive内のデータをすべてダウンロードし、そのままのディレクトリ構成でシミュレーション実行環境に配置してください。   
シーンを自作したい場合は[シーンを自作する場合](#シーンを自作する場合)を参照してください。

<img width="1920" height="1094" alt="scene_on_paper" src="https://github.com/user-attachments/assets/1f749054-2d83-4fad-9c19-3c918ee8b450" />

---

#### 送信機データファイル

送信機（スピーカー）の位置、向き、指向性パターンを定義したJSONファイルを用意します。  
指向性パターン`patterns`を`uniform`（指向性なし）にした場合は、向きによる影響はありません。  
`N_tx`は送信機配置パターンの総数です。各シミュレーションでは送信機は1台のみ配置します。

| key | 型 | shape | 内容 |
|---|---|---|---|
| positions | list | (N_tx, 3) | 送信機位置 [x, y, z] |
| orientations | list | (N_tx, 3) | 送信機の向き [x, y, z] |
| patterns | list | (N_tx,) | 送信機の指向性パターン（`"heart"` / `"donut"` / `"uniform"`） |

[論文](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/Challenge-068/2025_03/_article/-char/ja)で使用した、下図のようなグリッド上の橙点に配置されたスピーカーに対応するファイルは、[`speaker_data.json`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AcoustiX/simu_input/speaker_data.json)を参照してください。

<img width="500" height="426" alt="room_dim" src="https://github.com/user-attachments/assets/049b55de-3061-4ea8-bdd7-519d04ef4a4a" />


---

#### 受信機データファイル

受信機（マイクロフォンアレイ）の位置、向き、指向性パターンを定義したJSONファイルを用意します。  
指向性パターン`patterns`を`uniform`（指向性なし）にした場合は、向きによる影響はありません。  
`N_rx`は受信機（マイクロフォンアレイ）の配置数を表し、各受信機はN_chチャンネルで構成されます。ただし、アレイの中心と送信機位置が重なる受信機は除外してシミュレーションを行います。

| key | 型 | shape | 内容 |
|---|---|---|---|
| positions | list | (N_rx, N_ch, 3) | 受信機位置 [x, y, z] |
| orientations | list | (N_rx, N_ch, 3) | 受信機の向き [x, y, z] |
| patterns | list | (N_rx, N_ch) | 受信機の指向性パターン（`"heart"` / `"donut"` / `"uniform"`） |

[論文](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/Challenge-068/2025_03/_article/-char/ja)で使用した、下図のようなグリッド上に配置された8ch円形マイクロフォンアレイに対応するファイルは、[`receiver_data.json`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AcoustiX/simu_input/receiver_data.json)を参照してください。

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
| orientation_rx | ndarray | (N_ch, 3)  | 受信機の向き [x, y, z] |
| orientation_tx | ndarray | (3,)  | 送信機の向き [x, y, z]   |
| pattern_rx     | ndarray     | (N_ch,)  | 受信機の指向性パターン（`"heart"` / `"donut"` / `"uniform"`） |
| pattern_tx     | ndarray     | ()  | 送信機の指向性パターン（`"heart"` / `"donut"` / `"uniform"`） |

---

## シミュレーション手順

1. リポジトリのクローン

```
git clone https://github.com/KMASAHIRO/multichannel-soundfields
cd multichannel-soundfields/AcoustiX
```

2. 依存関係のインストール

```
pip install -r requirements.txt
```

3. シーンファイルのダウンロード（自作シーンを使用する場合はスキップ）

```
curl -L -o simu_input/AcoustiX_room.zip \
  [URL]

unzip simu_input/AcoustiX_room.zip -d simu_input
```

4. シミュレーションの実行

```
python simulation.py \
  --config simu_input/config.yml \
  --scene simu_input/AcoustiX_room/AcoustiX_room.xml \
  --speaker simu_input/speaker_data.json \
  --receiver simu_input/receiver_data.json \
  --output_dir outputs
```

---

## シーンを自作する場合

以下の手順は、[Sionna RTの公式チュートリアル動画](https://www.youtube.com/watch?v=7xHLDxUaQ7c)を元にまとめたものです。

### 1. 必要なソフトウェアのインストール

シーンファイルを自作するには、Blender及びMitsuba-Blenderアドオンが必要です。それぞれ以下のバージョンを使用することを推奨します。

- Blender 3.6.0  
  https://download.blender.org/release/Blender3.6/
- Mitsuba-Blender v0.3.0  
  https://github.com/mitsuba-renderer/mitsuba-blender/releases/tag/v0.3.0

まず、[Blender 3.6のダウンロードページ](https://download.blender.org/release/Blender3.6/)から自身の環境（OS/CPUアーキテクチャ）に応じてBlender 3.6.0 をダウンロードし、インストールを完了してください。  
次に、[Mitsuba-Blender v0.3.0](https://github.com/mitsuba-renderer/mitsuba-blender/releases/tag/v0.3.0)のAssetsにある`mitsuba-blender.zip`をダウンロードし、[インストールガイド](https://github.com/mitsuba-renderer/mitsuba-blender/wiki/Installation-&-Update-Guide)に従って、Mitsuba-Blenderアドオンのインストールをしてください。

### 2. 3Dオブジェクトの作成

ここでは、`6.11×8.807×2.7 [m]`の直方体のオブジェクトを作成することとします。
Blenderを立ち上げ、デフォルトで1辺2mの立方体とカメラ、ライトが用意されていることを確認します。  

<img width="1919" height="1093" alt="1_start_menu" src="https://github.com/user-attachments/assets/5aa6f87b-5f45-4329-9ab8-8a768a205cb8" />

画面右上の`Scene Collection`から`Camera`と`Light`を選択し、キーボードのDeleteボタンで削除します。  
次に、画面中央の立方体をクリックし、キーボードのNキーを押します。サイドバーが現れるので、`Transform`→`Dimensions`の`X`、`Y`、`Z`をそれぞれ`6.11 m`、`8.807 m`、`2.7 m`に設定し、`Location`の`X`、`Y`、`Z`をそれぞれ`6.11/2 m`、`8.807/2 m`、`2.7/2 m`と入力します（下画像の赤枠参照）。すると、角の位置が座標上の原点となるような直方体ができます。

<img width="1919" height="1092" alt="2_change_room_dim" src="https://github.com/user-attachments/assets/7890ff47-35d8-4e5f-adde-40ee2319f7f5" />


このように、仮定するシミュレーション状況に合わせて自由に3Dオブジェクトを作成してください。複数のオブジェクトを作成しても問題ありません。


### 3. 材料の設定

3Dオブジェクトの各面に使用する材料を設定します。材料は[`acoustic_absorptions.json`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AcoustiX/acoustic_absorptions.json)のキーから選択してください。このjsonファイルは、各材料の周波数ごとの吸音率を定義しています。  


例えば、直方体の各面に`Smooth concrete, painted or glazed`を使用する場合は、Blenderで直方体を選択した状態で右下パネルの`Material Properties`を開き、名前を`Smooth concrete, painted or glazed`とします。画像内右下で赤枠に囲まれたピンク色の円形マークが`Material Properties`であり、画像右側中央で赤枠に囲まれたテキストボックス部分に名前を入力します。

<img width="1919" height="1092" alt="3_set_mat_param" src="https://github.com/user-attachments/assets/d9844dff-b9ee-4d16-9afb-706fc4385a2c" />

### 4. Mitsuba形式でのシーンのエクスポート

Blender画面左上の`File`→`Export`→`Mitsuba (.xml)`を選択します。ここで、`Mitsuba (.xml)`が表示されない場合は、[Mitsuba-Blenderアドオンのインストールガイド](https://github.com/mitsuba-renderer/mitsuba-blender/wiki/Installation-&-Update-Guide)をもう一度参照してください。

<img width="1919" height="1090" alt="4_mitsuba_export_button" src="https://github.com/user-attachments/assets/b33ad8f1-9031-4d23-ace0-f50679bae5ba" />


`Export IDs`、`Ignore Default Background`にチェックが入っていること、`Y Forward`、`Z Up`となっていることを確認し、適当な名前を付けて保存します（ここでは、AcoustiX_room.xml）。

<img width="1232" height="812" alt="5_export_settings" src="https://github.com/user-attachments/assets/22ef1b24-98f1-40fd-8a2f-b934c2dc2bb3" />

### 5. 出力ファイルの確認

エクスポート時に名前を付けたxmlファイルの他に、同階層のmeshesディレクトリ以下にplyファイルが存在することを確認します。シミュレーション時にAcoustiXに入力するのはxmlファイルのパスだけですが、内部的にplyファイルも使用するので、ファイルを移動する場合はディレクトリ構成を崩さずにまとめて移動してください。

---

## ライセンス

本リポジトリは MIT License に従って公開されています。詳細は[`LICENSE`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AcoustiX/LICENSE)を確認してください。
