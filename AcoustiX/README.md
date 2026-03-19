# AcoustiX

AcoustiXを用いた多チャンネル音響シミュレーション  

[AcoustiX](https://github.com/penn-waves-lab/AcoustiX)をベースに、軽微なバグ修正と多チャンネル音響シミュレーション機能を追加しました。  
AcoustiXは、NVIDIAの電波用レイトレーシングシミュレータ[Sionna ray tracing (Sionna RT)](https://github.com/NVlabs/sionna)を音響向けに拡張したものです。

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

## シミュレーション手順

1. リポジトリのクローン

```bash
git clone https://github.com/KMASAHIRO/multichannel-soundfields
cd multichannel-soundfields/AcoustiX
```

2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

3. シーンファイルのダウンロード（自作シーンを使用する場合はスキップ）

```bash
curl -L -o simu_input/AcoustiX_room.zip \
  https://github.com/KMASAHIRO/multichannel-soundfields/releases/download/v0.1.0/AcoustiX_room.zip

unzip simu_input/AcoustiX_room.zip -d simu_input
```

4. シミュレーションの実行

```bash
python simulation.py \
  --config simu_input/config.yml \
  --scene simu_input/AcoustiX_room/AcoustiX_room.xml \
  --speaker simu_input/speaker_data.json \
  --receiver simu_input/receiver_data.json \
  --output_dir output_dir
```

---

## 入出力

### シミュレーション

```bash
python simulation.py \
  --config simu_input/config.yml \                     # シミュレーション設定ファイル
  --scene simu_input/AcoustiX_room/AcoustiX_room.xml \ # シーンファイル
  --speaker simu_input/speaker_data.json \             # 送信機データファイル
  --receiver simu_input/receiver_data.json \           # 受信機データファイル
  --output_dir output_dir                              # 出力先ディレクトリ
```

### 入力

以下の入力を使ってシミュレーションを実行します。すべての入力が必要です。

| 入力 | 説明 |
|---|---|
| [シミュレーション設定ファイル](#シミュレーション設定ファイル) | シミュレーション条件の設定 |
| [シーンファイル](#シーンファイル) | シミュレーション環境（シーン）の情報 |
| [送信機データファイル](#送信機データファイル) | 送信機（スピーカー）の位置・向き・指向性 |
| [受信機データファイル](#受信機データファイル) | 受信機（マイクロフォンアレイ）の位置・向き・指向性 |
| [出力先ディレクトリ](#出力) | シミュレーション結果の保存先 |

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
| attn | 0.001 | 距離による減衰係数 |
| fs | 16000 | サンプリング周波数 [Hz] |
| ir_len | 1600 | インパルス応答の時間方向のサンプル数（波形の長さ） |
| speed | 343.8 | 音速 [m/s] |
| noise | 0.0 | 波形に加えるノイズの大きさ |

---

#### シーンファイル

シーンはXMLとPLYで構成します。Blenderで作成し、Mitsuba形式でエクスポートします。

[論文](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/Challenge-068/2025_03/_article/-char/ja)で使用した、`6.11×8.807×2.7 [m]`の直方体シーンは以下のコマンドでダウンロードできます。ディレクトリ構成を保ったまま配置してください。  
自作する場合は[シーンを自作する場合](#シーンを自作する場合)を参照してください。  
AIでシーンを生成する場合は[AIでシーンを生成する場合](#AIでシーンを生成する場合)を参照してください。

```bash
curl -L -o simu_input/AcoustiX_room.zip \
  https://github.com/KMASAHIRO/multichannel-soundfields/releases/download/v0.1.0/AcoustiX_room.zip

unzip simu_input/AcoustiX_room.zip -d simu_input
```

<img width="1920" height="1094" alt="scene_on_paper" src="https://github.com/user-attachments/assets/1f749054-2d83-4fad-9c19-3c918ee8b450" />

---

#### 送信機データファイル

送信機（スピーカー）候補の位置・向き・指向性パターンを定義したJSONファイルを用意します。
`N_tx`は候補の総数で、1回のシミュレーションで使用する送信機は1台のみです（単一音源を想定）。
指向性パターン`patterns`を`uniform`にすると向きの影響は無視されます。

| key | 型 | shape | 内容 |
|---|---|---|---|
| positions | list | (N_tx, 3) | 送信機位置 [x, y, z] |
| orientations | list | (N_tx, 3) | 送信機の向き [x, y, z] |
| patterns | list | (N_tx,) | 送信機の指向性パターン（`"heart"` / `"donut"` / `"uniform"`） |

[論文](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/Challenge-068/2025_03/_article/-char/ja)で使用した[`speaker_data.json`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AcoustiX/simu_input/speaker_data.json)は、下図のようなグリッド上の橙点にスピーカーを配置したデータです。

<img width="500" height="426" alt="room_dim" src="https://github.com/user-attachments/assets/049b55de-3061-4ea8-bdd7-519d04ef4a4a" />


---

#### 受信機データファイル

受信機（マイクロフォンアレイ）の位置、向き、指向性パターンを定義したJSONファイルを用意します。  
指向性パターン`patterns`を`uniform`にすると向きの影響は無視されます。  
`N_rx`は受信機配置数で、各受信機はch_numチャンネルで構成されます。アレイ中心と送信機位置が重なる受信機は除外します。

| key | 型 | shape | 内容 |
|---|---|---|---|
| positions | list | (N_rx, ch_num, 3) | 受信機位置 [x, y, z] |
| orientations | list | (N_rx, ch_num, 3) | 受信機の向き [x, y, z] |
| patterns | list | (N_rx, ch_num) | 受信機の指向性パターン（`"heart"` / `"donut"` / `"uniform"`） |

[論文](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/Challenge-068/2025_03/_article/-char/ja)で使用した[`receiver_data.json`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AcoustiX/simu_input/receiver_data.json)は、下図のようなグリッド上に配置した8ch円形マイクロフォンアレイのデータです。

<img width="500" height="426" alt="room_dim" src="https://github.com/user-attachments/assets/049b55de-3061-4ea8-bdd7-519d04ef4a4a" />

---

### 出力

入力時に指定した出力先ディレクトリ`output_dir`に、以下の構成で出力します。

```text
output_dir/
├ config.yml
├ speaker_data.json
├ receiver_data.json
├ tx_0/                        送信機のインデックス（0,1,2,...）
│  ├ rx_0.npz                  受信機のインデックス（0,1,2,...）
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
| ir             | float32 | (ch_num, ir_len)  | インパルス応答の波形      |
| position_rx    | float32 | (ch_num, 3)  | 受信機位置 [x, y, z]  |
| position_tx    | float32 | (3,)  | 送信機位置 [x, y, z]    |
| orientation_rx | float32 | (ch_num, 3)  | 受信機の向き [x, y, z] |
| orientation_tx | float32 | (3,)  | 送信機の向き [x, y, z]   |
| pattern_rx     | str     | (ch_num,)  | 受信機の指向性パターン（`"heart"` / `"donut"` / `"uniform"`） |
| pattern_tx     | str     | ()  | 送信機の指向性パターン（`"heart"` / `"donut"` / `"uniform"`） |


この`output_dir`は、[AVR](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AVR#データセットディレクトリ)や[NAF](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/NAF#データセットディレクトリ)の`dataset_dir`としてそのまま利用できます。

---

## シーンを自作する場合

以下の手順は、[Sionna RTの公式チュートリアル動画](https://www.youtube.com/watch?v=7xHLDxUaQ7c)を元にまとめたものです。

### 1. 必要なソフトウェアのインストール

シーンファイルを自作するには、Blender及びMitsuba-Blenderアドオンが必要です。推奨バージョンは以下の通りです。

- Blender 3.6.0  
  https://download.blender.org/release/Blender3.6/
- Mitsuba-Blender v0.3.0  
  https://github.com/mitsuba-renderer/mitsuba-blender/releases/tag/v0.3.0

まず、[Blender 3.6のダウンロードページ](https://download.blender.org/release/Blender3.6/)から自身の環境（OS/CPUアーキテクチャ）に応じてBlender 3.6.0をダウンロードし、インストールします。  
次に、[Mitsuba-Blender v0.3.0](https://github.com/mitsuba-renderer/mitsuba-blender/releases/tag/v0.3.0)のAssetsにある`mitsuba-blender.zip`をダウンロードし、[インストールガイド](https://github.com/mitsuba-renderer/mitsuba-blender/wiki/Installation-&-Update-Guide)に従ってMitsuba-Blenderアドオンを導入します。

### 2. 3Dオブジェクトの作成

ここでは、`6.11×8.807×2.7 [m]`の直方体を作成する例を示します。  
Blenderを起動し、デフォルトで1辺2mの立方体とカメラ、ライトがあることを確認します。

<img width="1919" height="1093" alt="1_start_menu" src="https://github.com/user-attachments/assets/5aa6f87b-5f45-4329-9ab8-8a768a205cb8" />

画面右上の`Scene Collection`から`Camera`と`Light`を選択し、`Delete`キーで削除します。  
次に、画面中央の立方体を選択して`N`キーを押し、サイドバーの`Transform`を開きます。`Dimensions`の`X`、`Y`、`Z`をそれぞれ`6.11 m`、`8.807 m`、`2.7 m`に設定し、`Location`の`X`、`Y`、`Z`をそれぞれ`6.11/2 m`、`8.807/2 m`、`2.7/2 m`に設定します。これで角の位置が座標上の原点となるような直方体になります。  
用途に合わせて複数オブジェクトを作成しても問題ありません。

<img width="1919" height="1092" alt="2_change_room_dim" src="https://github.com/user-attachments/assets/7890ff47-35d8-4e5f-adde-40ee2319f7f5" />

### 3. 材料の設定

3Dオブジェクトの各面に使用する材料を設定します。材料名は[`acoustic_absorptions.json`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AcoustiX/acoustic_absorptions.json)のキーから選択してください。このjsonファイルは、各材料の周波数ごとの吸音率を定義しています。  
例えば、直方体の各面に`Smooth concrete, painted or glazed`を使う場合は、直方体を選択した状態で右下パネルの`Material Properties`を開き、同じ名前を入力します。

<img width="1919" height="1092" alt="3_set_mat_param" src="https://github.com/user-attachments/assets/d9844dff-b9ee-4d16-9afb-706fc4385a2c" />

### 4. Mitsuba形式でのシーンのエクスポート

Blender画面左上の`File`→`Export`→`Mitsuba (.xml)`を選択します。  
`Mitsuba (.xml)`が表示されない場合は、[Mitsuba-Blenderアドオンのインストールガイド](https://github.com/mitsuba-renderer/mitsuba-blender/wiki/Installation-&-Update-Guide)を再確認してください。

<img width="1919" height="1090" alt="4_mitsuba_export_button" src="https://github.com/user-attachments/assets/b33ad8f1-9031-4d23-ace0-f50679bae5ba" />

`Export IDs`と`Ignore Default Background`を有効にし、`Y Forward`、`Z Up`を確認して保存します（例：`AcoustiX_room.xml`）。

<img width="1232" height="812" alt="5_export_settings" src="https://github.com/user-attachments/assets/22ef1b24-98f1-40fd-8a2f-b934c2dc2bb3" />

### 5. 出力ファイルの確認

エクスポートしたXMLファイルに加えて、同階層の`meshes/`配下にPLYファイルがあることを確認してください。  
シミュレーション実行時はXMLのパスを指定しますが、内部的にPLYファイルも参照するため、移動時はディレクトリ構成を保ってください。

---

## AIでシーンを生成する場合

[WorldGen](https://github.com/ZiYang-xie/WorldGen)を用いてAIでシーンファイルを作成することができます。

### 1. 動作環境

- Python 3.11  
- CUDA 12.8  
- Blender 3.6.0  
  https://download.blender.org/release/Blender3.6/
- Mitsuba-Blender v0.3.0  
  https://github.com/mitsuba-renderer/mitsuba-blender/releases/tag/v0.3.0

[Blender 3.6のダウンロードページ](https://download.blender.org/release/Blender3.6/)から自身の環境（OS/CPUアーキテクチャ）に応じてBlender 3.6.0をダウンロードし、インストールします。  
次に、[Mitsuba-Blender v0.3.0](https://github.com/mitsuba-renderer/mitsuba-blender/releases/tag/v0.3.0)のAssetsにある`mitsuba-blender.zip`をダウンロードし、[インストールガイド](https://github.com/mitsuba-renderer/mitsuba-blender/wiki/Installation-&-Update-Guide)に従ってMitsuba-Blenderアドオンを導入します。  
また、[Hugging Face](https://huggingface.co/)のアカウント及びRead権限を持つ[トークン](https://huggingface.co/settings/tokens)が必要なので、持っていない場合は作成します。

### 2. WorldGenの環境構築

次のコマンドを実行し、WorldGenの環境構築を行います。

```bash
# リポジトリのクローン
git clone https://github.com/ZiYang-xie/WorldGen.git 
cd WorldGen

# 仮想環境の作成
conda create -n worldgen python=3.11
conda activate worldgen

# torchとtorchvisionのインストール
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# WorldGenのインストール
pip install .

# pytorch3dをインストール
pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation
```

### 3. Hugging Faceの認証

[Hugging Face](https://huggingface.co/)にログインした上で、[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)のページにアクセスし、利用規約への同意とアクセス許可の取得を行います。  
その後、次のコマンドでCLIでのHugging Faceの認証を行います。

```bash
pip install -U huggingface_hub
hf auth login
```
途中で[トークン](https://huggingface.co/settings/tokens)の入力が求められるので、Read権限を持つトークンの入力をします。

### 4. WorldGenによるメッシュファイルの生成

次のコマンドで、WorldGenによるメッシュファイル（GLBファイル）の作成を行います。  
オプションの詳細は[demo.py](https://github.com/ZiYang-xie/WorldGen/blob/main/demo.py)を参照してください。`-r`は解像度を決定し、大きいほど精緻なメッシュファイルになりますが、メッシュの面の数が増え、音響シミュレーション時の負荷も増加します。

```bash
mkdir -p scene_output
python demo.py -p "a realistic furnished indoor room inside a fully enclosed cubic architectural shell, cubic outer shape, equal width depth and height, all four walls intact, flat floor, flat ceiling, closed box room, no missing wall, no open side, not a cutaway, not a cross-section, not a dollhouse view, orthogonal architecture, sofa and table, clean geometry, no curved exterior, no cylindrical exterior, no broken geometry" --return_mesh --save_scene -o scene_output -r 512
```

実行後、`scene_output/`配下に`mesh.glb`があることを確認します。

### 5. Blenderを用いたMitsuba形式でのシーンのエクスポート

Blenderを立ち上げ、画面右上の`Scene Collection`から`Camera`、`Cube`、`Light`を削除します。

<img width="1919" height="1093" alt="Blender start menu" src="https://github.com/user-attachments/assets/5aa6f87b-5f45-4329-9ab8-8a768a205cb8" />

Blender画面左上の`File`→`Import`→`glTF 2.0 (.glb/gltf)`を選択し、生成した`mesh.glb`をインポートします。  

<img width="1915" height="1127" alt="Import mesh.glb" src="https://github.com/user-attachments/assets/1de82a61-2d86-4eef-b1a6-d653fa0336d2" />

<img width="1919" height="1090" alt="WorldGen mesh" src="https://github.com/user-attachments/assets/4376df35-8e3f-4fef-82d5-aeed27b554d6" />

Blender画面左上の`File`→`Export`→`Mitsuba (.xml)`を選択します。  
`Mitsuba (.xml)`が表示されない場合は、[Mitsuba-Blenderアドオンのインストールガイド](https://github.com/mitsuba-renderer/mitsuba-blender/wiki/Installation-&-Update-Guide)を再確認してください。

<img width="1914" height="1126" alt="Mitsuba export" src="https://github.com/user-attachments/assets/c4447b51-acc4-477b-8fd8-60fa3d7e3261" />

`Export IDs`と`Ignore Default Background`を有効にし、`Y Forward`、`Z Up`を確認して保存します（例：`worldgen_scene.xml`）。

<img width="1233" height="817" alt="export Mitsuba xml" src="https://github.com/user-attachments/assets/db892fd1-f168-4b40-a937-be3900d42753" />


### 6. 出力ファイルの確認

エクスポートしたXMLファイルに加えて、同階層の`meshes/`配下にPLYファイルがあることを確認してください。  
シミュレーション実行時はXMLのパスを指定しますが、内部的にPLYファイルも参照するため、移動時はディレクトリ構成を保ってください。

---

## ライセンス

本リポジトリは MIT License に従って公開されています。詳細は[`LICENSE`](https://github.com/KMASAHIRO/multichannel-soundfields/blob/main/AcoustiX/LICENSE)を確認してください。
