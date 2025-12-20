# Model Backup and Usage Guide

## モデルのバックアップと他のマシンでの使用

### 1. モデルをローカルに保存

```bash
python save_model_locally.py
```

これで `/home/ubuntu-user-2404/workspace/youtube-to-text/models/bert-base-japanese-char/` にモデルとトークナイザーが保存されます。

### 2. 他の場所にバックアップ

```bash
# ディレクトリ全体をコピー
cp -r /home/ubuntu-user-2404/workspace/youtube-to-text/models/bert-base-japanese-char /path/to/backup/

# またはtar.gzで圧縮して転送
tar -czf bert-base-japanese-char.tar.gz -C /home/ubuntu-user-2404/workspace/youtube-to-text/models bert-base-japanese-char
```

### 3. 他のマシンで使用

#### 方法A: main.pyでローカルパスを使用

`main.py` の `BERT_MODEL_PATH` を設定:

```python
BERT_MODEL_PATH = "/path/to/your/local/model/bert-base-japanese-char"
```

#### 方法B: コード内で直接指定

```python
# HuggingFaceモデル名を使用（インターネット接続が必要）
nlp = pipeline("fill-mask", model="cl-tohoku/bert-base-japanese-char")

# またはローカルパスを使用（インターネット接続不要）
nlp = pipeline("fill-mask", model="/path/to/local/model/bert-base-japanese-char")
```

### 4. 他のマシンへの転送手順

```bash
# 1. モデルを圧縮
tar -czf bert-model-backup.tar.gz -C /home/ubuntu-user-2404/workspace/youtube-to-text/models bert-base-japanese-char

# 2. 他のマシンに転送（SCPの例）
scp bert-model-backup.tar.gz user@other-machine:/path/to/destination/

# 3. 他のマシンで展開
tar -xzf bert-model-backup.tar.gz

# 4. main.pyでパスを指定
BERT_MODEL_PATH = "/path/to/destination/bert-base-japanese-char"
```

### 5. 機械学習での使用例

```python
from transformers import AutoModel, AutoTokenizer, pipeline

# ローカルパスからモデルをロード
model_path = "/path/to/local/model/bert-base-japanese-char"

# 方法1: pipelineを使用
nlp = pipeline("fill-mask", model=model_path)

# 方法2: 個別にモデルとトークナイザーをロード（ファインチューニングなど）
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# 使用例
result = nlp("これは[MASK]です。")
print(result)
```

### 注意事項

- モデルファイルは通常数百MB〜数GBのサイズがあります
- モデルディレクトリには以下のファイルが含まれます:
  - `config.json` - モデル設定
  - `pytorch_model.bin` または `model.safetensors` - モデルウェイト
  - `tokenizer.json`, `vocab.txt` など - トークナイザーファイル










