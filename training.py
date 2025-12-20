import torch
import os
import shutil
from datetime import datetime
from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments

from CustomTrainer import CustomTrainer

label_list = ["O", "COMMA", "PERIOD", "QUESTION"]

# パス設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "bert-base-japanese-char")
SOURCE_DIR = os.path.join(BASE_DIR, "source")


def backup_model(model_dir: str) -> str:
    """
    モデルをバックアップする
    models/bert-base-japanese-char -> models/bert-base-japanese-char-yyyymmdd-x
    
    Args:
        model_dir: バックアップするモデルのディレクトリパス
        
    Returns:
        バックアップ先のパス
    """
    if not os.path.exists(model_dir):
        print(f"警告: モデルディレクトリが見つかりません: {model_dir}")
        return None
    
    date_str = datetime.now().strftime("%Y%m%d")
    base_backup_name = f"bert-base-japanese-char-{date_str}"
    
    models_base_dir = os.path.dirname(model_dir)
    
    max_num = 0
    for item in os.listdir(models_base_dir):
        if item.startswith(base_backup_name + "-"):
            try:
                num = int(item.split("-")[-1])
                max_num = max(max_num, num)
            except ValueError:
                continue
    
    backup_num = max_num + 1
    backup_name = f"{base_backup_name}-{backup_num}"
    backup_path = os.path.join(models_base_dir, backup_name)
    
    print(f"モデルをバックアップ中: {model_dir} -> {backup_path}")
    shutil.copytree(model_dir, backup_path)
    print(f"バックアップ完了: {backup_path}")
    
    return backup_path


def read_training_data(source_dir: str, split_by_empty_line: bool = True):
    """
    source/before.txt と source/after.txt から学習データを読み込む
    
    Args:
        source_dir: sourceディレクトリのパス
        split_by_empty_line: Trueの場合、空行で分割して複数行を1サンプルとして扱う。
                            Falseの場合、1行を1サンプルとして扱う（従来の動作）
        
    Returns:
        {'text': [...], 'after': [...]} 形式の辞書
    """
    before_path = os.path.join(source_dir, "before.txt")
    after_path = os.path.join(source_dir, "after.txt")
    
    # ファイルが存在するか確認
    if not os.path.exists(before_path):
        print(f"警告: {before_path} が見つかりません。空のデータセットを返します。")
        return {"text": [], "after": []}
    
    if not os.path.exists(after_path):
        print(f"警告: {after_path} が見つかりません。空のデータセットを返します。")
        return {"text": [], "after": []}
    
    # ファイルを読み込み
    with open(before_path, "r", encoding="utf-8") as f:
        before_content = f.read()
    
    with open(after_path, "r", encoding="utf-8") as f:
        after_content = f.read()

    # 分割方法を選択
    if split_by_empty_line:
        before_samples = [' '.join(line.strip() for line in before_content.split('\n') if line.strip())]
        after_samples = [' '.join(line.strip() for line in after_content.split('\n') if line.strip())]

        print(f"空行で分割: before.txt -> {len(before_samples)}サンプル, after.txt -> {len(after_samples)}サンプル")
    else:
        before_samples = [line.strip() for line in before_content.split('\n') if line.strip()]
        after_samples = [line.strip() for line in after_content.split('\n') if line.strip()]
        print(f"1行1サンプル: before.txt -> {len(before_samples)}サンプル, after.txt -> {len(after_samples)}サンプル")
    
    # サンプル数が一致するか確認
    if len(before_samples) != len(after_samples):
        print(f"警告: before.txt ({len(before_samples)}サンプル) と after.txt ({len(after_samples)}サンプル) のサンプル数が一致しません。")
        min_len = min(len(before_samples), len(after_samples))
        before_samples = before_samples[:min_len]
        after_samples = after_samples[:min_len]
        print(f"最初の {min_len} サンプルを使用します。")
    
    return {"text": before_samples, "after": after_samples}


# List of dictionaries to dictionary of lists conversion
def convert_to_dict(data_list):
    dict_data = {"text": [], "after": []}
    for item in data_list:
        dict_data["text"].append(item["text"])
        dict_data["after"].append(item["after"])
    return dict_data


def align_punctuation(before_text, after_text):
    """
    "before"と"after"を文字単位でアライメントして、句読点の位置をマッピング
    
    Returns:
        dict: {before_position: punctuation_char} のマッピング
    """
    before_chars = list(before_text)
    after_chars = list(after_text)
    
    punct_map = {}  # {before_position: punctuation_char}
    before_idx = 0
    after_idx = 0
    
    while before_idx < len(before_chars) and after_idx < len(after_chars):
        if after_chars[after_idx] in ["、", "。", "？"]:
            # "after"に句読点がある → 前の"before"文字（before_idx-1）の後に来る
            if before_idx > 0:
                punct_map[before_idx - 1] = after_chars[after_idx]
            after_idx += 1
        elif before_chars[before_idx] == after_chars[after_idx]:
            # 同じ文字 → 次へ
            before_idx += 1
            after_idx += 1
        else:
            # 文字が一致しない（通常は発生しない）
            before_idx += 1
            after_idx += 1
    
    # 最後の文字の後に句読点がある場合
    while after_idx < len(after_chars):
        if after_chars[after_idx] in ["、", "。", "？"]:
            if len(before_chars) > 0:
                punct_map[len(before_chars) - 1] = after_chars[after_idx]
            after_idx += 1
        else:
            break
    
    return punct_map


os.makedirs(SOURCE_DIR, exist_ok=True)

# アライメント関数のテスト
print("=== アライメント関数のテスト ===")
test_before = "今朝のコメンテーター経済学者"
test_after = "今朝のコメンテーター、経済学者"
test_punct_map = align_punctuation(test_before, test_after)
print(f"before: {test_before}")
print(f"after: {test_after}")
print(f"punct_map: {test_punct_map}")
print(f"期待: 位置9('ー')に'、'がマッピングされる")
if 9 in test_punct_map and test_punct_map[9] == "、":
    print("✓ アライメント関数は正常に動作しています")
else:
    print("✗ アライメント関数に問題があります！")
print("=" * 50)

if os.path.exists(MODEL_DIR):
    backup_model(MODEL_DIR)
else:
    print(f"警告: モデルディレクトリが見つかりません: {MODEL_DIR}")
    print("バックアップをスキップします。")

print(f"学習データを読み込み中: {SOURCE_DIR}")
# split_by_empty_line=False にすることで、1行を1サンプルとして扱う
# before.txt: 71行、after.txt: 118行 → 71サンプルを使用（最小値）
training_data = read_training_data(SOURCE_DIR, split_by_empty_line=False)

if not training_data["text"]:
    print("警告: source/before.txt と source/after.txt からデータを読み込めませんでした。")
    print("デフォルトのサンプルデータを使用します。")
    # 例として、簡単なデータセットを作成
    data = {
    'train': [
        {
            "text": '友達に聞いたとその友達は誰に聞いたのって自分に聞いてたりするんですよこれ陰謀論あるあるなん'
            'ですよねそれ言っていくとどんどん論点ずらして無限に拡散していくんですよねこれが陰謀論の特徴なんですよ',
            "after": '友達に聞いたと、その友達は誰に聞いたのって自分に聞いてたりするんですよ。これ陰謀論あるあるなん'
            'ですよね。それ言っていくと、どんどん論点ずらして無限に拡散していくんですよね。これが陰謀論の特徴なんですよ。'
        }, {
            "text": 'だから基本的にはそういうことですけど特別にくっつく物質あるの?ないんですよ水だからトリチウム水なんだもん普通に排出されるんですよ',
            "after": 'だから基本的にはそういうことですけど、特別にくっつく物質あるの?ないんですよ。水だからトリチウム水なんだもん。普通に排出されるんですよ。'
        }
    ],
    'test': [
        {
            "text": '自民党の総裁選をめぐり読売新聞は党員党友への電話調査と国会議員の支持動向調査を行いました両調査の結果を合計すると高市経済安全保障大臣と',
            "after": '自民党の総裁選をめぐり、読売新聞は党員党友への電話調査と国会議員の支持動向調査を行いました。両調査の結果を合計すると、高市経済安全保障大臣と'
        }, {
            "text": 'これだけの政策論争だ国民はどのように判断しようか考えながら見ていると訴えました衆議院解散前の国会論戦ですがリーダーとしていかがなもんですか', 
            "after": 'これだけの政策論争だ、国民はどのように判断しようか考えながら見ている、と訴えました。衆議院解散前の国会論戦ですが、リーダーとしていかがなもんですか？'}
    ]
    }
    # デフォルトデータを使用
    train_dict = convert_to_dict(data['train'])
    test_dict = convert_to_dict(data['test'])
else:
    # 読み込んだデータを使用（80%を訓練、20%をテストに分割）
    all_text = training_data["text"]
    all_after = training_data["after"]
    
    split_idx = int(len(all_text) * 0.8)
    # 訓練データが少なくとも1件になるようにする
    if split_idx == 0 and len(all_text) > 0:
        split_idx = 1
    
    train_text = all_text[:split_idx]
    train_after = all_after[:split_idx]
    test_text = all_text[split_idx:]
    test_after = all_after[split_idx:]
    
    train_dict = {"text": train_text, "after": train_after}
    test_dict = {"text": test_text, "after": test_after}
    
    print(f"学習データ: {len(train_text)}件")
    print(f"テストデータ: {len(test_text)}件")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    labels = []
    for i, (before_text, after_text) in enumerate(zip(examples['text'], examples['after'])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        label_ids = []
        # "before"と"after"を文字単位でアライメントして、句読点の位置をマッピング
        # word_idは"before"の文字位置を表す
        # "after"の句読点が"before"のどの文字の後に来るかを判定
        
        punct_map = align_punctuation(before_text, after_text)

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                try:
                    # word_id位置の後に句読点があるかチェック
                    # word_idは"before"テキストの文字位置を表す
                    if word_id in punct_map:
                        punct = punct_map[word_id]
                        if punct == "、":
                            label_ids.append(label_list.index("COMMA"))
                        elif punct == "。":
                            label_ids.append(label_list.index("PERIOD"))
                        elif punct == "？":
                            label_ids.append(label_list.index("QUESTION"))
                        else:
                            label_ids.append(label_list.index("O"))
                    else:
                        label_ids.append(label_list.index("O"))
                except (ValueError, IndexError) as e:
                    # エラーが発生した場合はOラベルを付与
                    label_ids.append(label_list.index("O"))

        labels.append(label_ids)
        
        # デバッグ: 最初のサンプルでラベル分布を確認
        if i == 0:
            label_counts = {}
            for lid in label_ids:
                if lid != -100:
                    label_name = label_list[lid] if lid < len(label_list) else "UNKNOWN"
                    label_counts[label_name] = label_counts.get(label_name, 0) + 1
            print(f"\n=== デバッグ: サンプル0 ===")
            print(f"ラベル分布: {label_counts}")
            print(f"punct_map (位置: 句読点): {punct_map}")
            print(f"before_text (最初の100文字): {before_text[:100]}")
            print(f"after_text (最初の100文字): {after_text[:100]}")
            print(f"before長さ: {len(before_text)}, after長さ: {len(after_text)}")
            # word_idsの最初の20個を表示
            print(f"最初の20個のword_ids: {word_ids[:20]}")
            # punct_mapにマッピングされているword_idを確認
            mapped_word_ids = [wid for wid in word_ids[:50] if wid is not None and wid in punct_map]
            print(f"最初の50トークンでpunct_mapにマッピングされているword_id: {mapped_word_ids}")
            print("=" * 50)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def predict(text):
    # テキストをトークナイズ
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = inputs.to(device)

    # モデルを評価モードに
    model.eval()

    # 推論の実行
    with torch.no_grad():
        outputs = model(**inputs)

    # 予測ラベルを取得
    predictions = torch.argmax(outputs.logits, dim=2)

    # トークンとラベルの対応を取得
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [label_list[prediction] for prediction in predictions[0].tolist()]

    # ラベルに基づいて句読点を挿入
    punctuated_text = ""
    for token, label in zip(tokens, labels):
        if label == "COMMA":
            punctuated_text += token + "、"
        elif label == "PERIOD":
            punctuated_text += token + "。"
        elif label == "QUESTION":
            punctuated_text += token + "？"
        else:
            punctuated_text += token

    return punctuated_text


train_data = Dataset.from_dict(train_dict)
test_data = Dataset.from_dict(test_dict)

if os.path.exists(MODEL_DIR):
    print(f"ローカルモデルをロード中: {MODEL_DIR}")
    model_name = MODEL_DIR
else:
    print(f"警告: ローカルモデルが見つかりません: {MODEL_DIR}")
    print("HuggingFaceからダウンロードします: cl-tohoku/bert-base-japanese-char")
    model_name = "cl-tohoku/bert-base-japanese-char"

tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

# set device to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataset = train_data.map(tokenize_and_align_labels, batched=True)
test_dataset = test_data.map(tokenize_and_align_labels, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    remove_unused_columns=False
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# モデルのトレーニング
trainer.train()

# モデルの評価
results = trainer.evaluate()
print(results)

# 訓練済みモデルとトークナイザーを保存
print(f"\n訓練済みモデルを保存中: {MODEL_DIR}")
os.makedirs(MODEL_DIR, exist_ok=True)
# モデルのconfigにnum_labelsを明示的に設定
model.config.num_labels = len(label_list)
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"モデルとトークナイザーの保存完了: {MODEL_DIR}")

# テストするテキスト
texts = [
    '友達に聞いたとその友達は誰に聞いたのって自分に聞いてたりするんですよこれ陰謀論あるあるなん'
    'ですよねそれ言っていくとどんどん論点ずらして無限に拡散していくんですよねこれが陰謀論の特徴なんですよ',
    'だから基本的にはそういうことですけど特別にくっつく物質あるの?ないんですよ水だからトリチウム水なんだもん普通に排出されるんですよ',
    '自民党の総裁選をめぐり読売新聞は党員党友への電話調査と国会議員の支持動向調査を行いました両調査の結果を合計すると高市経済安全保障大臣と',
    'これだけの政策論争だ国民はどのように判断しようか考えながら見ていると訴えました衆議院解散前の国会論戦ですがリーダーとしていかがなもんですか'
]

# 各テキストに対して予測を実行
for text in texts:
    prediction = predict(text)
    print(f"Original: {text} | Punctuated: {prediction}")
