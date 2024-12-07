import torch
from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments

from CustomTrainer import CustomTrainer

label_list = ["O", "COMMA", "PERIOD", "QUESTION"]

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


# List of dictionaries to dictionary of lists conversion
def convert_to_dict(data_list):
    dict_data = {"text": [], "after": []}
    for item in data_list:
        dict_data["text"].append(item["text"])
        dict_data["after"].append(item["after"])
    return dict_data


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    labels = []
    for i, label in enumerate(examples['after']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                try:
                    if word_id < len(label):
                        char = label[word_id]
                        if char == "、":
                            label_ids.append(label_list.index("COMMA"))
                        elif char == "。":
                            label_ids.append(label_list.index("PERIOD"))
                        elif char == "？":
                            label_ids.append(label_list.index("QUESTION"))
                        else:
                            label_ids.append(label_list.index("O"))
                    else:
                        label_ids.append(label_list.index("O"))
                except ValueError:
                    label_ids.append(0)

        labels.append(label_ids)

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


# Convert train and test data
train_dict = convert_to_dict(data['train'])
test_dict = convert_to_dict(data['test'])

# データセットを作成
train_data = Dataset.from_dict(train_dict)
test_data = Dataset.from_dict(test_dict)

# モデルとトークナイザーのロード
model_name = "cl-tohoku/bert-base-japanese"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))
# print(f"type of train_data: {type(train_data)}")
# print(f"train_data: {train_data}")

# set device to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataset = train_data.map(tokenize_and_align_labels, batched=True)
test_dataset = test_data.map(tokenize_and_align_labels, batched=True)

# print(f"train_dataset: {train_dataset}")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
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
