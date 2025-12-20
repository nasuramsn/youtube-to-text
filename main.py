import array
import yt_dlp
import whisper
import os
import numpy as np
import soundfile as sf
import librosa
import time
import sys
import torch

from transformers import BertTokenizerFast, BertForTokenClassification

# args
# 1: YoutubeのURLの番号
# 2: is_do_download_youtube
# 3: is_do_download_whisper
# 4: is_do_output_to_text
# 5: is_do_output_punctuation
# example: qIW9NxF34Jo True True True True

args = sys.argv
video_url_org = "https://www.youtube.com/watch?v="
video_url = video_url_org + args[1]

is_do_download_youtube = True if args[2] == "True" else False
is_do_download_whisper = True if args[3] == "True" else False
is_do_output_to_text = True if args[4] == "True" else False
is_do_output_punctuation = True if args[5] == "True" else False

print(f"args: {args}")
print(f"args[2]: {args[2]}")
print(f"args[2]: {is_do_download_youtube}")

# ダウンロードディレクトリ
download_dir = r"/home/ubuntu-user-2404/workspace/youtube/medias"
output_path = os.path.join(download_dir, "transcription.txt")

# モデルパス設定
# ローカルパスを使用する場合: "/path/to/local/model"
# HuggingFaceからダウンロードする場合: "cl-tohoku/bert-base-japanese-char"
# または None でデフォルトを使用
BERT_MODEL_PATH = "/home/ubuntu-user-2404/workspace/youtube-to-text/models/bert-base-japanese-char"

# Whisperモデルのダウンロード/キャッシュディレクトリ設定
# Noneの場合はデフォルトの ~/.cache/whisper/ を使用
# カスタムパスを指定する場合: "/path/to/whisper/cache"
WHISPER_DOWNLOAD_ROOT = "/home/ubuntu-user-2404/workspace/youtube-to-text/models/whisper"
# WHISPER_DOWNLOAD_ROOT = None  # デフォルト: ~/.cache/whisper/

# MP3ファイルの保存パス
# audio_file = os.path.join(download_dir, "audio")
audio_file_org = "/home/ubuntu-user-2404/workspace/youtube/medias/" + args[1]
audio_file = audio_file_org + ".mp3"

# yt-dlp設定
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': audio_file_org + '.%(ext)s',  # 拡張子は自動で追加されます
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'ffmpeg_location': '/usr/bin/ffmpeg'  # ここでffmpegの場所を指定
}


def download_youtube(ydl_opts, video_url: str, audio_file: str):
    print("YouTube動画をダウンロードしてMP3に変換しています...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print(f"ダウンロードが完了しました: {audio_file}")
    except Exception as e:
        print(f"ダウンロード中にエラーが発生しました: {e}")


def check_exist_mp3(audio_file: str) -> bool:
    if not os.path.isfile(audio_file):
        return False
    else:
        return True


def load_wisper_model(download_root=None) -> array:
    """
    Whisperモデルをロードする
    注意: Whisperモデルは自動的にダウンロードされ、~/.cache/whisper/ に保存されます
    BERTモデル（句読点予測用）とは別物です
    
    Args:
        download_root: Whisperモデルのダウンロード/キャッシュディレクトリ
                      Noneの場合はデフォルトの~/.cache/whisper/を使用
                      カスタムパスを指定すると、そのディレクトリにモデルを保存
    """
    import traceback
    from pathlib import Path
    
    print("Whisperモデルをロードしています...")
    print("注意: Whisperモデルは音声認識用のモデルです（BERTモデルとは別物）")
    
    # キャッシュディレクトリの設定
    if download_root is None:
        # デフォルトのキャッシュディレクトリ
        download_root = Path.home() / ".cache" / "whisper"
    else:
        download_root = Path(download_root)
        # ディレクトリが存在しない場合は作成
        download_root.mkdir(parents=True, exist_ok=True)
    
    # キャッシュディレクトリの情報を表示
    print(f"Whisperモデルのキャッシュディレクトリ: {download_root}")
    
    # モデルファイルが既に存在するか確認
    model_name = "large"
    model_path = download_root / f"{model_name}.pt"
    if model_path.exists():
        print(f"✓ キャッシュからモデルを読み込みます: {model_path}")
        print("  (初回のみダウンロードが必要です。2回目以降はキャッシュから読み込まれます)")
    else:
        print(f"⚠ モデルが見つかりません。初回ダウンロードを開始します...")
        print(f"  ダウンロード先: {model_path}")
        print("  (このダウンロードは初回のみです。次回からはキャッシュから読み込まれます)")
    
    try:
        # download_rootパラメータを使用してモデルをロード
        model = whisper.load_model("large", device="cpu", download_root=str(download_root))
        _ = model.half()
        _ = model.cuda()

        for m in model.modules():
            if isinstance(m, whisper.model.LayerNorm):
                m.float()
        print("✓ Whisperモデルが正しくロードされました。")
        return [True, model]
    except Exception as e:
        error_msg = f"Whisperモデルのロード中にエラーが発生しました: {e}"
        print(error_msg)
        print(f"エラータイプ: {type(e).__name__}")
        print("トレースバック:")
        traceback.print_exc()
        return [False, e]


def transcription(model, output_path: str, data_resampled: array, chunk_size: int, language: str) -> bool:
    # 音声ファイルをチャンクに分割して処理
    # 英語で文字起こししたいときはlanguage='en'にする。
    import traceback
    
    # モデルの存在確認
    if model is None:
        error_msg = "エラー: WhisperモデルがNoneです。モデルが正しくロードされていません。"
        print(error_msg)
        print(f"詳細: is_do_download_whisperがTrueの場合、load_wisper_model()が成功しているか確認してください。")
        return False
    
    text_start = time.time()
    print("音声ファイルをチャンクに分割して文字起こしを開始します...")
    print(f"データサイズ: {len(data_resampled)}, チャンクサイズ: {chunk_size}, チャンク数: {(len(data_resampled) + chunk_size - 1) // chunk_size}")
    
    for i in range(0, len(data_resampled), chunk_size):
        chunk = data_resampled[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        try:
            print(f"チャンク {chunk_num} を処理中... (インデックス: {i} - {min(i + chunk_size, len(data_resampled))})")
            result = model.transcribe(
                chunk,
                verbose=True,
                language=language,
                fp16=True,
                without_timestamps=True)
            chunk_transcription = result['text']
            print(f"チャンク {chunk_num} の文字起こしが完了しました。")

            # 部分的な文字起こし結果をファイルに追記
            with open(output_path, "a", encoding="utf-8") as file:
                file.write(chunk_transcription + "\n")
        except AttributeError as e:
            error_msg = f"チャンク {chunk_num} の文字起こし中にAttributeErrorが発生しました: {e}"
            print(error_msg)
            print(f"エラータイプ: {type(e).__name__}")
            print(f"エラーメッセージ: {str(e)}")
            print(f"モデルの型: {type(model)}")
            print(f"モデルがNoneか: {model is None}")
            print("トレースバック:")
            traceback.print_exc()
            return False
        except Exception as e:
            error_msg = f"チャンク {chunk_num} の文字起こし中にエラーが発生しました: {e}"
            print(error_msg)
            print(f"エラータイプ: {type(e).__name__}")
            print(f"エラーメッセージ: {str(e)}")
            print("トレースバック:")
            traceback.print_exc()
            return False

    text_end = time.time()
    text_time = text_end - text_start
    print(f"Text出力時間: {text_time}")
    print(f"文字起こし結果が {output_path} に保存されました。")

    return True


def convert_audio_to_monoral(audio_file: str, model) -> array:
    print("オーディオファイルを読み込みます...")
    try:
        data, samplerate = sf.read(audio_file)
        print(f"オーディオファイルが正常に読み取られました。サンプルレート: {samplerate}, サイズ: {data.shape} バイト")
    except Exception as e:
        return [False, e]
        # raise RuntimeError(f"オーディオファイルを開く際にエラーが発生しました: {e}")

    convert_start = time.time()

    # サンプルレートを16kHzに変更し、モノラルに変換
    print("オーディオを16kHzモノラルに変換します...")
    try:
        data_mono = librosa.to_mono(data.T)
        data_resampled = librosa.resample(data_mono, orig_sr=samplerate, target_sr=16000)
        data_resampled = data_resampled.astype(np.float32)
        print(f"変換が完了しました。新しいサイズ: {data_resampled.shape}")
    except Exception as e:
        # raise RuntimeError(f"オーディオ変換中にエラーが発生しました: {e}")
        return [False, e]
    convert_end = time.time()
    convert_time = convert_end - convert_start
    print(f"変換時間: {convert_time}")
    return [True, data_resampled]


def insert_char_to_sentence(i: int, char: str, sentence: str) -> str:
    # sentenceのi文字目にcharを挿入する
    l: array = list(sentence)
    l.insert(i, char)
    if char == "。":
        l.insert(i + 1, "\n")
    text = "".join(l)
    return text


def add_punctuation_mark(input_path: str, model_path: str = None, disable_question: bool = True) -> str:
    # 句読点を入れる（トークン分類モデルを使用）
    # model_path: ローカルパスまたはHuggingFaceモデル名
    #   例: "/path/to/local/model" または "cl-tohoku/bert-base-japanese-char"
    #   デフォルト: "cl-tohoku/bert-base-japanese-char"
    # disable_question: Trueの場合、QUESTIONラベルを無視（訓練データが少ない場合に有効）
    
    label_list = ["O", "COMMA", "PERIOD", "QUESTION"]
    
    # モデルパスが指定されていない場合はデフォルトを使用
    if model_path is None:
        model_path = "cl-tohoku/bert-base-japanese-char"
    
    print(f"モデルをロード中: {model_path}")
    
    # トークン分類モデルとトークナイザーをロード
    # num_labels=4を明示的に指定（config.jsonにnum_labelsが保存されていない場合の対策）
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForTokenClassification.from_pretrained(model_path, num_labels=4)
    
    # デバッグ: モデルの設定を確認
    print(f"モデル設定: num_labels={model.config.num_labels}")
    print(f"期待されるラベル: {label_list}")
    
    # デバッグ: 分類器の重みが存在するか確認
    if hasattr(model, 'classifier'):
        print(f"分類器の重み形状: {model.classifier.weight.shape if hasattr(model.classifier, 'weight') else 'N/A'}")
        print(f"分類器のバイアス形状: {model.classifier.bias.shape if hasattr(model.classifier, 'bias') else 'N/A'}")
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    result = ""
    print(f"input_path: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            original_sentence: str = line.strip()
            if not original_sentence:
                result += "\n"
                continue
            
            # テキストをトークナイズ
            # word_idsを取得するために、まずエンコーディングを取得
            tokenized = tokenizer(original_sentence, truncation=True, max_length=512)
            word_ids = tokenized.word_ids()
            
            # モデル用の入力を準備
            inputs = tokenizer(original_sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 推論の実行
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 予測ラベルを取得
            predictions = torch.argmax(outputs.logits, dim=2)
            labels = [label_list[prediction] for prediction in predictions[0].tolist()]
            
            # デバッグ: 予測ラベルの分布を確認
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            print(f"予測ラベル分布: {label_counts}")
            
            # 文字列に句読点を挿入
            corrected_chars = list(original_sentence)
            insertions = []
            
            # word_idsとlabelsを対応させる
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # デバッグ: 最初の10個のトークンとword_idsを表示
            print(f"最初の10トークン: {tokens[:10]}")
            print(f"最初の10word_ids: {word_ids[:10]}")
            print(f"最初の10ラベル: {labels[:10]}")
            
            for i, (word_id, label, token) in enumerate(zip(word_ids, labels, tokens)):
                if word_id is None:  # 特殊トークン（[CLS], [SEP], [PAD]）をスキップ
                    continue
                if word_id >= len(original_sentence):  # 範囲外をスキップ
                    continue
                
                # ラベルに基づいて句読点を挿入位置として記録
                # word_idは文字位置を表す（bert-base-japanese-charは文字レベルのトークナイザー）
                # 信頼度を確認してから挿入
                probs = torch.softmax(outputs.logits[0][i], dim=0)
                label_prob = probs[label_list.index(label)].item()
                
                # デバッグ: 非Oラベルの場合、詳細を表示
                if label != "O" and i < 20:  # 最初の20個の非Oラベルのみ表示
                    max_prob = torch.max(probs).item()
                    max_label_idx = torch.argmax(probs).item()
                    max_label = label_list[max_label_idx]
                    print(f"非Oラベル発見: 位置={i}, word_id={word_id}, 予測={label}({label_prob:.3f}), 最大={max_label}({max_prob:.3f})")
                
                # 信頼度の閾値を下げる（訓練データが少ないため、一時的に低く設定）
                # デバッグモード: 信頼度チェックを緩くする
                use_confidence = False  # Trueにすると信頼度チェックあり、Falseで無条件に挿入
                min_confidence = 0.8 if label == "QUESTION" else 0.2  # 0.3から0.2に下げる
                
                if label == "COMMA" and (not use_confidence or label_prob > min_confidence):
                    insertions.append((word_id + 1, "、"))
                    print(f"COMMA挿入: 位置={word_id+1}, 信頼度={label_prob:.3f}, 文字={original_sentence[word_id] if word_id < len(original_sentence) else 'N/A'}")
                elif label == "PERIOD" and (not use_confidence or label_prob > min_confidence):
                    insertions.append((word_id + 1, "。"))
                    print(f"PERIOD挿入: 位置={word_id+1}, 信頼度={label_prob:.3f}, 文字={original_sentence[word_id] if word_id < len(original_sentence) else 'N/A'}")
                elif label == "QUESTION" and (not use_confidence or label_prob > min_confidence) and not disable_question:
                    insertions.append((word_id + 1, "？"))
                    print(f"QUESTION挿入: 位置={word_id+1}, 信頼度={label_prob:.3f}")
                # OラベルとQUESTION（無効化時）は何もしない
            
            print(f"挿入予定: {len(insertions)}箇所")
            
            # 後ろから前に挿入（位置がずれないように）
            insertions.sort(reverse=True, key=lambda x: x[0])
            for pos, punct in insertions:
                if pos <= len(corrected_chars) and (pos == 0 or corrected_chars[pos-1] not in ["、", "。", "？"]):
                    corrected_chars.insert(pos, punct)
            
            corrected_sentence = "".join(corrected_chars)
            
            print(f"original_sentence: {original_sentence}")
            print(f"corrected_sentence: {corrected_sentence}")
            result += corrected_sentence + "\n"
    
    return result


def export_result_sentence(input_path: str, output_path: str, model_path: str = None, disable_question: bool = True) -> bool:
    # 句読点入りの文字を出力する
    # model_path: ローカルパスまたはHuggingFaceモデル名
    # disable_question: QUESTIONラベルを無視するかどうか
    punctuation_start = time.time()
    result_sentence: str = add_punctuation_mark(input_path, model_path=model_path, disable_question=disable_question)
    with open(output_path, "a", encoding="utf-8") as file:
        file.write(result_sentence)
    punctuation_end = time.time()
    punctuation_time = punctuation_end - punctuation_start
    print(f"句読点追加時間: {punctuation_time}")
    print(f"句読点付き文字起こし結果が {output_path} に保存されました。")
    return True


# main
print(f"is_do_download_youtube: {is_do_download_youtube}")
if is_do_download_youtube:
    # YouTube動画をMP3に変換してダウンロード
    download_youtube(ydl_opts, video_url, audio_file_org)

    # MP3ファイルの存在確認
    if not check_exist_mp3(audio_file):
        raise RuntimeError(f"ファイルが見つかりません: {audio_file}")

    # ファイルアクセス権限の確認
    try:
        with open(audio_file, 'rb') as f:
            print("ファイルにアクセスできます。")
    except Exception as e:
        raise RuntimeError(f"ファイルにアクセスできません: {e}")

print(f"is_do_download_whisper: {is_do_download_whisper}")
model = None
if is_do_download_whisper:
    # Whisperモデルをロード（ダウンロード/キャッシュディレクトリを指定可能）
    # モデルは初回のみダウンロードされ、以降はキャッシュから読み込まれます
    load_whisper_result = load_wisper_model(download_root=WHISPER_DOWNLOAD_ROOT)
    if not load_whisper_result[0]:
        # raise RuntimeError("Wisperモデルをダウンロード出来ませんでした")
        raise RuntimeError(f"Whisperモデルのロード中にエラーが発生しました: {load_whisper_result[1]}")
    else:
        model = load_whisper_result[1]

print(f"is_do_output_to_text: {is_do_output_to_text}")
if is_do_output_to_text:
    # オーディオファイルを読み込む
    convert_audio_result = convert_audio_to_monoral(audio_file, model)
    if not convert_audio_result:
        raise RuntimeError("オーディオの変換に失敗しました")
    else:
        data_resampled = convert_audio_result[1]

    # チャンクサイズを設定（例: 60秒ごとに分割）
    chunk_size = 60 * 16000  # 60秒 * 16kHz

    # テキストファイルの保存先ディレクトリ
    os.makedirs(download_dir, exist_ok=True)

    # 既存のファイルがあれば削除
    if os.path.exists(output_path):
        os.remove(output_path)

    # モデルの存在確認
    if model is None:
        error_msg = "エラー: 文字起こしにはWhisperモデルが必要ですが、モデルがロードされていません。"
        print(error_msg)
        print("=" * 60)
        print("重要: WhisperモデルとBERTモデルは別物です")
        print("- Whisperモデル: 音声認識用（音声 → テキスト）")
        print("  → whisperライブラリが自動的にダウンロードします")
        print("  → 通常は ~/.cache/whisper/ に保存されます")
        print("- BERTモデル: 句読点予測用（テキスト → テキスト）")
        print(f"  → {BERT_MODEL_PATH}")
        print("=" * 60)
        print(f"is_do_download_whisper={is_do_download_whisper}")
        print("文字起こしを実行するには、is_do_download_whisper=Trueに設定してください。")
        print("例: python3 main.py <video_id> False True True True")
        raise RuntimeError(error_msg)

    # 文字起こし開始
    result: bool = transcription(model, output_path, data_resampled, chunk_size, 'ja')

# 句読点を入れる
print(f"is_do_output_punctuation: {is_do_output_punctuation}")
if is_do_output_punctuation:
    output_punctuation_path: str = os.path.join(download_dir, "punctuation_mark.txt")
    if os.path.exists(output_punctuation_path):
        os.remove(output_punctuation_path)
    result: bool = export_result_sentence(output_path, output_punctuation_path, model_path=BERT_MODEL_PATH)
