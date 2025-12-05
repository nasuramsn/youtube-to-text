import array
import yt_dlp
import whisper
import os
import numpy as np
import soundfile as sf
import librosa
import time
import sys

from transformers import pipeline

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
BERT_MODEL_PATH = None  # 例: "/home/ubuntu-user-2404/workspace/youtube-to-text/models/bert-base-japanese-char"

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


def load_wisper_model() -> array:
    print("Whisperモデルをロードしています...")
    try:
        model = whisper.load_model("large", device="cpu")
        _ = model.half()
        _ = model.cuda()

        for m in model.modules():
            if isinstance(m, whisper.model.LayerNorm):
                m.float()
        print("Whisperモデルが正しくロードされました。")
        return [True, model]
    except Exception as e:
        # print(f"Whisperモデルのロード中にエラーが発生しました: {e}")
        return [False, e]


def transcription(model, output_path: str, data_resampled: array, chunk_size: int, language: str) -> bool:
    # 音声ファイルをチャンクに分割して処理
    # 英語で文字起こししたいときはlanguage='en'にする。
    text_start = time.time()
    print("音声ファイルをチャンクに分割して文字起こしを開始します...")
    for i in range(0, len(data_resampled), chunk_size):
        chunk = data_resampled[i:i + chunk_size]
        try:
            result = model.transcribe(
                chunk,
                verbose=True,
                language=language,
                fp16=True,
                without_timestamps=True)
            chunk_transcription = result['text']
            print(f"チャンク {i//chunk_size + 1} の文字起こしが完了しました。")

            # 部分的な文字起こし結果をファイルに追記
            with open(output_path, "a", encoding="utf-8") as file:
                file.write(chunk_transcription + "\n")
        except Exception as e:
            print(f"チャンク {i//chunk_size + 1} の文字起こし中にエラーが発生しました: {e}")
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


def add_punctuation_mark(input_path: str, model_path: str = None) -> str:
    # 句読点を入れる
    # model_path: ローカルパスまたはHuggingFaceモデル名
    #   例: "/path/to/local/model" または "cl-tohoku/bert-base-japanese-char"
    #   デフォルト: "cl-tohoku/bert-base-japanese-char"
    thresh: float = 0.8     # このスコア以上の場合、句読点を挿入する
    i: int = 0
    punctuations: array = ["、", "。", "?"]
    chars_after_mask: int = 100
    
    # モデルパスが指定されていない場合はデフォルトを使用
    if model_path is None:
        model_path = "cl-tohoku/bert-base-japanese-char"
    
    print(f"モデルをロード中: {model_path}")
    nlp = pipeline("fill-mask", model=model_path)
    result = ""

    print(f"input_path: {input_path}")
    with open(input_path) as fin:
        for line in fin:
            original_sentence: str = line
            corrected_sentence: str = original_sentence
            while i < len(corrected_sentence):
                i += 1
                if corrected_sentence[i-1] in punctuations:
                    continue    # 句読点が連続してくることはない
                masked_text = insert_char_to_sentence(i, nlp.tokenizer.mask_token, corrected_sentence)

                try:
                    pre_context, post_context = masked_text.split("。")[-1].split(nlp.tokenizer.mask_token)
                    # scoreが一番高い文
                    res = nlp(f"{pre_context}{nlp.tokenizer.mask_token}{post_context[:chars_after_mask]}")[0]
                    if res["token_str"] not in punctuations:
                        continue
                    if res["score"] < thresh:
                        continue
                except ValueError as ve:
                    print('Error had orrcured')
                    print(ve)
                    print(i)
                    print(masked_text)
                    continue

                # punctuation = res["token_str"] if res["token_str"] != "?" else "。" # 今回は"？"は"。"として扱う
                punctuation = res["token_str"]
                corrected_sentence = insert_char_to_sentence(i, punctuation, corrected_sentence)
            print(f"original_sentence: {original_sentence}")
            print(f"corrected_sentence: {corrected_sentence}")
            result += corrected_sentence

    return result


def export_result_sentence(input_path: str, output_path: str, model_path: str = None) -> bool:
    # 句読点入りの文字を出力する
    # model_path: ローカルパスまたはHuggingFaceモデル名
    punctuation_start = time.time()
    result_sentence: str = add_punctuation_mark(input_path, model_path=model_path)
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
    # Whisperモデルをロード
    load_whisper_result = load_wisper_model()
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

    # 文字起こし開始
    result: bool = transcription(model, output_path, data_resampled, chunk_size, 'ja')

# 句読点を入れる
print(f"is_do_output_punctuation: {is_do_output_punctuation}")
if is_do_output_punctuation:
    output_punctuation_path: str = os.path.join(download_dir, "punctuation_mark.txt")
    if os.path.exists(output_punctuation_path):
        os.remove(output_punctuation_path)
    result: bool = export_result_sentence(output_path, output_punctuation_path, model_path=BERT_MODEL_PATH)
