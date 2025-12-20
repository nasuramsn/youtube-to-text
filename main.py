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
from janome.tokenizer import Tokenizer as JanomeTokenizer

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

# ダウンロードディレクトリ（相対パスを使用して移植性を向上）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
download_dir = os.path.join(BASE_DIR, "medias")
output_path = os.path.join(download_dir, "transcription.txt")

# MP3ファイルの保存パス
audio_file_org = os.path.join(download_dir, args[1])
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


def add_punctuation_mark_word_boundary(input_path: str) -> str:
    """
    句読点を単語境界にのみ挿入する改良版
    Janomeを使用して単語境界を検出し、BERTで句読点を予測する
    これにより「言い、ます」のような単語内への誤挿入を防ぐ
    """
    thresh: float = 0.5
    punctuations = ["、", "。", "？"]
    chars_after_mask: int = 100
    
    print("句読点予測モデルをロードしています...")
    nlp = pipeline("fill-mask", model="cl-tohoku/bert-base-japanese-char")
    janome = JanomeTokenizer()
    result = ""

    print(f"input_path: {input_path}")
    with open(input_path, encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            original_sentence = line.strip()
            if not original_sentence:
                result += "\n"
                continue
            
            tokens = list(janome.tokenize(original_sentence))
            word_boundaries = []
            pos = 0
            for token in tokens:
                pos += len(token.surface)
                word_boundaries.append(pos)
            
            corrected_sentence = original_sentence
            offset = 0
            
            for boundary in word_boundaries[:-1]:
                adjusted_pos = boundary + offset
                
                if adjusted_pos >= len(corrected_sentence):
                    break
                if adjusted_pos > 0 and corrected_sentence[adjusted_pos - 1] in punctuations:
                    continue
                
                pre_context = corrected_sentence[max(0, adjusted_pos - 50):adjusted_pos]
                post_context = corrected_sentence[adjusted_pos:adjusted_pos + chars_after_mask]
                
                if not pre_context or not post_context:
                    continue
                
                masked_input = f"{pre_context}{nlp.tokenizer.mask_token}{post_context}"
                
                try:
                    predictions = nlp(masked_input)
                    if not predictions:
                        continue
                    
                    top_pred = predictions[0]
                    token_str = top_pred["token_str"]
                    score = top_pred["score"]
                    
                    if token_str in punctuations and score >= thresh:
                        corrected_sentence = (
                            corrected_sentence[:adjusted_pos] + 
                            token_str + 
                            corrected_sentence[adjusted_pos:]
                        )
                        offset += 1
                        
                        if token_str == "。":
                            corrected_sentence = (
                                corrected_sentence[:adjusted_pos + 1] + 
                                "\n" + 
                                corrected_sentence[adjusted_pos + 1:]
                            )
                            offset += 1
                            
                except Exception as e:
                    print(f"行 {line_num} の位置 {boundary} で予測エラー: {e}")
                    continue
            
            print(f"行 {line_num}: 処理完了")
            result += corrected_sentence + "\n"

    return result


def add_punctuation_mark(input_path: str) -> str:
    """
    句読点を追加する（改良版: 単語境界を使用）
    """
    return add_punctuation_mark_word_boundary(input_path)


def export_result_sentence(input_path: str, output_path: str) -> bool:
    # 句読点入りの文字を出力する
    punctuation_start = time.time()
    result_sentence: str = add_punctuation_mark(input_path)
    with open(output_path, "a", encoding="utf-8") as file:
        file.write(result_sentence)
    punctuation_end = time.time()
    punctuation_time = punctuation_end - punctuation_start
    print(f"句読点追加時間: {punctuation_time}")
    print(f"句読点付き文字起こし結果が {output_punctuation_path} に保存されました。")
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
    result: bool = export_result_sentence(output_path, output_punctuation_path)
