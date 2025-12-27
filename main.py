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
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# args
# 1: YoutubeのURLの番号
# 2: is_do_download_youtube
# 3: is_do_download_whisper
# 4: is_do_output_to_text
# 5: is_do_output_punctuation
# 6: is_do_speaker_diarization (optional, default: False)
# 7: is_do_summarization (optional, default: False) - uses LLM for abstractive summarization
# example: qIW9NxF34Jo True True True True True True

args = sys.argv
video_url_org = "https://www.youtube.com/watch?v="
video_url = video_url_org + args[1]

is_do_download_youtube = True if args[2] == "True" else False
is_do_download_whisper = True if args[3] == "True" else False
is_do_output_to_text = True if args[4] == "True" else False
is_do_output_punctuation = True if args[5] == "True" else False
is_do_speaker_diarization = True if len(args) > 6 and args[6] == "True" else False
is_do_summarization = True if len(args) > 7 and args[7] == "True" else False

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


def perform_speaker_diarization(audio_file: str, num_speakers: int = None) -> list:
    """
    音声ファイルから話者分離を行う
    
    Args:
        audio_file: 音声ファイルのパス
        num_speakers: 話者数（Noneの場合は自動推定）
    
    Returns:
        話者セグメントのリスト: [(start_time, end_time, speaker_id), ...]
    """
    print("話者分離を開始します...")
    diarization_start = time.time()
    
    # Voice encoderをロード
    print("Voice Encoderをロードしています...")
    encoder = VoiceEncoder()
    
    # 音声ファイルを読み込み・前処理
    print(f"音声ファイルを読み込んでいます: {audio_file}")
    wav = preprocess_wav(audio_file)
    
    # セグメント分割のパラメータ
    segment_duration = 1.5  # 各セグメントの長さ（秒）
    step_duration = 0.75    # セグメント間のステップ（秒）
    sample_rate = 16000
    
    segment_samples = int(segment_duration * sample_rate)
    step_samples = int(step_duration * sample_rate)
    
    # 音声をセグメントに分割してembeddingを計算
    print("音声セグメントのembeddingを計算しています...")
    embeddings = []
    segment_times = []
    
    for start_sample in range(0, len(wav) - segment_samples, step_samples):
        end_sample = start_sample + segment_samples
        segment = wav[start_sample:end_sample]
        
        # セグメントが短すぎる場合はスキップ
        if len(segment) < segment_samples * 0.5:
            continue
        
        try:
            embedding = encoder.embed_utterance(segment)
            embeddings.append(embedding)
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            segment_times.append((start_time, end_time))
        except Exception as e:
            print(f"セグメント {start_sample} でエラー: {e}")
            continue
    
    if len(embeddings) == 0:
        print("有効なセグメントが見つかりませんでした")
        return []
    
    embeddings = np.array(embeddings)
    print(f"計算されたembedding数: {len(embeddings)}")
    
    # クラスタリングで話者を分離
    print("話者クラスタリングを実行しています...")
    if num_speakers is None:
        # 話者数を自動推定（2-5人の範囲で最適なクラスタ数を探す）
        from sklearn.metrics import silhouette_score
        best_score = -1
        best_n = 2
        for n in range(2, min(6, len(embeddings))):
            try:
                clustering = AgglomerativeClustering(n_clusters=n)
                labels = clustering.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_n = n
            except Exception:
                continue
        num_speakers = best_n
    
    clustering = AgglomerativeClustering(n_clusters=num_speakers)
    labels = clustering.fit_predict(embeddings)
    
    # セグメントと話者ラベルを結合
    speaker_segments = []
    for i, (start_time, end_time) in enumerate(segment_times):
        speaker_id = labels[i]
        speaker_segments.append((start_time, end_time, speaker_id))
    
    # 連続する同じ話者のセグメントをマージ
    merged_segments = []
    if speaker_segments:
        current_start, current_end, current_speaker = speaker_segments[0]
        for start_time, end_time, speaker_id in speaker_segments[1:]:
            if speaker_id == current_speaker:
                current_end = end_time
            else:
                merged_segments.append((current_start, current_end, current_speaker))
                current_start, current_end, current_speaker = start_time, end_time, speaker_id
        merged_segments.append((current_start, current_end, current_speaker))
    
    diarization_end = time.time()
    print(f"話者分離完了。処理時間: {diarization_end - diarization_start:.2f}秒")
    print(f"検出されたセグメント数: {len(merged_segments)}")
    
    return merged_segments


def transcription_with_timestamps(model, data_resampled: array, chunk_size: int, language: str) -> list:
    """
    タイムスタンプ付きで文字起こしを行う
    
    Returns:
        セグメントのリスト: [(start_time, end_time, text), ...]
    """
    print("タイムスタンプ付き文字起こしを開始します...")
    all_segments = []
    
    for chunk_idx, i in enumerate(range(0, len(data_resampled), chunk_size)):
        chunk = data_resampled[i:i + chunk_size]
        chunk_start_time = i / 16000  # 16kHzサンプルレート
        
        try:
            result = model.transcribe(
                chunk,
                verbose=False,
                language=language,
                fp16=True,
                without_timestamps=False  # タイムスタンプを有効化
            )
            
            for segment in result.get('segments', []):
                start = chunk_start_time + segment['start']
                end = chunk_start_time + segment['end']
                text = segment['text'].strip()
                if text:
                    all_segments.append((start, end, text))
            
            print(f"チャンク {chunk_idx + 1} の文字起こしが完了しました。")
        except Exception as e:
            print(f"チャンク {chunk_idx + 1} の文字起こし中にエラーが発生しました: {e}")
            continue
    
    return all_segments


def align_transcription_with_speakers(transcription_segments: list, speaker_segments: list) -> list:
    """
    文字起こしセグメントと話者セグメントを時間で照合する
    
    Args:
        transcription_segments: [(start, end, text), ...]
        speaker_segments: [(start, end, speaker_id), ...]
    
    Returns:
        話者付きセグメント: [(start, end, speaker_id, text), ...]
    """
    print("文字起こしと話者情報を照合しています...")
    aligned_segments = []
    
    for trans_start, trans_end, text in transcription_segments:
        trans_mid = (trans_start + trans_end) / 2
        
        # 最も重複が大きい話者セグメントを見つける
        best_speaker = 0
        best_overlap = 0
        
        for spk_start, spk_end, speaker_id in speaker_segments:
            # 重複区間を計算
            overlap_start = max(trans_start, spk_start)
            overlap_end = min(trans_end, spk_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_id
        
        aligned_segments.append((trans_start, trans_end, best_speaker, text))
    
    return aligned_segments


def export_diarized_transcription(aligned_segments: list, output_path: str) -> bool:
    """
    話者分離された文字起こし結果をファイルに出力する
    """
    print(f"話者分離結果を {output_path} に保存しています...")
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            current_speaker = None
            for start, end, speaker_id, text in aligned_segments:
                if speaker_id != current_speaker:
                    if current_speaker is not None:
                        f.write("\n")
                    f.write(f"SPEAKER_{speaker_id}:\n")
                    current_speaker = speaker_id
                f.write(f"{text}\n")
        
        print(f"話者分離結果が {output_path} に保存されました。")
        return True
    except Exception as e:
        print(f"ファイル出力中にエラーが発生しました: {e}")
        return False


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


def extract_sentences(text: str) -> list:
    """
    テキストを文に分割する
    """
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ["。", "？", "！"]:
            sentence = current.strip()
            if sentence and len(sentence) > 5:
                sentences.append(sentence)
            current = ""
    if current.strip() and len(current.strip()) > 5:
        sentences.append(current.strip())
    return sentences


# LLMモデルのグローバル変数（遅延ロード用）
_llm_model = None
_llm_tokenizer = None


def load_llm_model():
    """
    LLMモデルを遅延ロードする
    初回呼び出し時にモデルをダウンロード・ロードし、以降はキャッシュを使用
    """
    global _llm_model, _llm_tokenizer
    
    if _llm_model is not None:
        return _llm_model, _llm_tokenizer
    
    print("LLMモデルをロードしています（初回は時間がかかります）...")
    load_start = time.time()
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 日本語対応のinstruction-tunedモデルを使用
        # ELYZA-japanese-Llama-2-7b-instruct は日本語に強いモデル
        model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
        
        print(f"モデル: {model_name}")
        
        # トークナイザーをロード
        _llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # モデルをロード（GPUがあれば使用、なければCPU）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"デバイス: {device}")
        
        if device == "cuda":
            # GPU使用時は4bit量子化でメモリ節約
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                _llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            except ImportError:
                # bitsandbytesがない場合はfloat16で読み込み
                _llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
        else:
            # CPU使用時
            _llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        
        load_end = time.time()
        print(f"LLMモデルのロード完了: {load_end - load_start:.2f}秒")
        
        return _llm_model, _llm_tokenizer
        
    except Exception as e:
        print(f"LLMモデルのロード中にエラーが発生しました: {e}")
        raise


def split_text_into_chunks_by_tokens(text: str, tokenizer, max_tokens: int = 1200) -> list:
    """
    テキストをトークン数ベースで文境界でチャンクに分割する
    
    Args:
        text: 入力テキスト
        tokenizer: トークナイザー
        max_tokens: 各チャンクの最大トークン数（プロンプト用に余裕を持たせる, デフォルト: 1200）
    
    Returns:
        チャンクのリスト
    """
    sentences = extract_sentences(text)
    
    if not sentences:
        return [text] if text.strip() else []
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        # 文のトークン数を計算
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            current_chunk += sentence
            current_tokens += sentence_tokens
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def clean_summary_output(summary: str) -> str:
    """
    要約出力をクリーンアップする
    空の箇条書きや不正な出力を除去する
    
    Args:
        summary: 生成された要約テキスト
    
    Returns:
        クリーンアップされた要約
    """
    lines = summary.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # 空行は保持（段落区切り用）
        if not stripped:
            if cleaned_lines and cleaned_lines[-1] != '':
                cleaned_lines.append('')
            continue
        # 「・」だけの行や、「・」と空白だけの行は除去
        if stripped == '・' or stripped.replace('・', '').replace(' ', '').replace('　', '') == '':
            continue
        # 「・」で始まるが内容が5文字未満の行は除去
        if stripped.startswith('・') and len(stripped) < 6:
            continue
        cleaned_lines.append(line)
    
    # 末尾の空行を除去
    while cleaned_lines and cleaned_lines[-1] == '':
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)


def summarize_chunk_with_llm(text: str, model, tokenizer) -> str:
    """
    LLMを使用してテキストチャンクを要約する
    
    Args:
        text: 要約するテキスト
        model: LLMモデル
        tokenizer: トークナイザー
    
    Returns:
        要約テキスト
    """
    import torch
    
    # プロンプトテンプレート（ELYZA形式）
    # 明確な指示と出力例を含める
    prompt = f"""[INST] <<SYS>>
あなたは日本語の要約編集者です。以下の文字起こしを読み、内容を捏造せず、重要なトピックを整理して要約してください。

ルール：
- 出力は必ず箇条書き形式で、各項目は「・」で始めてください
- 各項目は必ず10文字以上の具体的な内容を含めてください
- 3〜5個の重要なポイントを抽出してください
- 固有名詞・数値・政策名は可能な限り保持してください
- 空の箇条書きは禁止です

出力例：
・経済成長率は前年比2.5%増加し、政府目標を達成した
・日銀は金融緩和政策を継続する方針を示した
・消費者物価指数は3ヶ月連続で上昇傾向にある
<</SYS>>

以下のテキストを要約してください：

{text}
[/INST]"""
    
    # トークン化（truncationを無効化して警告を出す）
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_length = inputs["input_ids"].shape[1]
    
    # 入力が長すぎる場合は警告
    if input_length > 3500:
        print(f"警告: 入力トークン数が多いです ({input_length} tokens)")
        # 安全のため切り詰める
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500)
    
    # デバイスに移動
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成（グリーディデコーディングで安定した出力を得る）
    with torch.no_grad():
        # RTX 3070Ti Laptop (8GB 前後を想定) では 7B モデル + 長コンテキストで
        # 512 トークン生成はかなり重いため、max_new_tokens を 256 に抑える
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # グリーディデコーディング
            num_beams=1,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # デコード
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # プロンプト部分を除去して要約部分のみ抽出
    if "[/INST]" in generated_text:
        summary = generated_text.split("[/INST]")[-1].strip()
    else:
        summary = generated_text[len(prompt):].strip()
    
    # 出力をクリーンアップ
    summary = clean_summary_output(summary)
    
    return summary


def summarize_text_with_llm(input_path: str, num_sections: int = 5) -> str:
    """
    LLMを使用して抽象型要約を行う
    長いテキストは階層的に要約する
    
    Args:
        input_path: 入力ファイルのパス
        num_sections: 出力するセクション数の目安
    
    Returns:
        構造化された要約
    """
    print("LLM要約処理を開始します...")
    summary_start = time.time()
    
    # ファイルを読み込む
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    
    if not text.strip():
        return "要約できるテキストが見つかりませんでした。"
    
    print(f"入力テキスト長: {len(text)}文字")
    
    # LLMモデルをロード
    model, tokenizer = load_llm_model()
    
    # テキストをトークン数ベースでチャンクに分割（RTX 3070Ti Laptop 想定でやや小さめに分割）
    # チャンクが大きすぎると 7B モデル + CPU オフロード環境で generate が極端に遅くなるため、
    # 1 チャンクあたりおよそ 1200 トークンを目安に分割する
    chunks = split_text_into_chunks_by_tokens(text, tokenizer, max_tokens=1200)
    print(f"チャンク数: {len(chunks)}")
    
    if len(chunks) == 0:
        return "要約できるテキストが見つかりませんでした。"
    
    if len(chunks) == 1:
        # 短いテキストは直接要約
        summary = summarize_chunk_with_llm(chunks[0], model, tokenizer)
    else:
        # 長いテキストは階層的に要約
        # Step 1: 各チャンクを要約
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"チャンク {i+1}/{len(chunks)} を要約中...")
            chunk_summary = summarize_chunk_with_llm(chunk, model, tokenizer)
            if chunk_summary.strip():  # 空でない要約のみ追加
                chunk_summaries.append(chunk_summary)
        
        if not chunk_summaries:
            return "要約を生成できませんでした。"
        
        # Step 2: チャンク要約を統合して最終要約を生成
        combined_summaries = "\n\n".join(chunk_summaries)
        print("最終要約を生成中...")
        
        # 統合用のプロンプト
        import torch
        
        final_prompt = f"""[INST] <<SYS>>
あなたは日本語の要約編集者です。以下は長い文字起こしの各部分の要約です。
これらを統合して、全体の内容を{num_sections}つのセクションにまとめてください。

ルール：
- 各セクションは番号付きの見出しで始めてください
- 各セクションには2〜4個の箇条書き（・で始まる）を含めてください
- 各箇条書きは必ず10文字以上の具体的な内容を含めてください
- 空の箇条書きは禁止です
- 内容を捏造せず、固有名詞・数値は保持してください

出力例：
1. 経済政策について
・政府は来年度の経済成長率を2.5%と予測している
・日銀は金融緩和政策を継続する方針を示した

2. 政局分析
・与党は次期選挙に向けて政策調整を進めている
<</SYS>>

以下の要約を統合してください：

{combined_summaries}
[/INST]"""
        
        # トークン化（truncationを無効化して警告を出す）
        inputs = tokenizer(final_prompt, return_tensors="pt", truncation=False)
        input_length = inputs["input_ids"].shape[1]
        
        # 入力が長すぎる場合は警告して切り詰める
        if input_length > 3500:
            print(f"警告: 統合プロンプトのトークン数が多いです ({input_length} tokens)")
            inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=3500)
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # グリーディデコーディングで安定した出力を得る
        with torch.no_grad():
            # 最終統合要約も生成トークン数を抑えて計算量を削減
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # グリーディデコーディング
                num_beams=1,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "[/INST]" in generated_text:
            summary = generated_text.split("[/INST]")[-1].strip()
        else:
            summary = generated_text[len(final_prompt):].strip()
        
        # 出力をクリーンアップ
        summary = clean_summary_output(summary)
    
    summary_end = time.time()
    print(f"LLM要約処理時間: {summary_end - summary_start:.2f}秒")
    
    return summary


def export_summary(input_path: str, output_path: str, num_sections: int = 5) -> bool:
    """
    LLMを使用して要約を生成してファイルに出力する
    
    Args:
        input_path: 入力ファイルのパス
        output_path: 出力ファイルのパス
        num_sections: 出力するセクション数の目安
    
    Returns:
        成功した場合はTrue、失敗した場合はFalse
    """
    try:
        summary = summarize_text_with_llm(input_path, num_sections)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("【要約】\n\n")
            f.write(summary)
            f.write("\n")
        print(f"要約が {output_path} に保存されました。")
        return True
    except Exception as e:
        print(f"要約処理中にエラーが発生しました: {e}")
        return False


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

# 話者分離を行う
print(f"is_do_speaker_diarization: {is_do_speaker_diarization}")
if is_do_speaker_diarization:
    output_diarization_path: str = os.path.join(download_dir, "diarization.txt")
    if os.path.exists(output_diarization_path):
        os.remove(output_diarization_path)
    
    # オーディオファイルを読み込む（まだ読み込んでいない場合）
    if 'data_resampled' not in dir():
        convert_audio_result = convert_audio_to_monoral(audio_file, model)
        if not convert_audio_result[0]:
            raise RuntimeError("オーディオの変換に失敗しました")
        data_resampled = convert_audio_result[1]
    
    # Whisperモデルをロード（まだロードしていない場合）
    if model is None:
        load_whisper_result = load_wisper_model()
        if not load_whisper_result[0]:
            raise RuntimeError(f"Whisperモデルのロード中にエラーが発生しました: {load_whisper_result[1]}")
        model = load_whisper_result[1]
    
    # チャンクサイズを設定
    chunk_size = 60 * 16000  # 60秒 * 16kHz
    
    # 話者分離を実行
    print("話者分離処理を開始します...")
    speaker_segments = perform_speaker_diarization(audio_file)
    
    # タイムスタンプ付き文字起こしを実行
    transcription_segments = transcription_with_timestamps(model, data_resampled, chunk_size, 'ja')
    
    # 文字起こしと話者情報を照合
    aligned_segments = align_transcription_with_speakers(transcription_segments, speaker_segments)
    
    # 結果を出力
    result: bool = export_diarized_transcription(aligned_segments, output_diarization_path)
    print(f"話者分離付き文字起こし結果が {output_diarization_path} に保存されました。")

# 要約を生成する
print(f"is_do_summarization: {is_do_summarization}")
if is_do_summarization:
    output_summary_path: str = os.path.join(download_dir, "summary.txt")
    if os.path.exists(output_summary_path):
        os.remove(output_summary_path)
    
    # 句読点付きテキストがあればそれを使用、なければ元のテキストを使用
    if is_do_output_punctuation:
        summary_input_path = os.path.join(download_dir, "punctuation_mark.txt")
    else:
        summary_input_path = output_path
    
    if os.path.exists(summary_input_path):
        result: bool = export_summary(summary_input_path, output_summary_path, num_sections=5)
    else:
        print(f"要約の入力ファイルが見つかりません: {summary_input_path}")
