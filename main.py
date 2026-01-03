import os
import time
import sys
from datetime import datetime

import boto3
from transformers import pipeline
from janome.tokenizer import Tokenizer as JanomeTokenizer

from convert_youtube import (
    download_youtube,
    check_exist_mp3,
    load_wisper_model,
    run_output_to_text,
    run_speaker_diarization_flow,
)


def validate_channel_and_date(channel: str, record_date: str) -> None:
    """
    channel と date(yyyyMMDD) のバリデーションを行う
    不正な場合は ValueError を送出する
    """
    # channel: 空は禁止、ファイル名やパスで問題になりそうな文字は弾く
    if not channel:
        raise ValueError("channel が指定されていません。例: MyChannel")

    invalid_chars = r'\/:*?"<>|'
    if any(c in channel for c in invalid_chars):
        raise ValueError(f"channel に使用できない文字が含まれています: {invalid_chars}")

    # date: yyyyMMDD の8桁数字かつ実在する日付
    if not record_date:
        raise ValueError("date が指定されていません。フォーマット: yyyyMMDD (例: 20250101)")

    if len(record_date) != 8 or not record_date.isdigit():
        raise ValueError("date は 8桁の数字で指定してください。フォーマット: yyyyMMDD (例: 20250101)")

    try:
        datetime.strptime(record_date, "%Y%m%d")
    except ValueError:
        raise ValueError("date が有効な日付ではありません。yyyyMMDD 形式で指定してください。")


def validate_execution_env(env: str) -> str:
    """
    実行環境パラメータ（local/aws）をバリデーションして正規化して返す
    不正な場合は ValueError を送出する
    """
    env_normalized = (env or "aws").lower()
    if env_normalized not in ("local", "aws"):
        raise ValueError('env は "local" または "aws" のいずれかで指定してください。')
    return env_normalized


def create_dynamodb_execute_log(
    execution_env: str,
    channel: str,
    record_date: str,
    video_url: str,
    no: int,
) -> int:
    """
    実行ログを DynamoDB に作成する
    - テーブル名: execute_youtube_encode_log
    - id: 既存の最大値 + 1
    """
    table_name = "execute_youtube_encode_log"

    # 実行環境ごとのエンドポイント設定
    if execution_env == "local":
        endpoint_url = "http://localhost:8000"
    else:
        endpoint_url = os.environ.get("DYNAMODB_ENDPOINT")
        if not endpoint_url:
            raise RuntimeError(
                "DYNAMODB_ENDPOINT が設定されていません（env=aws）。環境変数にエンドポイントを設定してください。"
            )

    dynamodb = boto3.resource("dynamodb", endpoint_url=endpoint_url)
    table = dynamodb.Table(table_name)

    # 既存レコードから最大 id を取得（シンプルに全スキャン）
    max_id = 0
    scan_kwargs = {"ProjectionExpression": "id"}
    while True:
        response = table.scan(**scan_kwargs)
        items = response.get("Items", [])
        for item in items:
            try:
                current_id = int(item.get("id", 0))
                if current_id > max_id:
                    max_id = current_id
            except (TypeError, ValueError):
                continue

        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break
        scan_kwargs["ExclusiveStartKey"] = last_evaluated_key

    new_id = max_id + 1

    # 日付のフォーマット変換
    iso_date = datetime.strptime(record_date, "%Y%m%d").strftime("%Y-%m-%d")
    start_dt = datetime.utcnow().isoformat() + "Z"

    item = {
        "id": new_id,
        "channel": channel,
        "no": no,
        "date": iso_date,
        "start_date_time": start_dt,
        "url": video_url,
        "execute_status": False,
    }

    table.put_item(Item=item)
    print(f"DynamoDB に実行ログを作成しました: {item}")
    return new_id


def update_dynamodb_execute_log_result(
    execution_env: str,
    log_id: int,
    success: bool,
) -> None:
    """
    実行ログレコードの end_date_time と execute_status を更新する
    """
    table_name = "execute_youtube_encode_log"

    # 実行環境ごとのエンドポイント設定
    if execution_env == "local":
        endpoint_url = "http://localhost:8000"
    else:
        endpoint_url = os.environ.get("DYNAMODB_ENDPOINT")
        if not endpoint_url:
            raise RuntimeError(
                "DYNAMODB_ENDPOINT が設定されていません（env=aws）。環境変数にエンドポイントを設定してください。"
            )

    dynamodb = boto3.resource("dynamodb", endpoint_url=endpoint_url)
    table = dynamodb.Table(table_name)

    # タイムゾーン付き UTC に変更（utcnow は将来廃止予定）
    from datetime import timezone

    end_dt = datetime.now(timezone.utc).isoformat()

    table.update_item(
        Key={"id": log_id},
        UpdateExpression="SET end_date_time = :end, execute_status = :st",
        ExpressionAttributeValues={
            ":end": end_dt,
            ":st": success,
        },
    )
    print(
        f"DynamoDB の実行ログを更新しました: id={log_id}, end_date_time={end_dt}, execute_status={success}"
    )


def get_s3_client(execution_env: str):
    """
    実行環境に応じて S3 クライアントを返す
    - local: エンドポイント http://localhost:4566
    - aws:   環境変数 S3_ENDPOINT からエンドポイントを取得
    """
    if execution_env == "local":
        endpoint_url = "http://localhost:4566"
    else:
        endpoint_url = os.environ.get("S3_ENDPOINT")
        if not endpoint_url:
            raise RuntimeError(
                "S3_ENDPOINT が設定されていません（env=aws）。環境変数にエンドポイントを設定してください。"
            )

    return boto3.client("s3", endpoint_url=endpoint_url)


def sanitize_bucket_name(base_name: str) -> str:
    """
    S3 の制約に合わせてバケット名をサニタイズする
    - 小文字英数字とドット・ハイフンのみ許可
    - 先頭と末尾は英数字
    """
    name = base_name.lower()
    name = re.sub(r"[^a-z0-9.-]", "-", name)
    name = re.sub(r"^[^a-z0-9]+", "", name)
    name = re.sub(r"[^a-z0-9]+$", "", name)
    if len(name) < 3:
        name = f"bkt-{name}".ljust(3, "0")
    return name


def upload_txt_files_to_s3(
    download_dir: str,
    execution_env: str,
    channel: str,
    record_date: str,
    no: int,
) -> bool:
    """
    medias 配下の *.txt を S3 にアップロードし、ローカルのファイルを削除する
    - バケット名: 環境変数 BUCKET_SURNIVERS_NEWS
    - オブジェクトキー: {channel}/{yyyyMM}/{yyyyMMdd}/{no}/{filename}
    Returns:
        True: すべてのアップロードが成功
        False: 途中でエラーが発生した
    """
    yyyy_mm = datetime.strptime(record_date, "%Y%m%d").strftime("%Y%m")
    yyyy_mmdd = record_date  # すでに yyyyMMDD 形式

    # S3 バケット名は小文字英数字とハイフンのみ・小文字推奨のため、必ず lower() で正規化する
    raw_bucket_name = os.environ.get("BUCKET_SURNIVERS_NEWS")
    if not raw_bucket_name:
        raise RuntimeError(
            "環境変数 BUCKET_SURNIVERS_NEWS が設定されていません。例: BUCKET_SURNIVERS_NEWS=surnivers-news-xxxx"
        )
    bucket_name = raw_bucket_name.lower()
    base_prefix = f"{channel}/{yyyy_mm}/{yyyy_mmdd}"

    s3 = get_s3_client(execution_env)
    print(f"S3 バケットを使用します: {bucket_name}, base_prefix={base_prefix}")

    try:
        # バケットの存在確認（なければ作成）
        existing_buckets = s3.list_buckets().get("Buckets", [])
        if not any(b.get("Name") == bucket_name for b in existing_buckets):
            if execution_env == "local":
                # LocalStack 側で事前にバケットを作っておく運用にする場合は、
                # ここでの create_bucket をスキップしたい場合もあるが、
                # まずは us-east-1 で作成を試みる
                from boto3.session import Session

                session = Session()
                region = (
                    session.region_name
                    or os.environ.get("AWS_REGION")
                    or os.environ.get("AWS_DEFAULT_REGION")
                    or "us-east-1"
                )
                if region == "us-east-1":
                    s3.create_bucket(Bucket=bucket_name)
                else:
                    s3.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": region},
                    )
            else:
                # AWS 環境ではリージョンに応じて LocationConstraint を指定
                from boto3.session import Session

                session = Session()
                region = (
                    session.region_name
                    or os.environ.get("AWS_REGION")
                    or os.environ.get("AWS_DEFAULT_REGION")
                    or "us-east-1"
                )
                if region == "us-east-1":
                    s3.create_bucket(Bucket=bucket_name)
                else:
                    s3.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": region},
                    )
            print(f"S3 バケットを作成しました: {bucket_name}")
        else:
            print(f"S3 バケットは既に存在します: {bucket_name}")

        # medias 配下の *.txt をアップロード
        if not os.path.isdir(download_dir):
            print(f"アップロード対象ディレクトリが存在しません: {download_dir}")
            return False

        dir_prefix = str(no)
        for filename in os.listdir(download_dir):
            if not filename.endswith(".txt"):
                continue
            local_path = os.path.join(download_dir, filename)
            if not os.path.isfile(local_path):
                continue

            if base_prefix:
                key = f"{base_prefix}/{dir_prefix}/{filename}"
            else:
                key = f"{dir_prefix}/{filename}"
            s3.upload_file(local_path, bucket_name, key)
            print(f"S3 にアップロードしました: bucket={bucket_name}, key={key}")

        # ローカル medias 配下のファイルを削除（必要に応じてコメントアウト解除）
        # for filename in os.listdir(download_dir):
        #     local_path = os.path.join(download_dir, filename)
        #     if os.path.isfile(local_path):
        #         os.remove(local_path)
        # print(f"ローカルディレクトリ内のファイルを削除しました: {download_dir}")
        return True
    except Exception as e:
        print(f"S3 アップロード中にエラーが発生しました: {e}")
        return False

# args
# 1: YoutubeのURLの番号（動画ID）
# 2: channel（チャンネル名などの識別子）
# 3: date（収録・配信日など、フォーマット: yyyyMMDD）
# 4: no（連番などの識別用番号。例: 1, 2）
# 5: is_do_download_youtube
# 6: is_do_download_whisper
# 7: is_do_output_to_text
# 8: is_do_output_punctuation
# 9: is_do_speaker_diarization (optional, default: False)
# 10: is_do_summarization (optional, default: False) - uses LLM for abstractive summarization
# 11: is_do_uploads (optional, default: False) - uses upload to S3 and update DynamoDB
# 12: env（実行環境: "local" または "aws"。省略時は "aws" として扱う）
# example:
#   qIW9NxF34Jo MyChannel 20250101 1 True True True True True True aws

args = sys.argv
video_url_org = "https://www.youtube.com/watch?v="
video_url = video_url_org + args[1]

channel = args[2] if len(args) > 2 else ""
record_date = args[3] if len(args) > 3 else ""

# channel / date のバリデーション
validate_channel_and_date(channel, record_date)

# no パラメータ（数字）。未指定や不正値のときは 1 にフォールバック
try:
    no = int(args[4]) if len(args) > 4 else 1
except ValueError:
    no = 1

# 実行環境パラメータ（local / aws）。省略時は "aws"
raw_env = args[12] if len(args) > 12 else "aws"
execution_env = validate_execution_env(raw_env)

is_do_download_youtube = True if len(args) > 5 and args[5] == "True" else False
is_do_download_whisper = True if len(args) > 6 and args[6] == "True" else False
is_do_output_to_text = True if len(args) > 7 and args[7] == "True" else False
is_do_output_punctuation = True if len(args) > 8 and args[8] == "True" else False
is_do_speaker_diarization = True if len(args) > 9 and args[9] == "True" else False
is_do_summarization = True if len(args) > 10 and args[10] == "True" else False
is_do_uploads = True if len(args) > 11 and args[11] == "True" else False

print(f"args: {args}")
print(f"video_id: {args[1] if len(args) > 1 else ''}")
print(f"channel: {channel}")
print(f"date: {record_date}")
print(f"no: {no}")
print(f"execution_env: {execution_env}")
print(f"is_do_download_youtube: {is_do_download_youtube}")

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


def summarize_with_gemini(text: str, num_sections: int = 5) -> str:
    """
    Gemini APIを使用してテキストを要約する
    
    Args:
        text: 要約するテキスト
        num_sections: 出力するセクション数の目安
    
    Returns:
        要約テキスト
    """
    import google.generativeai as genai
    
    # APIキーを環境変数から取得（必須。コードに直接書かない）
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY が設定されていません。環境変数 GEMINI_API_KEY に Gemini の API キーを設定してください。"
        )
    genai.configure(api_key=api_key)
    
    # Gemini 2.5 Flashモデルを使用（高速で高品質）
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # プロンプトを作成
    prompt = f"""あなたは日本語の要約編集者です。以下の文字起こしを読み、内容を{num_sections}つのセクションに整理して要約してください。

ルール：
- 各セクションは番号付きの見出しで始めてください（例：1. 経済政策について）
- 各セクションには2〜4個の箇条書きを含めてください
- 内容を捏造せず、固有名詞・数値・政策名は正確に保持してください
- 重要なポイントを漏らさず、具体的な内容を含めてください

以下のテキストを要約してください：

{text}"""
    
    print("Gemini APIで要約を生成中...")
    start_time = time.time()
    
    # 生成
    response = model.generate_content(prompt)
    
    end_time = time.time()
    print(f"Gemini API要約完了: {end_time - start_time:.2f}秒")
    
    return response.text


def export_summary(input_path: str, output_path: str, num_sections: int = 5) -> bool:
    """
    Gemini APIを使用して要約を生成してファイルに出力する
    
    Args:
        input_path: 入力ファイルのパス
        output_path: 出力ファイルのパス
        num_sections: 出力するセクション数の目安
    
    Returns:
        成功した場合はTrue、失敗した場合はFalse
    """
    try:
        # ファイルを読み込む
        with open(input_path, encoding="utf-8") as f:
            text = f.read()
        
        if not text.strip():
            print("要約できるテキストが見つかりませんでした。")
            return False
        
        print(f"入力テキスト長: {len(text)}文字")
        
        # Gemini APIで要約を生成
        summary = summarize_with_gemini(text, num_sections)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("【要約】\n\n")
            f.write(summary)
            f.write("\n")
        print(f"要約が {output_path} に保存されました。")
        return True
    except Exception as e:
        print(f"要約処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
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

# DynamoDB に実行ログを作成
execute_log_id = create_dynamodb_execute_log(
    execution_env=execution_env,
    channel=channel,
    record_date=record_date,
    video_url=video_url,
    no=no,
)

print(f"is_do_output_to_text: {is_do_output_to_text}")
if is_do_output_to_text:
    # 通常の文字起こしフロー（関数に分離）
    result: bool = run_output_to_text(audio_file, model, download_dir, output_path)
    # run_speaker_diarization_flow で再利用できるように data_resampled を保持しておく
    # ただし run_output_to_text 内部ではローカル変数なので、ここで再読み込みする場合は
    # 別途処理が必要（現状は話者分離側でも必要に応じて再読み込みしている）

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
    # 話者分離フロー（関数に分離）
    # data_resampled は現状このスコープにはないので None を渡し、
    # 関数側で必要に応じて再サンプリングする
    result: bool = run_speaker_diarization_flow(
        audio_file=audio_file,
        model=model,
        download_dir=download_dir,
        data_resampled=None,
    )

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

# 生成されたテキストファイルを S3 にアップロードしてローカルから削除
print(f"is_do_uploads: {is_do_uploads}")
if is_do_uploads:
    result: bool = upload_txt_files_to_s3(
        download_dir=download_dir,
        execution_env=execution_env,
        channel=channel,
        record_date=record_date,
        no=no,
    )

    if result:
        # DynamoDB の実行ログを更新（終了時刻と結果 True）
        update_dynamodb_execute_log_result(
            execution_env=execution_env,
            log_id=execute_log_id,
            success=True,
        )
