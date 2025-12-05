"""
モデルとトークナイザーをローカルディレクトリに保存するスクリプト
Save model and tokenizer to a local directory for backup/transfer
"""
import os
from transformers import AutoModel, AutoTokenizer, pipeline

def save_model_to_local(model_name: str, local_path: str):
    """
    モデルとトークナイザーを指定したローカルパスに保存
    
    Args:
        model_name: HuggingFace model identifier (e.g., "cl-tohoku/bert-base-japanese-char")
        local_path: 保存先のローカルパス
    """
    print(f"モデル '{model_name}' をダウンロード中...")
    
    # モデルとトークナイザーをダウンロード
    print("トークナイザーをダウンロード中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("モデルをダウンロード中...")
    model = AutoModel.from_pretrained(model_name)
    
    # ローカルパスに保存
    print(f"ローカルパス '{local_path}' に保存中...")
    os.makedirs(local_path, exist_ok=True)
    
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)
    
    print(f"✅ 保存完了: {local_path}")
    print(f"\n使用方法:")
    print(f"  pipeline('fill-mask', model='{local_path}')")
    print(f"\nまたは他のマシンに転送する場合:")
    print(f"  1. ディレクトリ全体をコピー: cp -r {local_path} /path/to/destination/")
    print(f"  2. 転送後、同じローカルパスから読み込み可能")
    
    return local_path


if __name__ == "__main__":
    # 設定
    MODEL_NAME = "cl-tohoku/bert-base-japanese-char"
    LOCAL_MODEL_PATH = "/home/ubuntu-user-2404/workspace/youtube-to-text/models/bert-base-japanese-char"
    
    # モデルを保存
    save_model_to_local(MODEL_NAME, LOCAL_MODEL_PATH)
