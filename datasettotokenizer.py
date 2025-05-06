import json
import sentencepiece as spm
import MeCab
import os
from transformers import PreTrainedTokenizerFast

class DatasetToTokenizer:
    def __init__(self, vocab_size, json_file_path, tokenizer_output_path="./result", special_tokens=None):
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>", "<s>", "</s>"]
        self.vocab_size = vocab_size
        self.json_file_path = json_file_path
        self.tokenizer_output_path = tokenizer_output_path
        self.special_tokens = special_tokens
        self.mecab = MeCab.Tagger("-Owakati")  # MeCabの初期化

    def load_json(self):
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def process_sentences(self, data):
        sentences = []
        if 'test_processed' in self.json_file_path:
            # test_processed.jsonの場合
            for entry in data:
                if 'text' in entry:
                    sentence = entry['text']
                    if sentence.strip():
                        wakati_sentence = self.mecab.parse(sentence).strip()
                        sentences.append(wakati_sentence)
        elif 'combined_processed' in self.json_file_path:
            # combined_processed.jsonの場合
            for entry in data:
                if 'instruction' in entry and 'output' in entry:
                    combined_sentence = f"{entry['instruction']} {entry['output']}"
                    if combined_sentence.strip():
                        wakati_sentence = self.mecab.parse(combined_sentence).strip()
                        sentences.append(wakati_sentence)
        return sentences

    def create_sentencepiece_model(self, sentences):
        # 一時ファイルに分かち書き結果を書き込み
        temp_file = "temp.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + "\n")

        # SentencePiece用にスペシャルトークンを追加する設定を用意
        user_defined_symbols = ",".join(self.special_tokens)

        # SentencePieceモデルの学習
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=self.tokenizer_output_path + "/sentencepiece",
            vocab_size=self.vocab_size,
            user_defined_symbols=user_defined_symbols,
            train_extremely_large_corpus=True
        )

        # 一時ファイルの削除
        os.remove(temp_file)

    def save_tokenizer(self):
        # トークナイザーを作成し保存する
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{self.tokenizer_output_path}/sentencepiece.model",
        )

        # トークナイザーを指定したディレクトリに保存
        tokenizer.save_pretrained(self.tokenizer_output_path)
        print(f"トークナイザー設定ファイルが {self.tokenizer_output_path} に保存されました。")

    def create_tokenizer(self):
        # JSONファイルの読み込み
        data = self.load_json()

        # 文の処理
        sentences = self.process_sentences(data)

        # 分かち書きされた文が空でないか確認
        if not sentences:
            print("分かち書きされた文がありません。トレーニングをスキップします。")
            return

        # SentencePieceモデルの作成
        self.create_sentencepiece_model(sentences)

        # トークナイザーの保存
        self.save_tokenizer()

# 例としての呼び出し方
vocab_size = 2 ** 16
json_file_path = 'tokenizer.json'  # 分かち書きするJSONファイルのパス
tokenizer_output_path = './result'  # 出力されるトークナイザーのパス（拡張子は自動で追加される）
special_tokens = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>", "<s>", "</s>"]  # 追加するスペシャルトークン

tokenizer_creator = DatasetToTokenizer(vocab_size, json_file_path, tokenizer_output_path, special_tokens)
tokenizer_creator.create_tokenizer()