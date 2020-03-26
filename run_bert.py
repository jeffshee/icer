# coding: utf-8
import numpy as np
import pandas as pd
import torch
from transformers import BertJapaneseTokenizer, BertTokenizer
from transformers import BertModel


class MyBertEnglish:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def sentence_to_vector(self, sentence="Hello, my dog is cute."):
        input_ids = torch.tensor(self.tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
        last_hidden_states, _ = self.model(input_ids)
        return last_hidden_states.detach().numpy()[0, 0, :]


class MyBertJapanese:
    def __init__(self):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese')
        self.model = BertModel.from_pretrained('bert-base-japanese')

    def sentence_to_vector(self, sentence="こんにちは,私の犬はかわいいです."):
        inputs = torch.tensor(self.tokenizer.encode(self.tokenizer.tokenize(sentence))).unsqueeze(0)  # Batch size 1
        last_hidden_states, _ = self.model(input_ids=inputs)
        return last_hidden_states.detach().numpy()[0, 0, :]


def list2tsv(out_vecs, out_fpath):
    sub_seqs = []
    for vec in out_vecs:
        vec = map(str, vec)
        sub_seqs.extend(["\t".join(vec)])
    out_vec = np.array(["\n".join(map(str, sub_seqs))])
    np.savetxt(
        out_fpath,
        out_vec,
        delimiter="",
        fmt="%s"
    )


def main():
    df_csv = pd.read_csv("transcript_fukuoka.csv", encoding="shift-jis", header=0, index_col=False)
    sentence_list = [tex for tex in df_csv.Text]
    speaker_id = []
    speaker_id.extend([["{}".format(s, o)] for s, o, in zip(df_csv.loc[:, "Start time(ms)"], df_csv.Order)])
    print(speaker_id)
    list2tsv(speaker_id, "id.tsv")

    my_bert = MyBertJapanese()
    vector_list = []
    for sentence in sentence_list:
        vector_list.append(my_bert.sentence_to_vector(sentence))
    for i, vector in enumerate(vector_list):
        np.savetxt(
            "sample_{}.tsv".format(i),
            vector_list[i],
            delimiter="\t"
        )

    print(len(vector_list))
    sub_seqs = []
    for vec in vector_list:
        vec = map(str, vec)
        sub_seqs.extend(["\t".join(vec)])
    out_vec = np.array(["\n".join(map(str, sub_seqs))])
    np.savetxt(
        "sample.tsv",
        out_vec,
        delimiter="",
        fmt="%s"
    )


if __name__ == "__main__":
    main()
