import argparse
import datetime
import os

import MeCab
import numpy
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
# parser.add_argument('transcription', help='A file of result Speech to Text.')
# parser.add_argument('answer', help='A file of correct answer data.')
parser.add_argument('--output', '-o', nargs='?', const='./result_evaluateSTT.txt',
                    help='An output file (default: ./result_evaluateSTT.txt)')
parser.add_argument('--compare', '-c', nargs='?', const='./compare_list.txt',
                    help='A compare file (default: ./compare_list.txt)')


class Levenshtein_distance():
    num_all_words = 0
    m = []
    error_hashmap = {}

    def __init__(self, trans, ans):
        self.trans = trans
        self.ans = ans
        self.num_all_words = len(self.ans)
        self.m = numpy.zeros((len(self.ans) + 1) * (len(self.trans) + 1), dtype=numpy.uint16)
        self.m = numpy.reshape(self.m, (len(self.ans) + 1, len(self.trans) + 1))
        self.calc_WER()
        self.error_hashmap = self.calc_error_class()

    def get_num_all(self):
        return self.num_all_words

    def get_error_type(self):
        return self.error_hashmap

    def get_WER(self):
        return self.m[-1][-1] / self.num_all_words

    def calc_WER(self):
        for i in range(len(self.ans) + 1):
            self.m[i][0] = i

        for j in range(len(self.trans) + 1):
            self.m[0][j] = j

        for i in range(1, len(self.ans) + 1):
            for j in range(1, len(self.trans) + 1):
                if self.ans[i - 1].lower() == self.trans[j - 1].lower():
                    self.m[i][j] = self.m[i - 1][j - 1]
                else:
                    # distance_list = [ins_distance, del_distance, sub_distance]
                    distance_list = [self.m[i - 1][j] + 1, self.m[i][j - 1] + 1, self.m[i - 1][j - 1] + 1]
                    self.m[i][j] = min(distance_list)

            print("\rProgress: {0:d}%".format(int(i / self.get_num_all() * 100)), end="")
        print('')
        print('Complete evaluate.')
        return 0

    def calc_error_class(self):
        a_i = len(self.ans)
        t_i = len(self.trans)
        error_hashmap = {"ins": 0, "del": 0, "sub": 0, "equal": 0}
        # error_list = []
        # compare_list = [["Answer word", "Transcription word"], ["Answer word", "Transcription word"], ...]
        #        |  Ans   |  Trans
        # ----------------------------
        # Correct|Ans word|    ""
        #   Ins  |   *    |Trans word
        #   Del  |Ans word|    *
        #   Sub  |Ans word|Trans word
        compare_list = []
        while not (a_i == 0 and t_i == 0):
            if a_i >= 1 and t_i >= 1 and self.m[a_i][t_i] == self.m[a_i - 1][t_i - 1] and self.ans[a_i - 1].lower() == \
                    self.trans[t_i - 1].lower():
                a_i -= 1
                t_i -= 1
                error_hashmap["equal"] += 1
                # error_list.append("C")
                compare_list.append([self.ans[a_i], u""])
            elif t_i >= 1 and self.m[a_i][t_i] == self.m[a_i][t_i - 1] + 1:
                t_i -= 1
                error_hashmap["ins"] += 1
                # error_list.append("I")
                compare_list.append([u"*", self.trans[t_i]])
            elif a_i >= 1 and self.m[a_i][t_i] == self.m[a_i - 1][t_i] + 1:
                a_i -= 1
                error_hashmap["del"] += 1
                # error_list.append("D")
                compare_list.append([self.ans[a_i], u"*"])
            elif a_i >= 1 and t_i >= 1 and self.m[a_i][t_i] == self.m[a_i - 1][t_i - 1] + 1:
                a_i -= 1
                t_i -= 1
                error_hashmap["sub"] += 1
                # error_list.append("S")
                compare_list.append([self.ans[a_i], self.trans[t_i]])
            else:
                print('error')
                break
        # with open('error_list.txt', 'a') as f:
        #     for i in error_list:
        #         f.write(i)
        compare_list.reverse()
        if parser.parse_args().compare:
            with open(parser.parse_args().compare, 'a', encoding='utf-8') as f:
                for compare_pair in compare_list:
                    f.write(compare_pair[0] + u"," + compare_pair[1] + "\n")

        return error_hashmap


def output_result(evaluate):
    # evaluate: class Levenshtein_distance
    result_str_list = []
    result_str_list.append('WER: %f' % evaluate.get_WER())
    hashmap = evaluate.get_error_type()
    result_str_list.append('ins: %d, del: %d, sub: %d, correct: %d, words: %d' % (
        hashmap["ins"], hashmap["del"], hashmap["sub"], hashmap["equal"], evaluate.get_num_all()))

    if parser.parse_args().output:
        resultpath = os.path.split(parser.parse_args().output)
        os.makedirs(resultpath[0], exist_ok=True)
        file = open(parser.parse_args().output, mode='a', encoding="utf8")
        file.write(datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S") + '\n')
        file.write(result_str_list[0] + '\n')
        file.write(result_str_list[1] + '\n')
        file.write('---' + '\n')
        file.close()
    return result_str_list


def separate_text(in_fpath):
    tagger = MeCab.Tagger("-Owakati")
    df = pd.read_csv(in_fpath)
    texts = []
    for t in df.Text:
        if t is np.nan:
            continue
        t = t.replace(';', '')
        t = t.replace(' ', '')
        t = t.replace('　', '')
        t = t.replace('「', '')
        t = t.replace('」', '')
        t = t.replace('?', '')
        t = t.replace('？', '')
        t = t.replace('。', '')
        t = t.replace('…', '')
        t = t.replace('、', '')
        s = tagger.parse(t)
        s = s.replace('\n', '')
        s = s.split(' ')
        texts.extend(s)
    texts = [t for t in texts if t != '']
    # Convert all kanji into yomi
    for i, t in enumerate(texts):
        tagger = MeCab.Tagger("-Oyomi")
        yomi = tagger.parse(t)
        yomi = yomi.replace('\n', '')
        print(f"{t} -> {yomi}")
        texts[i] = yomi
    return texts


def main():
    # Setting of command-line parameters
    # transcription: result of Speech to Text
    # answer: correct words
    in_fpath = 'evaluation/s2t_result/expt58_transcript.csv'
    transcription_list = separate_text(in_fpath)

    in_fpath = 'evaluation/s2t_gt/200309_expt58_video_gt_reshaped.csv'
    answer_list = separate_text(in_fpath)

    evaluate = Levenshtein_distance(transcription_list, answer_list)
    result = output_result(evaluate)

    for i in result:
        print(i)


if __name__ == '__main__':
    main()
