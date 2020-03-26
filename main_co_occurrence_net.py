from make_co_occurrence_net import *


def run(file_name, min_num_occurrences=2):
    file_id = file_name.split(".")[0]

    tagger = MeCab.Tagger("-Ochasen")
    tagger.parse('')

    lines = read_data(file_name=file_name, dir_path=data_dir_path)

    stopword_list = get_stopword_lsit()

    splited_sentences = get_splited_sentence(tagger=tagger, lines=lines, stopword_list=stopword_list, sep='。', flag_unique=True)

    splited_setences_1d = []
    for sentence in splited_sentences:
        splited_setences_1d.extend(sentence)

    word_and_count = count_word(words=splited_setences_1d)
    words, values = split_word_and_count(word_and_count=word_and_count, n_word=30)

    plot_barh(words, values, figsize=(10, 6), save_file_name='{}_word_count_top30.png'.format(file_id), save_dir_path=image_dir_path)

    n_word = len(word_and_count)
    words, values = split_word_and_count(word_and_count=word_and_count, n_word=n_word)
    print(word_and_count)
    print('文の数：{}, 単語数：{}'.format(len(splited_sentences), n_word))

    #  出現回数ごとの単語数を集計
    df = pd.DataFrame(word_and_count.most_common(n_word), columns=['word', 'count'])
    df_word_count = df.groupby('count').count()
    df_word_count.reset_index(inplace=True)
    df_word_count.columns = ['count', 'n_word']
    df_word_count['cumsum'] = df_word_count['n_word'].cumsum()
    df_word_count['cumsum_rate'] = df_word_count['cumsum'] / df_word_count['n_word'].sum()

    print(df_word_count.head())

    #  単語数を制限
    try:
        n_word = int(n_word * (1 - df_word_count['cumsum_rate'][min_num_occurrences]))
    except KeyError:
        n_word = 1

    bag_of_words, cut_splited_sentences = get_bag_of_words(splited_sentences=splited_sentences,
                                                           n_word=n_word,
                                                           save_file_name='{}_bag_of_words.npy'.format(file_id),
                                                           save_dir_path=data_dir_path)

    #  出現回数による条件を満たした単語群の出力
    output_file = [','.join(sentence) + '\n' for sentence in cut_splited_sentences]
    write_data(data=output_file, file_name=file_name.replace('.txt', '_cut_splited_word.txt'), dir_path=data_dir_path)

    #  アソシエーション分析のためのデータファイルを作成
    output_file = [','.join(sentence) + '\n' for sentence in splited_sentences]
    write_data(data=output_file, file_name=file_name.replace('.txt', '_splited_word.txt'), dir_path=data_dir_path)
    plt.show()

    base_file_name = file_id

    lines = read_data(base_file_name + '_cut_splited_word.txt', dir_path=data_dir_path)

    #  単語リストのテキストを単語に分割してリスト化し、見出しは省いている
    sentences = [line.replace('\n', '').split(',') for line in lines if not ('見出し' in line)]
    #  文内の単語が1語しかない場合は削除
    sentences = [sentence for sentence in sentences if len(sentence) > 1]

    #  データの調査のための関数
    experiment(base_file_name=base_file_name, sentences=sentences)

    #  共起ネットワーク構築の実行
    co_occurrence_network(sentences,
                          n_word_lower=150,
                          edge_threshold=0.01,
                          fig_size=(15, 13),
                          word_jaccard_file_name=base_file_name + '_word_combi_jaccard.csv',
                          result_dir_path=result_dir_path,
                          plot_file_name=base_file_name + '_co_occurrence_network.png',
                          plot_dir_path=image_dir_path)

    plt.show()

    print("finished")


if __name__ == '__main__':
    file_name = 'transcript.txt'
    run(file_name, min_num_occurrences=3)

    for i in range(3):
        file_name = 'transcript_{}.txt'.format(i)
        run(file_name)
