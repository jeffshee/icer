# -*- coding: utf-8 -*-
from pydub import AudioSegment

# mp3ファイルの読み込み
sound = AudioSegment.from_file("wavs/Take05.wav", format="wav")

# 5000ms～10000ms(5～10秒)を抽出
sound1 = sound[:50000]

# 抽出した部分を出力
sound1.export("wavs/Take05_Trim.wav", format="wav")
