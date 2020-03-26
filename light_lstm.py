import random

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Masking
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

feature_dim = 5  # 入力特徴の次元数
output_dim = 3  # 出力データの次元数
batch_size = 300  # ミニバッチサイズ
num_of_training_epochs = 10  # 学習エポック数
learning_rate = 0.001  # 学習率
nb_of_samples = 10000  # 学習データのサンプル数


# ランダムにデータを作成
def create_data():
    # 長さを対数正規分布に従って決める
    leng = np.around(np.random.lognormal(np.log(5.0), 0.5, (nb_of_samples, 1))).astype("int")
    max_sequence_len = leng.max()

    # 乱数で 0 or 1 の列を生成する
    X = np.random.randint(0, 1 + 1, (nb_of_samples, max_sequence_len, feature_dim)).astype("float32")
    # 長さを超えた部分を-1.0に置き換える
    X[np.arange(max_sequence_len).reshape((1, -1)) >= leng] = -1.0

    # -1.0を除いた要素から正解ラベルを作成する
    score1 = np.ma.array(X, mask=(X == -1.0)).mean(axis=-1).mean(axis=-1)
    score2 = np.ma.array(X, mask=(X == -1.0)).max(axis=-1).max(axis=-1)
    score3 = np.ma.array(X, mask=(X == -1.0)).min(axis=-1).min(axis=-1)
    score = np.concatenate((score1[:, np.newaxis], score2[:, np.newaxis], score3[:, np.newaxis]), axis=-1)

    # LSTMに与える入力は (サンプル, 時刻, 特徴量の次元) の3次元になる。
    return X.reshape((nb_of_samples, max_sequence_len, feature_dim)), score


# 乱数シードを固定値で初期化
random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

X, y = create_data()

# モデル構築 #todo タスクに適したモデルの設計
model = Sequential()
# パディングの値を指定してMaskingレイヤーを作成する
model.add(Masking(input_shape=(None, feature_dim), mask_value=-1.0))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(output_dim, activation="sigmoid"))
model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))
model.summary()

save_weight_path = "model/light_lstm.h5"

# 学習
mode = "min"
monitor = "val_loss"
patience = 5
early_stopping = EarlyStopping(patience=patience, verbose=1, monitor=monitor, mode=mode)
checkpoint = ModelCheckpoint(save_weight_path, monitor=monitor, verbose=1, save_best_only=True, save_weights_only=True, mode=mode)
callbacks = [early_stopping, checkpoint]
model.fit(X, y, shuffle=True, batch_size=batch_size, epochs=num_of_training_epochs, validation_split=0.1, verbose=2, callbacks=callbacks)

model.load_weights(save_weight_path)

# 予測
# 任意の長さの入力を受け付ける
test = np.array([[1, 1, 0, 0, 1], [1, 1, 0, 0, 1], [1, 1, 0, 0, 1], [1, 1, 0, 0, 1], [1, 1, 0, 0, 1], [1, 1, 0, 0, 1], [1, 1, 0, 0, 1]]).astype("float32").reshape((1, -1, feature_dim))
print(model.predict(test))
test = np.array([[1, 1, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 1]]).astype("float32").reshape((1, -1, feature_dim))
print(model.predict(test))
test = np.array([[1, 0, 0, 0, 1]]).astype("float32").reshape((1, -1, feature_dim))
print(model.predict(test))
