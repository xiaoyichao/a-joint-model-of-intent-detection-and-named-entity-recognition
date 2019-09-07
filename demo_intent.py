import numpy as np
import os
import platform
import tensorflow as tf
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Embedding, Input, Concatenate, Flatten, MaxPooling1D, Lambda, GlobalAveragePooling1D, \
    Dropout, TimeDistributed, BatchNormalization, GlobalMaxPool1D, Bidirectional, LSTM
from keras import regularizers, optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras_contrib.layers import CRF
# from matplotlib import pyplot
# from sklearn import metrics
# from get_ip import get_host_ip
import seaborn as sns
import shutil

from keras.initializers import Constant
import keras.backend.tensorflow_backend as KTF

if platform.system() == "Darwin":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    print('MAC')
if platform.system() == "Linux":
    print("Linux")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    KTF.set_session(sess)


base_dir = os.path.dirname(__file__)
emb_dim = 300
max_sequence_length = 50
max_features = 20000
batch_size = 128
learning_rate = 0.25e-4
epochs = 1
save_dir = os.path.join(os.getcwd(), 'demo_intent')
glove_data='glove/glove.6B.300d.txt'

# The path of the dataset is modified in there
middle = 'new_snips_part'
# middle = 'new_atis_part'
# middle = 'new_slur_part'
# middle = 'new_snips_all'
# middle = 'new_atis_all'
# middle = 'new_slur_all'



def load_sentences(path):
    """f返回的的是文本的list嵌套list"""
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    # 去掉最后一行的空list
    lines = [x for x in lines if x]
    sentences = []
    for sentence in lines:
        # print(sentence)
        # 作用一样，一个是自己写的，一个是自带的API
        sent_tmp = text_to_word_sequence(sentence, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', split=' ', lower=True)
        # sent_tmp = sentence.split(' ')
        sentences.append(sent_tmp)
    return sentences


def load_train_labels(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    # 去掉最后一行的空list
    y_intent = [x for x in lines if x]
    # 取第一个标签作为这个数据的标签
    # y_intent = [x.split("#")[0] for x in y_intent ]

    label_set_trian = set(y_intent)
    label_set = set(y_intent)
    if ('atis' in middle) or (middle == 'new_slur_alltest'):
        label_set.add('unknown')
        print("add an unknown class")
    class_num = len(label_set)
    labels_index = {}
    for name in label_set:
        label_id = len(labels_index)
        labels_index[name] = label_id

    y_train_label = []
    for label in y_intent:
        y_id = labels_index[label]
        y_train_label.append(y_id)

    y_train_label = to_categorical(np.asarray(y_train_label), num_classes=class_num)

    print(class_num,labels_index)
    return y_train_label, class_num, labels_index, label_set_trian


def load_test_labels(labels_index, label_set_trian, path):
    "根据给的index，得初valid和test的label 的 ndarray"
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    # 去掉最后一行的空list
    y_intent = [x for x in lines if x]
    # 取第一个标签作为这个数据的标签
    # y_intent = [x.split("#")[0] for x in y_intent ]

    class_num = len(labels_index)

    y_intent_seq = []
    for label in y_intent:
        if (label in label_set_trian):
            y_id = labels_index[label]
        else:
            y_id = labels_index['unknown']

        y_intent_seq.append(y_id)

    y_intent_seq = to_categorical(np.asarray(y_intent_seq), num_classes=class_num)

    return y_intent_seq


# 根据上边的函数，对输入和意图标签进行处理

# 对输入的文本进行处理
train_sentence_seq = load_sentences(os.path.join(base_dir, middle, 'train/seq.in'))
valid_sentence_seq = load_sentences(os.path.join(base_dir, middle, 'valid/seq.in'))
test_sentence_seq = load_sentences(os.path.join(base_dir, middle, 'test/seq.in'))

# glove 的读取
embeddings_index = {}
glove_emb_dim = 300
with open(glove_data) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# token = Tokenizer(num_words=max_features)
token = Tokenizer(num_words=max_features, filters='', oov_token='<UNK>')
token.fit_on_texts([utterance for utterance in train_sentence_seq])

# 文本数值化
x_train_dia_list = token.texts_to_sequences(train_sentence_seq)
x_valid_dia_list = token.texts_to_sequences(valid_sentence_seq)
x_test_dia_list = token.texts_to_sequences(test_sentence_seq)
word_index = token.word_index
# print(word_index)

# 数值化之后padding 过程
x_train_dia_list = sequence.pad_sequences(x_train_dia_list, maxlen=max_sequence_length)
x_valid_dia_list = sequence.pad_sequences(x_valid_dia_list, maxlen=max_sequence_length)
x_test_dia_list = sequence.pad_sequences(x_test_dia_list, maxlen=max_sequence_length)

# 对意图标签进行处理
y_train_label, class_num, labels_index, label_set_trian = load_train_labels(
    os.path.join(base_dir, middle, 'train/label'))
y_valid_label = load_test_labels(labels_index, label_set_trian, os.path.join(base_dir, middle, 'valid/label'))
y_test_label = load_test_labels(labels_index, label_set_trian, os.path.join(base_dir, middle, 'test/label'))


# 意图标签和文本信息都处理好了


# prepare embedding matrix
num_words = min(max_features, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, glove_emb_dim))
for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
glove_embedding_layer = Embedding(num_words,
                                  glove_emb_dim,
                                  embeddings_initializer=Constant(embedding_matrix),
                                  input_length=max_sequence_length,
                                  trainable=True, name='glove_embedding', mask_zero=True)

# 构建model

char_input = Input(shape=[max_sequence_length,], name='input_layer', dtype='int32')
word_emb = glove_embedding_layer(char_input)
word_emb = Dropout(0.5, name='drop4embedding')(word_emb)
word_emb = BatchNormalization(name = 'BatchNormalization4em')(word_emb)
encoder_output = Bidirectional(LSTM(200, return_sequences=True), name='encoder')(word_emb)
# encoder4SF = TimeDistributed(Dropout(0.3),name='timedistributed4encoder')(encoder_output)
#
# crf = CRF(units=tags_num, learn_mode='join',
#           test_mode='viterbi', sparse_target=False,name='ner')
# crf_outputs = crf(encoder4SF)


encoder4ID = Dropout(0.3, name='drop_ID')(encoder_output)
encoder4ID = BatchNormalization(name='BatchNormalization4ID')(encoder4ID)

h = Bidirectional(LSTM(200, return_sequences=False, return_state=False), name='BiLSTM_decoder1')(encoder4ID)
# h = LSTM(units=200, return_sequences=False, return_state=False, name='BiLSTM_decoder1')(encoder4ID)

intent_output = Dense(units=class_num, activation='softmax', kernel_regularizer=regularizers.l2(0.0001), name='intent')(
    h)

model = Model(input=char_input, output=intent_output)
print(model.summary())


try:
    shutil.rmtree(save_dir)
except:
    pass

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

Adam = optimizers.Adam(lr=learning_rate)

filepath = "model_{epoch:02d}_{val_acc:.5f}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_acc', verbose=1,
                             save_best_only=True)

model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=["accuracy"])

plot_model(model, to_file=os.path.join(base_dir, middle, 'model_intent.png'), show_layer_names=True, show_shapes=True)

history = model.fit(x=x_train_dia_list, y=y_train_label,
                    validation_data=(x_test_dia_list, y_test_label),
                    batch_size=batch_size, epochs=epochs,
                    callbacks=[checkpoint], shuffle=True)

# pyplot.figtext(0.1, 0.92, 'demo_intent', color='green')
# # pyplot.subplot(121)
# # pyplot.plot(history.history['acc'], label='train_acc')
# # pyplot.plot(history.history['val_acc'], label='val_acc')
# # pyplot.xlabel('Epochs')
# #
# # pyplot.subplot(122)
# pyplot.plot(history.history['loss'], label='train_loss')
# pyplot.plot(history.history['val_loss'], label='val_loss')
# pyplot.xlabel('Epochs')
# pyplot.legend()
# pyplot.show()

files = os.listdir(save_dir)
files.sort(key=lambda x: float(x[-11:-5]))
print(files[-1])

model.load_weights(os.path.join(save_dir, files[-1]))

# loss, score = model.evaluate(
#     x=x_valid_dia_list, y=y_valid_label, batch_size=batch_size)
# print("val_loss：{loss}, val_score {score}".format(loss=loss, score=score))

# test_pre = model.predict(x_test_dia_list)
#
# ## 评价预测效果，计算混淆矩阵
# confm = metrics.confusion_matrix(np.argmax(test_pre, axis=1), np.argmax(y_test_label, axis=1))
# ## 混淆矩阵可视化
# Labname = label_set_trian
# pyplot.figure(figsize=(9, 9),dpi=700)
# sns.heatmap(confm.T, square=True, annot=True,
#             fmt='d', cbar=False, linewidths=.8,
#             cmap="YlGnBu")
# pyplot.xlabel('True label', size=30)
# pyplot.ylabel('Predicted label', size=30)
# pyplot.xticks(np.arange(10) + 0.5, Labname, size=5)
# pyplot.yticks(np.arange(10) + 0.5, Labname, size=5)
# pyplot.show()
#
# print(metrics.classification_report(np.argmax(test_pre, axis=1), np.argmax(y_test_label, axis=1)))

try:
    shutil.rmtree(save_dir)
except:
    pass
