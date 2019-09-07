import spacy
import os
from data_utils import load_sentences4spacy, load_labels

base_dir = os.path.dirname(__file__)

# The path of the dataset is modified in there
# 获取当前文件目录,替换middle参数即可。
# new_middle = 'new_atis_part'
# middle = 'atis'
# mid_middle = 'tmp_atis'

# new_middle = 'new_atis_all'
# middle = 'atis'
# mid_middle = 'tmp_atis'

# new_middle = 'new_snips_all'
# middle = 'snips'
# mid_middle = 'tmp_snips'

# new_middle = 'new_snips_part'
# middle = 'snips'
# mid_middle = 'tmp_snips'

# new_middle = 'new_slur_all'
# middle = 'slur'
# mid_middle = 'tmp_slur'

new_middle = 'new_slur_part'
middle = 'slur'
mid_middle = 'tmp_slur'


train_ner_path = os.path.join(base_dir, middle, 'train/seq.ner')
new_train_in_path = os.path.join(base_dir, new_middle, 'train/seq.in')
new_train_ner_path = os.path.join(base_dir, new_middle, 'train/seq.ner')
new_train_out_path = os.path.join(base_dir, new_middle, 'train/seq.out')
new_train_label_path = os.path.join(base_dir, new_middle, 'train/label')

valid_ner_path = os.path.join(base_dir, middle, 'valid/seq.ner')
new_valid_in_path = os.path.join(base_dir, new_middle, 'valid/seq.in')
new_valid_ner_path = os.path.join(base_dir, new_middle, 'valid/seq.ner')
new_valid_out_path = os.path.join(base_dir, new_middle, 'valid/seq.out')
new_valid_label_path = os.path.join(base_dir, new_middle, 'valid/label')

test_ner_path = os.path.join(base_dir, middle, 'test/seq.ner')
new_test_in_path = os.path.join(base_dir, new_middle, 'test/seq.in')
new_test_ner_path = os.path.join(base_dir, new_middle, 'test/seq.ner')
new_test_out_path = os.path.join(base_dir, new_middle, 'test/seq.out')
new_test_label_path = os.path.join(base_dir, new_middle, 'test/label')

# Load English tokenizer, tagger, parser, NER and word vectors2
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_lg")


def ner_text(sentence_list, new_ner_path):  # 去掉空NER行，做数据的筛选，生成新的数据集
    j = 0  # j 用来统计全是空NER标签的句子数量，进而计算比例
    need_list = []
    with open(new_ner_path, 'w') as seqner:
        for i, sentence in enumerate(sentence_list):
            sentence_str = ' '.join(sentence)
            doc = nlp(sentence_str)
            dic_ner = {}
            for entity in doc.ents:
                # print(entity.text, ':', entity.label_)
                # print(len(entity.text.split(' ')))
                # 一个单词的情况
                if len(entity.text.split(' ')) == 1:
                    dic_ner[entity.text] = entity.label_
                #    对于一个单词的情况
                if len(entity.text.split(' ')) > 1:
                    for word in entity.text.split(' '):
                        if word == entity.text.split(' ')[0]:
                            dic_ner[word] = entity.label_
                        else:
                            dic_ner[word] = entity.label_

            if doc.ents:  # 如果这行数据有NER标签,则写入数据，否则这条数据忽略
                #     for entity in doc.ents:
                # print(entity.text,entity.label_)
                j = j + 1
                need_list.append(i)  # 有NER数据的时候就把这个数据的ID保存下来
                for position, word in enumerate(sentence):
                    # print(position, len(sentence) - 1)
                    if word in dic_ner:
                        if position != len(sentence) - 1:
                            seqner.write((dic_ner[word] + ' '))
                        else:
                            seqner.write((dic_ner[word]))

                    else:  # 否则写入O
                        if position != len(sentence) - 1:
                            seqner.write('O' + ' ')
                        else:
                            seqner.write('O')

                seqner.write('\n')
    print('非空NER数据的句子占文件中所有句子的比例{:.1%}： '.format(j / len(sentence_list)), j, len(sentence_list))
    return need_list


def ner_text_all(sentence_list, new_ner_path):  # 不！！！去掉空NER行，生成新的数据集
    j = 0  # j 用来统计全是空NER标签的句子数量，进而计算比例
    need_list = []
    with open(new_ner_path, 'w') as seqner:
        for i, sentence in enumerate(sentence_list):
            sentence_str = ' '.join(sentence)
            doc = nlp(sentence_str)
            dic_ner = {}
            # 曾经不合适的字典
            # for entity in doc.ents:
            #     for word in entity.text.split(' '):
            #         dic_ner[word] = entity.label_
            # print(dic_ner)
            for entity in doc.ents:
                # print(entity.text, ':', entity.label_)
                # print(len(entity.text.split(' ')))
                # 一个单词的情况
                if len(entity.text.split(' ')) == 1:
                    dic_ner[entity.text] = entity.label_
                #    对于一个单词的情况
                if len(entity.text.split(' ')) > 1:
                    for word in entity.text.split(' '):
                        if word == entity.text.split(' ')[0]:
                            dic_ner[word] = entity.label_
                        else:
                            dic_ner[word] = entity.label_

            if doc.ents:  # 如果这行数据有NER标签,则写入数据，否则这条数据忽略
                #     for entity in doc.ents:
                # print(entity.text,entity.label_)
                j = j + 1
                need_list.append(i)  # 有NER数据的时候就把这个数据的ID保存下来
                for position, word in enumerate(sentence):
                    # print(position, len(sentence) - 1)
                    if word in dic_ner:
                        if position != len(sentence) - 1:
                            seqner.write((dic_ner[word] + ' '))
                        else:
                            seqner.write((dic_ner[word]))

                    else:  # 否则写入O
                        if position != len(sentence) - 1:
                            seqner.write('O' + ' ')
                        else:
                            seqner.write('O')

                seqner.write('\n')
            else:
                j = j + 1
                need_list.append(i)  # 有NER数据的时候就把这个数据的ID保存下来
                for position, word in enumerate(sentence):
                    # print(position, len(sentence) - 1)
                    if word in dic_ner:
                        if position != len(sentence) - 1:
                            seqner.write((dic_ner[word] + ' '))
                        else:
                            seqner.write((dic_ner[word]))

                    else:  # 否则写入O
                        if position != len(sentence) - 1:
                            seqner.write('O' + ' ')
                        else:
                            seqner.write('O')

                seqner.write('\n')

    print('非空NER数据的句子占文件中所有句子的比例{:.1%}： '.format(j / len(sentence_list)), j, len(sentence_list))
    return need_list


def generat_new_data(need_list, old_list, new_path):
    # print(old_list)
    with open(new_path, 'w')  as new:
        for i, id in enumerate(need_list):
            new.write(str((old_list[id])).strip() + '\n')
    print("ok")


if __name__ == '__main__':
    train_sentence_list = load_sentences4spacy(
        os.path.join(base_dir, mid_middle, 'train/seq.in'))
    valid_sentence_list = load_sentences4spacy(
        os.path.join(base_dir, mid_middle, 'valid/seq.in'))
    test_sentence_list = load_sentences4spacy(
        os.path.join(base_dir, mid_middle, 'test/seq.in'))
    if 'all' in new_middle:
        # 生成新的数据seq.ner
        print('all ', new_middle)
        need_list_train = ner_text_all(train_sentence_list, new_train_ner_path)
        need_list_valid = ner_text_all(valid_sentence_list, new_valid_ner_path)
        need_list_test = ner_text_all(test_sentence_list, new_test_ner_path)
    else:
        print('part ', new_middle)
        need_list_train = ner_text(train_sentence_list, new_train_ner_path)
        need_list_valid = ner_text(valid_sentence_list, new_valid_ner_path)
        need_list_test = ner_text(test_sentence_list, new_test_ner_path)

    train_labels = load_labels(os.path.join(base_dir, middle, 'train/label'))
    valid_labels = load_labels(os.path.join(base_dir, middle, 'valid/label'))
    test_labels = load_labels(os.path.join(base_dir, middle, 'test/label'))
    # # print(len(need_list_test), need_list_test)
    generat_new_data(need_list_train, train_labels, new_train_label_path)
    generat_new_data(need_list_valid, valid_labels, new_valid_label_path)
    generat_new_data(need_list_test, test_labels, new_test_label_path)

    train_in = load_labels(os.path.join(base_dir, middle, 'train/seq.in'))
    valid_in = load_labels(os.path.join(base_dir, middle, 'valid/seq.in'))
    test_in = load_labels(os.path.join(base_dir, middle, 'test/seq.in'))

    generat_new_data(need_list_train, train_in, new_train_in_path)
    generat_new_data(need_list_valid, valid_in, new_valid_in_path)
    generat_new_data(need_list_test, test_in, new_test_in_path)

    train_out = load_labels(os.path.join(base_dir, middle, 'train/seq.out'))
    valid_out = load_labels(os.path.join(base_dir, middle, 'valid/seq.out'))
    test_out = load_labels(os.path.join(base_dir, middle, 'test/seq.out'))

    generat_new_data(need_list_train, train_out, new_train_out_path)
    generat_new_data(need_list_valid, valid_out, new_valid_out_path)
    generat_new_data(need_list_test, test_out, new_test_out_path)
