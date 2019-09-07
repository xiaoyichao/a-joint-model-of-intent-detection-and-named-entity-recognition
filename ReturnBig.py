import spacy
import os


base_dir = os.path.dirname(__file__)

# The path of the dataset is modified in there
middle = 'atis'
mid_middle = 'tmp_atis'

# middle = 'snips'
# mid_middle = 'tmp_snips'

# SLUR 的数据不需要转化大小写，原始文件就是应该大写的就大写了，所以不需要执行这个文件
# middle = 'slur'
# mid_middle = 'tmp_slur'

# Load English tokenizer, tagger, parser, NER and word vectors2
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_lg")
GPE_upper_list = []
GPE_lower_list = []

PERSON_upper_list = []
PERSON_lower_list = []


def get_GPE(sentence_list):
    for i, sentence in enumerate(sentence_list):
        sentence_str = ''.join(sentence)
        doc = nlp(sentence_str)
        dic_ner = {}

        for entity in doc.ents:

            # 一个单词的情况
            if len(entity.text.split(' ')) == 1:
                dic_ner[entity.text] = entity.label_
                if entity.label_ == 'GPE' and entity.text not in GPE_upper_list:
                    GPE_upper_list.append(entity.text)
            #    多于一个单词的情况
            if len(entity.text.split(' ')) > 1:
                for word in entity.text.split(' '):

                    if word == entity.text.split(' ')[0]:
                        dic_ner[word] = entity.label_
                        if entity.label_ == 'GPE' and word not in GPE_upper_list:
                            GPE_upper_list.append(word)
                    else:
                        dic_ner[word] = entity.label_
                        if entity.label_ == 'GPE' and word not in GPE_upper_list:
                            GPE_upper_list.append(word)

def get_PERSON(sentence_list):
    for i, sentence in enumerate(sentence_list):
        sentence_str = ''.join(sentence)
        doc = nlp(sentence_str)
        dic_ner = {}

        for entity in doc.ents:

            # 一个单词的情况
            if len(entity.text.split(' ')) == 1:
                dic_ner[entity.text] = entity.label_
                if entity.label_ == 'GPE' and entity.text not in PERSON_upper_list:
                    PERSON_upper_list.append(entity.text)
            #    多于一个单词的情况
            if len(entity.text.split(' ')) > 1:
                for word in entity.text.split(' '):

                    if word == entity.text.split(' ')[0]:
                        dic_ner[word] = entity.label_
                        if entity.label_ == 'GPE' and word not in PERSON_upper_list:
                            PERSON_upper_list.append(word)
                    else:
                        dic_ner[word] = entity.label_
                        if entity.label_ == 'GPE' and word not in PERSON_upper_list:
                            PERSON_upper_list.append(word)

def return_all_big(old_path):
    new_sentences = []
    with open(old_path, 'r') as f:
        lines = f.read().split('\n')
    # 去掉最后一行的空list
    list2 = [x for x in lines if x]
    # 去掉snips数据集中每行结尾和开头可能出现的多余的空格，对atis数据集没有影响
    list2 = map(lambda x: x.rstrip(), list2)
    list2 = map(lambda x: x.lstrip(), list2)
    # 去掉snips数据集中每行内部可能出现的多余的空格，对atis数据集没有影响
    list2 = map(lambda x: x.replace('     ', ' '), list2)  # 5个空格换1个
    list2 = map(lambda x: x.replace('    ', ' '), list2)  # 4个空格换1个
    list2 = map(lambda x: x.replace('   ', ' '), list2)  # 3个空格换1个
    list2 = map(lambda x: x.replace('  ', ' '), list2)  # 2个空格换1个

    sentences = list(map(lambda x: x.split(' '), list2))
    # print(sentences)
    # 测试变成两层list的数据中有没有空格掺杂进来
    for sentence in sentences:
        for word in sentence:
            if word == '':
                sentence_str = ' '.join(sentence)
                print('有空格掺杂', sentence_str)

    #  小写变大写

    for sentence in sentences:
        new_sentence = ''
        for i, word in enumerate(sentence):
            word_upper = word.capitalize()
            if i != len(sentence) - 1:
                new_sentence = new_sentence + word_upper + ' '
            else:
                new_sentence = new_sentence + word_upper
        new_sentences.append(new_sentence)
    return new_sentences


def return_big_lower(sentence_list, new_path):
    with open(new_path, 'w') as newfile:
        for sentence in sentence_list:
            # list转字符串，然哦切分出单词来。
            sentence = str(sentence)
            sentence = sentence.split(' ')
            # print(sentence)
            for i, word in enumerate(sentence):
                # print(word)
                # 如果不在单词在地缘政治列表和人名列表中，就转化成小写
                if word not in (real_GPE_upper_list + real_PERSON_upper_list):
                    new_word = word.lower()
                    # print(new_word)
                    if i == len(sentence) - 1:
                        newfile.write(new_word)
                    else:
                        newfile.write(new_word + ' ')
                else:
                    if i == len(sentence) - 1:
                        newfile.write(word)
                    else:
                        newfile.write(word + ' ')
            newfile.write('\n')

if __name__ == '__main__':

    # 先都转化成大写，看那些会变成政治地缘名词
    train_upper_list = return_all_big(os.path.join(base_dir, middle, 'train/seq.in'))
    valid_upper_list = return_all_big(os.path.join(base_dir, middle, 'valid/seq.in'))
    test_upper_list = return_all_big(os.path.join(base_dir, middle, 'test/seq.in'))
    # print(train_upper_list)
    # 生成政治地缘名词列表
    get_GPE(train_upper_list)
    get_GPE(valid_upper_list)
    get_GPE(test_upper_list)
    # print(GPE_upper_list)
    get_PERSON(train_upper_list)
    get_PERSON(valid_upper_list)
    get_PERSON(test_upper_list)
    # print(len(PERSON_upper_list),PERSON_upper_list)

    # 去除部分单词
    Out_list = ['Which', 'From', 'To', 'The', 'With', 'Using', 'Type', 'Of', 'A',
                    'And', 'Make', 'Connections', 'On', 'Any', 'Making','In','Time',"What",'Off','Flights']
    real_GPE_upper_list = [item for item in GPE_upper_list if item not in set(Out_list)]
    real_PERSON_upper_list = [item for item in PERSON_upper_list if item not in set(Out_list)]


    print(real_PERSON_upper_list)
    print(len(real_GPE_upper_list),len(GPE_upper_list))
    print(len(real_PERSON_upper_list), len(PERSON_upper_list))
    if real_PERSON_upper_list==real_GPE_upper_list:
        print("same")

    # 把不是政治地缘名词和人名的单词变成小写，然后写到新的文件里，让政治地缘名词和人名大写，其他的单词尽量小写。因为政治地缘名词和人名小写spaCy不识别，开头必须大写
    return_big_lower(train_upper_list, os.path.join(base_dir, mid_middle, 'train/seq.in'))
    return_big_lower(valid_upper_list, os.path.join(base_dir, mid_middle, 'valid/seq.in'))
    return_big_lower(test_upper_list, os.path.join(base_dir, mid_middle, 'test/seq.in'))
