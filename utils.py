"""Sentence: store all information for each token in one sentence"""


class Sentence:
    def __init__(self):
        self.id = []
        self.form = []
        self.lemma = []
        self.pos = []
        self.xpos = []
        self.morph = []
        self.head = []
        self.rel = []
        self.gold_arc = []  # manually create, used for evaluation

        #  In test file, 2 column is empty
        self.empty_1 = []
        self.empty_2 = []


'''Reader: read sentences from dataset and build them into sentence structure'''


class Reader:
    def __init__(self, path, model_type, language):
        self.path = path  # path, takes the dataset file
        self.type = model_type  # model_type: train/dev/test
        self.language = language  # language: en or de

        self.sentences = []
        self.sentence_process(self.path)

    def sentence_process(self, path) -> 'file of sentence':
        file = open(path)
        sentence_info = Sentence()
        for line in file:
            if line != "\n":
                tokens = line.split("\t")

                # ROOT should place at index 0 for each sentence
                if tokens[0]=="1":
                    sentence_info.id.append('0')
                    sentence_info.form.append('root')
                    sentence_info.lemma.append('root')
                    sentence_info.pos.append('root_pos')

                    sentence_info.xpos.append('_')
                    sentence_info.morph.append('_')
                    sentence_info.rel.append('_')

                    sentence_info.empty_1.append('_')
                    sentence_info.empty_2.append('_')

                    if self.language == 'deutsch':
                        sentence_info.morph.append('root_morph')  # German: add morph information

                    if self.type == 'train' or 'dev':
                        sentence_info.head.append(-1)  # for train/dev: add -1 for root token

                # finish root, start process other information
                sentence_info.id.append(tokens[0])
                sentence_info.form.append(tokens[1])
                sentence_info.lemma.append(tokens[2])
                sentence_info.pos.append(tokens[3])

                # The following is not that useful, just for reading file in the required format
                sentence_info.xpos.append(tokens[4])
                sentence_info.morph.append(tokens[5])

                # If the process is train and dev, we need gold_arc for training and evaluation
                if self.type == 'train' or 'dev':
                    sentence_info.head.append(int(tokens[6]))
                    sentence_info.gold_arc.append((int(tokens[6]), int(tokens[0])))

                sentence_info.rel.append(tokens[7])
                sentence_info.empty_1.append(tokens[8])
                sentence_info.empty_2.append(tokens[9].strip('\n'))

            else:
                self.sentences.append(sentence_info)  # we need to use "sentences" instance later for the parser
                sentence_info = Sentence()  # empty sentence_info for next sentence


""" Evaluation: evaluate LAS and UAS """


def evaluation(pred_path, gold_path, process, language):
    # Reader class will open the file and store the information
    pred_reader = Reader(pred_path, process, language).sentences
    gold_reader = Reader(gold_path, process, language).sentences

    total_tokens = 0
    total_correct_h = 0
    total_correct_hr = 0

    for pred_sent, gold_sent in zip(pred_reader, gold_reader):
        token_num = len(pred_sent.id) - 1 # minus one because one do not count root
        print(pred_sent.head)
        print(gold_sent.head)
        print(pred_sent.rel)
        print(gold_sent.rel)
        total_tokens += token_num

        # do not count ROOT token so we start from --> index 1
        for pred_h, pred_r, gold_h, gold_r in zip(pred_sent.head[1:], pred_sent.rel[1:], gold_sent.head[1:], gold_sent.rel[1:]):
            if pred_h == gold_h:   # calculate USA
                total_correct_h += 1
                print(pred_h, gold_h)
            if pred_h == gold_h and pred_r == gold_r:  # calculate LSA
                total_correct_hr += 1

    print("Unlabeled Attachment Score: ", str(total_correct_h/ total_tokens))
    print("Labeled Attachment Score:", str(total_correct_hr/total_tokens))

    return None

