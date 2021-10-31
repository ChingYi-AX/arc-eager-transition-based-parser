import argparse
from utils import Reader
import numpy as np
from feature_engineer import FeatureMapping
from avg_perceptron import AvgPerceptron
from parser import State, Parsing
import pickle


def load_model(model_name, language):
    model_file = model_name + '_' + language
    file = open(model_file, 'rb')
    model = pickle.load(file)
    file.close()
    return model


'''Process argument: --model --language --process --file_path'''

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--model', dest='m', default="model", type=str)
arg_parser.add_argument('--language', dest='lang', default="en", type=str)
arg_parser.add_argument('--process', dest='process', type=str)
arg_parser.add_argument('--file_path', dest='f', type=str)

args = arg_parser.parse_args()
model_type = args.m
language = args.lang
process = args.process
file_path = args.f

initial_stack = np.arange(1)

'''test----only write file for testing process'''

if process == 'test':
    model = load_model(model_type, language)
    sentences = Reader(file_path, process, language).sentences
    file_name = language + 'result' + '.conll06'
    output_file = open(file_name, 'w')

    for sent in sentences:
        initial_buffer = np.arange(1, len(sent.form))
        state = State(initial_stack, initial_buffer)
        pred_arcs = Parsing(state, sent, model.map).parse(model)

        for arc in range(1, len(pred_arcs)):
            ''' If the language is German, we have to write morph information'''
            if language == 'english':
                output_file.write(
                    str(arc) + "\t" + sent.form[arc] + "\t" + sent.lemma[arc] + "\t" + sent.pos[
                        arc] + "\t" + "_" + "\t" + "_" + "\t" + str(
                        pred_arcs[arc]) + "\t" + "_" + "\t" + "_" + "\t" + "_")

            elif language == 'deutsch':
                output_file.write(
                    str(arc) + "\t" + sent.form[arc] + "\t" + sent.lemma[arc] + "\t" + sent.pos[
                        arc] + "\t" + "_" + "\t" + sent.morph[arc] + "\t" + str(
                        pred_arcs[arc]) + "\t" + "_" + "\t" + "_" + "\t" + "_")
            else:
                print("Sorry, we are not capable of processing file in this language.")
            output_file.write('\n')
        output_file.write('\n')
    output_file.close()

'''dev----print dev accuracy for evaluation '''

if process == 'dev':
    model = load_model(model_type, language)
    sentences = Reader(file_path, process, language).sentences
    total = 0
    correct = 0
    for sent in sentences:
        initial_buffer = np.arange(1, len(sent.form))
        state = State(initial_stack, initial_buffer)
        pred_arcs = Parsing(state, sent, model.map).parse(model)

        '''checking each arc, do not consider root token (first word)'''
        for gold_arc, pred_arc in zip(sent.head[1:], pred_arcs[1:]):
            if gold_arc == pred_arc :
                correct += 1
            total += 1
    print("Dev Accuracy: ", str(correct/total))


''' train----print the accuracy every 10000 state'''

if process == 'train':
    feature_map = FeatureMapping()  # establish and populate feature mapping
    sentences = Reader(file_path, process, language).sentences
    training_data = []

    for sent in sentences:
        initial_buffer = np.arange(1, len(sent.form))
        state = State(initial_stack, initial_buffer)
        parser_data = Parsing(state, sent, feature_map).oracle_parser()
        training_data.append(parser_data)

    # train and save model
    feature_map.frozen = True  # freeze feature mapping
    weight = np.zeros((4, feature_map.id), dtype=np.float32)  # initialize weight matrix
    training_data = np.concatenate(training_data).flatten().tolist()  # data for training model

    model = AvgPerceptron(feature_map, weight)
    model.train(training_data)
    model.save_model(model, language)

