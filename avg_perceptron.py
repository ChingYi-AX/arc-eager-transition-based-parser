import numpy as np
import pickle
import random


'''AvgPerceptron: use feature template and weight from FeatureMapping'''


class AvgPerceptron:
    def __init__(self, feature_map, weight):
        pass
        self.map = feature_map  # map will be used in parser
        self.weight = weight

    def train(self, training_data):
        print("----------------Training Process----------------")
        u = np.zeros(self.weight.shape, dtype=np.float32)  # cached weight
        q = 0  # example counter
        for epoch in range(0, 1):
            # shuffle for each epochs
            random.shuffle(training_data, random.random)

            total = 0
            correct = 0
            print("Epoch: ", epoch + 1)

            for data in training_data:
                score = np.asarray([0., 0., 0., 0.])
                q += 1  # increment regardless of update
                total += 1

                for ix in data.feature_vector:
                    for i in range(0, 4):
                        score[i] += self.weight[i][ix]
                predict_y = np.argmax(score)

                if predict_y != data.label:
                    for ix in data.feature_vector:
                        self.weight[data.label][ix] += 1
                        u[predict_y][ix] -= q

                        self.weight[predict_y][ix] -= 1
                        u[data.label][ix] += q

                if predict_y == data.label:
                    correct += 1

                if total % 10000 == 0:
                    print("States", total, ": ", (correct/total))

            train_accuracy = correct / len(training_data)
            print("Training Accuracy: ", train_accuracy)

        self.weight -= u * (1 / q)  # average all weight vectors seen during training
        # print(self.weight)
        return self.weight

    def save_model(self, model, language):
        file_name = 'model_' + language
        file = open(file_name, 'wb')
        pickle.dump(model, file)
        file.close()

