import math
import random
import matplotlib.pyplot as plt
import json
import numpy as np


def get_random(a, b):
    result = (b - a) * random.random() + a
    return result


def fill_matrix(a, b, fill):
    output = []
    for i in range(a):
        temp = []
        for j in range(b):
            temp.append(get_random(-fill, fill))
        output.append(temp)
    return output


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


def d_relu(x):
    if x > 0:
        return 1.0
    else:
        return 0.1


def relu(x):  # leacky relu
    if x > 0:
        return x
    else:
        return 0.1 * x


def active(x, ACTIVATE_FUNCTION):
    if ACTIVATE_FUNCTION == 'relu':
        return relu(x)
    elif ACTIVATE_FUNCTION == 'tanh':
        return math.tanh(x)
    elif ACTIVATE_FUNCTION == 'sigmoid':
        return sigmoid(x)
    else:
        return relu(x)


def d_active(x, ACTIVATE_FUNCTION):
    if ACTIVATE_FUNCTION == 'relu':
        return d_relu(x)
    elif ACTIVATE_FUNCTION == 'tanh':
        return 1 - ((math.tanh(x)) ** 2)
    elif ACTIVATE_FUNCTION == 'sigmoid':
        return d_sigmoid(x)
    else:
        return d_relu(x)


class MLP(object):
    def __init__(self):
        self.Num_of_Inputs = 0  # number of inputs
        self.Num_of_Hidden = 0  # number of hidden layers
        self.Num_of_Outputs = 0  # number of outputs

        self.Input_Neurons = []  # Neurons
        self.Hidden_Neurons = []
        self.Output_Neurons = []

        # self.Input_Bias = 1  # bias
        # self.Output_Bias = 1

        self.Input_Weights = []  # weights
        self.Output_Weights = []

        self.Prediction = []

        self.ACTIVATE_FUNCTION = 'sigmoid'

    def build(self, n_input, n_hidden, n_output, ACTIVATE_FUNCTION='sigmoid', initial_value=1.0, fill_w1=0.2,
              fill_w2=2.0):
        self.Num_of_Inputs = n_input + 1
        self.Num_of_Hidden = n_hidden + 1
        self.Num_of_Outputs = n_output

        # used for Q1&Q3
        # self.Input_Neurons = [1.0] * self.Num_of_Inputs

        # used for special
        self.Input_Neurons = [initial_value] * self.Num_of_Inputs

        # print(self.Input_Neurons)

        # used for Q1&Q3
        # self.Hidden_Neurons = [1.0] * self.Num_of_Hidden

        # used for special
        self.Hidden_Neurons = [initial_value] * self.Num_of_Hidden

        # print(self.Hidden_Neurons)

        # used for Q1&Q3
        # self.Output_Neurons = [1.0] * self.Num_of_Outputs

        # used for special
        self.Output_Neurons = [initial_value] * self.Num_of_Outputs
        # print(self.Output_Neurons)

        # use for special test
        self.Input_Weights = fill_matrix(self.Num_of_Inputs, self.Num_of_Hidden, fill=fill_w1)
        # use for Q1 & Q3
        # self.Input_Weights = fill_matrix(self.Num_of_Inputs, self.Num_of_Hidden, fill=0.2)
        # print(self.Input_Weights)
        # use for special test
        self.Output_Weights = fill_matrix(self.Num_of_Hidden, self.Num_of_Outputs, fill=fill_w2)
        # self.Output_Weights = fill_matrix(self.Num_of_Hidden, self.Num_of_Outputs, fill=2.0)
        # print(self.Output_Weights)

        self.ACTIVATE_FUNCTION = ACTIVATE_FUNCTION
        self.Prediction = []

    def forward(self, inputs):
        for i in range(self.Num_of_Inputs - 1):
            self.Input_Neurons[i] = inputs[i]
            # print(self.Input_Neurons)

        for i in range(self.Num_of_Hidden):
            h = 0.0  # sum
            for j in range(self.Num_of_Inputs):
                h += self.Input_Neurons[j] * self.Input_Weights[j][i]  # do the add on job
            h = active(h, self.ACTIVATE_FUNCTION)
            self.Hidden_Neurons[i] = h
        # print(self.Hidden_Neurons)

        for i in range(self.Num_of_Outputs):
            o = 0.0  # sum
            for j in range(self.Num_of_Hidden):
                o += self.Hidden_Neurons[j] * self.Output_Weights[j][i]
            o = active(o, self.ACTIVATE_FUNCTION)
            self.Output_Neurons[i] = o
        # print(self.Output_Neurons)

        self.Prediction = self.Output_Neurons

    def backwards(self, label, learning_rate):
        delta_output = [0.0] * self.Num_of_Outputs
        # output delta
        for i in range(self.Num_of_Outputs):
            error = label[i] - self.Output_Neurons[i]
            # print(error)
            delta_output[i] = d_active(self.Output_Neurons[i], self.ACTIVATE_FUNCTION) * error
            # print(delta_output[i])
        # print('do:' + str(delta_output))

        # hidden delta
        # refresh weights (hidden -> output)
        delta_hidden = [0.0] * self.Num_of_Hidden
        for i in range(self.Num_of_Hidden):
            error = 0.0
            for j in range(self.Num_of_Outputs):
                error += delta_output[j] * self.Output_Weights[i][j]
                change = delta_output[j] * self.Hidden_Neurons[i]
                self.Output_Weights[i][j] += change * learning_rate
            delta_hidden[i] = d_active(self.Hidden_Neurons[i], self.ACTIVATE_FUNCTION) * error
        # print(self.Output_Weights)
        # print('dh: ' + str(delta_hidden))

        # refresh weights (input -> hidden)
        for i in range(self.Num_of_Inputs):
            for j in range(self.Num_of_Hidden):
                change = delta_hidden[j] * self.Input_Neurons[i]
                self.Input_Weights[i][j] += change * learning_rate

        # get error
        out_error = 0.0
        for i in range(self.Num_of_Outputs):
            # print(label[i])
            # print(self.Output_Neurons[i])
            out_error += 0.5 * ((label[i] - self.Output_Neurons[i]) ** 2)
        # print('Error: ' + str(error))
        # print(self.Input_Weights)
        # print(self.Output_Weights)
        return out_error

    def predict(self, inputs):
        self.forward(inputs)
        # print(self.Hidden_Neurons)
        # print(self.Output_Neurons)
        # print(self.Input_Weights)
        # print(self.Output_Weights)
        return self.Prediction

    def train(self, data, labels, epoch, learning_rate=0.05):
        error_per_epoch = []
        output_data = {}
        # plt.xlabel('EPOCH')
        # plt.ylabel('ERROR')
        for i in range(epoch):
            error = []
            for j in range(len(data)):
                self.forward(data[j])
                error.append(self.backwards(labels[j], learning_rate))
            # print(self.Hidden_Neurons)
            # print(self.Output_Neurons)
            # print(self.Input_Weights)
            # print(self.Output_Weights)
            # print(error)
            error_per_epoch.append(np.mean(error))
            output_data['epoch_' + str(i)] = np.mean(error)
            print('Error = ' + str(np.mean(error)))
            print('epoch' + str(i) + ' finished')
        with open(self.ACTIVATE_FUNCTION + '.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        f.close()
        # print(error_per_epoch)
        # plt.show()
        return error_per_epoch


def Q1():
    data = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    labels = [[0], [1], [1], [0]]

    mlp = MLP()
    mlp.build(2, 4, 1, ACTIVATE_FUNCTION='sigmoid')
    # mlp.build(2, 4, 1, ACTIVATE_FUNCTION='relu')
    # mlp.build(2, 4, 1, ACTIVATE_FUNCTION='tanh')
    error = mlp.train(data, labels, 10000, 0.1)
    plt.title('Sigmoid')
    # plt.title('Relu')
    # plt.title('Tanh')
    plt.xlabel('EPOCH')
    plt.ylabel('ERROR')
    plt.plot(error, 'r-')
    plt.show()

    with open('Q1_result', 'w') as f:
        i = 0
        for d in data:
            json.dump({str(i): mlp.predict(d)}, f, indent=2)
            i += 1


def show_test_error(result, label):
    error = 0
    for i in range(len(result)):
        error += 0.5 * ((result[i] - label[i]) ** 2)
    print('ERROR on the test set:' + str(error))


def show_error(result, label):
    error = 0
    for i in range(len(result)):
        temp = np.array(result[i]['result']) - np.array(label[i])
        for d in temp:
            error += 0.5 * (d ** 2)

    print('ERROR on the test set:' + str(error))


def make_graph():
    with open('Q3_result_train.json', 'r') as f:
        train_result = f.readlines()
        train_result = [float(e.replace('[', '').replace(']', '').replace('\n', '')) for e in train_result]

    train_labels = json.load(open('train_Q3.json'))['labels']
    plt.plot(train_labels, 'r*')
    plt.plot(train_result, 'b.')
    plt.show()
    plt.close()

    with open('Q3_result_test.json', 'r') as f:
        test_result = f.readlines()
        test_result = [float(e.replace('[', '').replace(']', '').replace('\n', '')) for e in test_result]

    test_labels = json.load(open('test_Q3.json'))['labels']
    plt.plot(test_labels, 'r*')
    plt.plot(test_result, 'b.')
    plt.show()
    plt.close()
    # result = json.load(open('Q3_result.json'))
    # print(test_labels)
    # print(result)
    show_test_error(test_result, test_labels)


def Q3():
    # data_preprocess_q3()
    train_data = json.load(open('train_Q3.json'))['data']
    train_labels = json.load(open('train_Q3.json'))['labels']
    labels = []
    for label in train_labels:
        labels.append([label])
    mlp = MLP()
    # mlp.build(4, 7, 1, ACTIVATE_FUNCTION='sigmoid')
    mlp.build(4, 7, 1, ACTIVATE_FUNCTION='relu')
    # mlp.build(4, 7, 1, ACTIVATE_FUNCTION='tanh')
    error = mlp.train(train_data, labels, 10000, 0.05)
    # 将预测点和label点分别用红蓝打印出来
    # 将预测点和label点的差值的绝对值统计出来
    # plt.title('Sigmoid')
    plt.title('Relu')
    # plt.title('Tanh')
    plt.xlabel('EPOCH')
    plt.ylabel('ERROR')
    plt.plot(error, 'r-')
    plt.show()

    test_data = json.load(open('test_Q3.json'))['data']
    # test_labels = json.load(open('test_Q3.json'))['labels']
    with open('Q3_result_train.json', 'w') as f:
        i = 0
        for d in train_data:
            f.write(str(mlp.predict(d)) + '\n')
            i += 1
        f.close()

    with open('Q3_result_test.json', 'w') as f:
        i = 0
        for d in test_data:
            f.write(str(mlp.predict(d)) + '\n')
            i += 1
        f.close()
    # result = np.array(result)
    # test_labels = np.array(test_labels)
    # distance = result - test_labels
    make_graph()

    # x = []
    # for d in train_data:
    #     x.append(d[0]-d[1]+d[2]-d[3])
    #
    # points = []
    # for i in range(149):
    #     points.append((x[i], train_labels[i]))
    # print(train_data)
    # print(train_labels)
    # train_labels = sorted(train_labels)
    # plt.plot(points, 'r.')
    # plt.show()


def data_preprocess_q3():
    data = []
    label = []
    for i in range(199):
        x1 = get_random(-1.0, 1.0)
        x2 = get_random(-1.0, 1.0)
        x3 = get_random(-1.0, 1.0)
        x4 = get_random(-1.0, 1.0)
        data.append([x1, x2, x3, x4])
        # print([x1, x2, x3, x4])
        label.append(math.sin(x1 - x2 + x3 - x4))
        # print(math.sin(x1 - x2 + x3 - x4))

    # print(data)
    # print(label)
    q3_train = {'data': data[:150],
                'labels': label[:150]}

    print(q3_train)

    q3_test = {'data': data[150:],
               'labels': label[150:]}

    print(q3_test)

    with open('train_Q3.json', 'w') as f:
        json.dump(q3_train, f, indent=2)

    with open('test_Q3.json', 'w') as f:
        json.dump(q3_test, f, indent=2)


def get_letter(output):
    position = 0
    max = output[0]
    for i in range(len(output)):
        if output[i] < max:
            continue
        else:
            max = output[i]
            position = i
            continue
    return chr(position + 65)


def data_preprocess_special():
    with open('letter-recognition.data', 'r') as f:
        data = f.readlines()
        data = [d.replace('\n', '') for d in data]
        data = [d.split(',') for d in data]

    label_letter = []
    full_data = []
    for d in data:
        label_letter.append(d[0])
        full_data.append(d[1:])

    full_data = [[int(number) / 10 for number in example] for example in full_data]
    # print(full_data)
    label_in_list = []

    for letter in label_letter:
        temp = [0.0001] * 26
        temp[ord(letter) - 65] = 1.0
        label_in_list.append(temp)

    return full_data, label_in_list, label_letter
    # print(label_in_list)
    # for i in range(len(label_letter)):
    #     print(label_letter[i] + ' | ' + get_letter(label_in_list[i]))


def statistic_accuracy():
    # with open('special1/Special_train_result.txt', 'r') as f:
    with open('special2/Special_train_result.txt', 'r') as f:
        train = f.readlines()
    # with open('special1/Special_test_result.txt', 'r') as f:
    with open('special2/Special_test_result.txt', 'r') as f:
        test = f.readlines()

    c_train = 0
    w_train = 0
    c_test = 0
    w_test = 0
    for d in train[1:]:
        d = d.replace(" ", '')
        d = d.replace("\n", '')
        temp = d.split("|")
        # print(temp)
        if temp[0] == temp[1]:
            c_train += 1
        else:
            w_train += 1

    for d in test[1:]:
        d = d.replace(" ", '')
        d = d.replace("\n", '')
        temp = d.split("|")
        if temp[0] == temp[1]:
            c_test += 1
        else:
            w_test += 1

    # with open('special1/accuracy.txt', 'w') as f:
    with open('special2/accuracy.txt', 'w') as f:
        f.write(
            'On training set, in ' + str(c_train + w_train) + ' samples, Correct = ' + str(c_train) + ' Wrong = ' + str(
                w_train) + ' Accuracy = ' + str(c_train / (c_train + w_train)) + '\n')
        f.write('On testing set, in ' + str(c_test + w_test) + ' samples, Correct = ' + str(c_test) + ' Wrong = ' + str(
            w_test) + ' Accuracy = ' + str(c_test / (c_test + w_test)) + '\n')


def special_question():
    full_data, label_in_list, label_letter = data_preprocess_special()

    train_data = full_data[:16000]
    train_label_list = label_in_list[:16000]
    train_label_letter = label_letter[:16000]

    test_data = full_data[16000:]
    test_label_list = label_in_list[16000:]
    test_label_letter = label_letter[16000:]

    test_result = []
    train_result = []

    mlp = MLP()
    mlp.build(16, 12, 26, ACTIVATE_FUNCTION='sigmoid', initial_value=0.1, fill_w1=0.1, fill_w2=0.01)
    error = mlp.train(train_data, train_label_list, 2000, 0.025)

    plt.title('Sigmoid')
    # plt.title('Relu')
    # plt.title('Tanh')
    plt.xlabel('EPOCH')
    plt.ylabel('ERROR')
    plt.plot(error, 'r-')
    plt.show()

    with open('Special_train_result.txt', 'w') as f:
        f.write('predict | label\n')
        for i in range(len(train_data)):
            dic = {'result': mlp.predict(train_data[i])}
            train_result.append(dic)
            f.write(get_letter(mlp.predict(train_data[i])) + ' | ' + train_label_letter[i] + '\n')

    with open('Special_test_result.txt', 'w') as f:
        f.write('predict | label')
        for i in range(len(test_data)):
            dic = {'result': mlp.predict(test_data[i])}
            test_result.append(dic)
            f.write(get_letter(mlp.predict(test_data[i])) + ' | ' + test_label_letter[i] + '\n')

    show_error(train_result, train_label_list)
    show_error(test_result, test_label_list)


# Q1()
# Q3()
# data_preprocess_q3()
# test()
# make_graph()
# data_preprocess_special()
# special_question()
statistic_accuracy()