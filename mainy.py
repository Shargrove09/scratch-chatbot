import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import random
import tflearn
import os

nltk.download('punkt')

with open("data.json") as json_data:
    data = json.load(json_data)

    # print(data)

words = []
documents = []
classes = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word = nltk.word_tokenize(pattern)

        words.extend(word)
        documents.append((word, intent["tag"]))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

stemmer = LancasterStemmer()
words_lowercase = [stemmer.stem(word.lower())
                   for word in words]  # Puts all words into lowercase

words = sorted(list(set(words_lowercase)))

training_data = []
empty_output = [0] * len(classes)

for document in documents:
    bag_of_words = []

    pattern_words = document[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    for word in words:
        bag_of_words.append(
            1) if word in pattern_words else bag_of_words.append(0)

    # 0 for each tag and 1 for the current tag'
    output_row = list(empty_output)
    output_row[classes.index(document[1])] = 1
    training_data.append([bag_of_words, output_row])


random.shuffle(training_data)
print("Training Data: \n", training_data)

training_numpy = numpy.array(training_data, dtype=object)


train_X = list(training_numpy[:, 0])
train_y = list(training_numpy[:, 1])

neural_network = tflearn.input_data(shape=[None, len(train_X[0])])

neural_network = tflearn.fully_connected(neural_network, 8)

print(neural_network)

neural_network = tflearn.fully_connected(neural_network, 8)

print(neural_network)

neural_network = tflearn.fully_connected(
    neural_network, len(train_y[0]), activation="softmax")

print(neural_network)

neural_network = tflearn.regression(neural_network)

print(neural_network)

model = tflearn.DNN(neural_network)

print(model)

loaded = model.load("chatbot_dnn.tflearn")

# Don't think I should be looking for index - duct-tape
if not os.path.exists("chatbot_dnn.tflearn.index"):
    model.fit(train_X, train_y, n_epoch=2000, batch_size=8, show_metric=True)
    model.save("chatbot_dnn.tflearn")


question = "Do you sell any coding course?"


def process_question(question):
    question_tokenized = nltk.word_tokenize(question)

    question_stemmed = [stemmer.stem(word.lower())
                        for word in question_tokenized]

    bag = [0] * len(words)

    for stem in question_stemmed:
        for index, word in enumerate(words):
            if word == stem:
                bag[index] = 1
    return (numpy.array(bag))


def categorize(prediction):
    prediction_top = [
        [index, result] for index, result in enumerate(prediction) if result > 0.5
    ]

    prediction_top.sort(key=lambda x: x[1], reverse=True)

    result = []

    for prediction_value in prediction_top:
        result.append((classes[prediction_value[0]], prediction_value[1]))

    return result


def chatbot(question):
    prediction = model.predict([process_question(question)])
    # print(prediction)
    result = categorize(prediction[0])

    return result


user_input = input("Do you have a question for me?")


def respond_to_input(user_input):
    question_category = chatbot(user_input)

    if question_category:
        while question_category:
            for intent in data["intents"]:
                if intent["tag"] == question_category[0][0]:
                    return random.choice(intent["responses"])


for i in range(4):
    user_input = input("Do you have a question for me?\n")
    response = respond_to_input(user_input)
    print(response)
