# Load Training Data
with open("resources/imdb_labelled.txt", "r") as text_file:
    lines = text_file.read().split('\n')
with open("resources/yelp_labelled.txt", "r") as text_file:
    lines += text_file.read().split('\n')
with open("resources/amazon_cells_labelled.txt", "r") as text_file:
    lines += text_file.read().split('\n')

print(lines)
print("Loaded " + str(len(lines)) + " IMDB/Yelp/Amazon labeled comments ")

# Separate Problem Instance(comment) from its label(0(Negative)/1(Positive)) and filter out incorrect data
splitLines = [line.split("\t") for line in lines if len(line.split("\t"))==2 and line.split("\t")[1]!='']

print(splitLines)

train_documents = [line[0] for line in splitLines]
train_labels = [line[1] for line in splitLines]

# Use CountVectorizer to represent comments numerically using Term Frequency Representation
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(binary='true')
train_documents_term_frequencies = count_vectorizer.fit_transform(train_documents)

print(train_documents_term_frequencies[0])

# Create a BernoulliNB object that allows us to use the Naive Bayes Algorithm
from sklearn.naive_bayes import BernoulliNB

# Train Model
classifier = BernoulliNB().fit(train_documents_term_frequencies, train_labels)


# Test Model
def test_model(classifier_param, problem_instance):
    return classifier_param.predict(count_vectorizer.transform([problem_instance]))


def print_prediction_result(classifier_param, comment):
    prediction = test_model(classifier_param, comment)
    print(comment + " was rated " + ("Positive" if prediction == '1' else "Negative"))


print_prediction_result(classifier, "Best movie ever!")
print_prediction_result(classifier, "Worst movie ever!")
print_prediction_result(classifier, "What a crappy movie!")
print_prediction_result(classifier, "Such a movie!")  # Wrong classification
print_prediction_result(classifier, "I've seen better")
print_prediction_result(classifier, "Impeccable performance of Johnny Bravo!")
print_prediction_result(classifier, "Just wow!")


