from io import SEEK_CUR
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn import tree
from matplotlib import pyplot as plt


def load_dataset():
    with open('clean_real.txt') as f:
        real = f.readlines()
    with open('clean_fake.txt') as f2:
        fake = f2.readlines()
    
    headlines = real + fake
    labels = [1 for _ in range(len(real))] + [0 for _ in range(len(fake))]
    
    global v 
    v = CountVectorizer()
    X = v.fit_transform(headlines)
    
    x_train, x_test, y_train, y_test = train_test_split(X, labels)
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test)
    
    return x_train, y_train, x_validation, y_validation, x_test, y_test

def select_model():
    x_train, y_train, x_validation, y_validation, x_test, y_test = load_dataset()
    criterions = ["gini", "entropy"]
    top_criterion = [0, 0, 0]
    for c in criterions:
        for i in range(3, 8):
            clf = DecisionTreeClassifier(criterion=c, max_depth=i)
            clf.fit(x_train, y_train)
            score = clf.score(x_validation, y_validation)
            if score > top_criterion[2]:
                top_criterion = [c, i, score]
            #print(f"Criterion: {c}, Max depth: {i}, Accuracy: {score}")
    return top_criterion

def load_clf():
    parameters = select_model()
    x_train, y_train, x_validation, y_validation, x_test, y_test = load_dataset()
    clf = DecisionTreeClassifier(criterion=parameters[0], max_depth=parameters[1])
    clf.fit(x_train, y_train)
    return clf 
    
def visualization(clf):
    text_representation = export_text(clf, feature_names = v.get_feature_names(), max_depth = 2)
    print(text_representation)
        
def compute_information_gain(class_, i):
    clf = load_clf()
    # IG(node) = H(node) - H(node|child) 
 
    x_train, y_train, x_validation, y_validation, x_test, y_test = load_dataset()
    
    features = v.get_feature_names()
    root = features[clf.tree_.feature[i]]
    left = features[clf.tree_.feature[clf.tree_.children_left[i]]]
    right = features[clf.tree_.feature[clf.tree_.children_left[i]]]

    input_ = x_train[:i+1]
    prob = clf.predict_proba(input_)
    log_prob = clf.predict_log_proba(input_)
    
    ig = 1
    for i in range(len(prob)):
        ig -= prob[i][0]*log_prob[i][0]
        ig -= prob[i][1]*log_prob[i][1]    
    
    print(f"Information gain for {root}: {ig}")    
    
    