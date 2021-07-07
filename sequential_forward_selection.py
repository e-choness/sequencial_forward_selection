from sklearn.datasets import load_breast_cancer
import numpy as np


# This class is created to find the first feature, selected by the maximum value of information gain
# For the use of organizing the indices about the selected subset and discarded features
class Subset:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.target_entropy = self.entropy(target)
        self.n_instances = features.shape[0]
        self.n_features = features.shape[1]
        self.selected_idx = [self.find_first_feature()]
        self.remain_idx = np.delete(np.arange(self.n_features), self.selected_idx).tolist()

    # Once a feature is found that can improve target ML method's accuracy, use this function to update the best set
    # of features
    def add_feature(self, important_idx):
        # print("important", important_idx)
        self.selected_idx.append(important_idx)
        # self.selected_idx = sorted(self.selected_idx)
        self.remain_idx.remove(important_idx)

    # Calculate entropy
    def entropy(self, y):
        hist = np.bincount(y)
        possibilities = hist / len(y)
        # print(possibilities)
        return -np.sum([p * np.log2(p) for p in possibilities if p > 0])

    # calculate information gain by each feature
    def feature_information_gain(self, feature_idx):
        feature = self.features[:, feature_idx]
        f_unique = np.unique(feature)
        individual_entropy = []
        individual_p = []
        for f in f_unique:
            unique_idx = np.argwhere(feature == f).flatten()
            individual_entropy.append(self.entropy(self.target[unique_idx]))
            individual_p.append(len(feature[unique_idx]) / self.n_instances)
            # print(unique_idx)
        feature_entropy = np.dot(individual_p, individual_entropy)
        # print(feature_entropy)
        information_gain = self.target_entropy - feature_entropy
        return information_gain

    def find_first_feature(self):
        first_idx = 0
        best_ig = 0
        # Find the feature index that yield the maximum information gain
        for i in range(self.n_features):
            current_ig = self.feature_information_gain(i)
            # print(current_ig)
            if current_ig > best_ig:
                best_ig = current_ig
                first_idx = i

        return first_idx


# Gaussian Distribution Naive Bayes as the supervised learning method for evaluating selected feature
class GaussianNaiveBayes:
    def __init__(self):
        # add a small constant in case that result of pdf is zero
        self.c = 1e-4

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)
            # print(X_c.mean(axis=0))
            # break

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        # print(y_pred)
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x) + self.c))
            posterior = prior + posterior
            # print(posterior)
            posteriors.append(posterior)

        # return class with its highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


def get_accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# A function require for stratified_cross_validation, which is to find k fold for each class
def separate_by_class(feature, target):
    unique_values = np.unique(target)
    separated_feature = []
    separated_target = []
    for uv in unique_values:
        idx = np.argwhere(target == uv)
        separated_feature.append(feature[idx])
        separated_target.append(target[idx])

    return separated_feature, separated_target


# As the function name suggests, separate training and testing set by fold
def separate_train_test(feature, target, k, n):
    n_instances = len(target)
    # n_feature = feature.shape[1]
    # print(feature.shape)
    # find the indices of the test set, the others are regarded as training set
    n_range = n_instances // k
    if k == (n + 1):
        # idx = [i for i in range(n * n_range, n_instances - 1)]
        idx = np.arange((n * n_range), (n_instances - 1))
    else:
        # idx = [i for i in range(n * n_range, (n + 1) * n_range)]
        idx = np.arange(n * n_range, ((n + 1) * n_range))

    if feature.ndim >= 2:
        x_test = feature[idx, :]
        x_train = np.delete(feature, idx, axis=0)
    else:
        # Reshape 1 dimensional array so that it can fit into ML model
        x_test = feature[idx].reshape(-1, 1)
        x_train = np.delete(feature, idx).reshape(-1, 1)

    y_test = target[idx]
    y_train = np.delete(target, idx)

    return x_train, y_train, x_test, y_test


def stratified_cross_validation(features, targets, k):
    s_f, s_t = separate_by_class(features, targets)
    accuracies = []

    for n in range(k):
        x_train = None
        y_train = None
        x_test = None
        y_test = None
        for f, t in zip(s_f, s_t):
            xt, yt, xtt, ytt = separate_train_test(f.squeeze(), t.squeeze(), k, n)
            if x_train is None:
                x_train = xt
                y_train = yt
                x_test = xtt
                y_test = ytt
            else:
                x_train = np.concatenate((x_train, xt), axis=0)
                y_train = np.concatenate((y_train, yt))
                x_test = np.concatenate((x_test, xtt), axis=0)
                y_test = np.concatenate((y_test, ytt))

            # print(xt.shape, yt.shape)
            # print(xtt.shape, ytt.shape)

        # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        # initialize Naive Bayes model for evaluation
        gaussian_nb = GaussianNaiveBayes()
        # train the Naive Bayes model and calculate accuracy
        gaussian_nb.fit(x_train, y_train)
        prediction = gaussian_nb.predict(x_test)
        accuracy = get_accuracy(y_test, prediction)
        accuracies.append(accuracy)

    # print(accuracies)
    # take average accuracy from the result of 5 folds
    return np.mean(accuracies)


def select_best_index(subset):
    best_accuracy = stratified_cross_validation(subset.features[:, subset.selected_idx], subset.target, 5)
    best_idx = -1
    for rid in subset.remain_idx:
        # print(rid)
        # copy() is needed, otherwise the list will grow with the loop
        new_id = subset.selected_idx.copy()
        new_id.append(rid)
        # print(new_id)
        new_features = subset.features[:, new_id]
        # print(new_features.shape)
        current_accuracy = stratified_cross_validation(new_features, subset.target, 5)
        # print(current_accuracy)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_idx = rid

    return best_idx


def sequential_forward_selection(subset, threshold):
    n_limit = threshold * subset.n_features

    for _ in subset.remain_idx:
        # when 0.75 of the entire feature is selected, terminate the selection process
        if len(subset.selected_idx) > n_limit:
            break
        else:
            # when no improvement by adding feature, terminate the loop
            best_idx = select_best_index(subset)
            if best_idx == -1:
                break
            else:
                subset.add_feature(best_idx)

    entire_data_accuracy = stratified_cross_validation(subset.features, subset.target, 5)
    selected_data_accuracy = stratified_cross_validation(subset.features[:, subset.selected_idx], subset.target, 5)
    print("Entire data accuracy:", entire_data_accuracy)
    print("Selected data accuracy:", selected_data_accuracy)
    print("Best Set of Features (index start from 0):", sorted(subset.selected_idx))


# Load breast cancer dataset
data_bc = load_breast_cancer()
x = data_bc.data
y = data_bc.target
feature_names = data_bc.feature_names
print("The dataset has {0} features, {1} instances with {2} classes.".format(x.shape[1], x.shape[0], len(np.unique(y))))

subset = Subset(x, y)
threshold = 0.75
# get result
sequential_forward_selection(subset, threshold)

print("The best feature names:", feature_names[subset.selected_idx])
