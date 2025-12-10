import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    if len(np.unique(feature_vector)) < 2:
        return np.array([]), np.array([]), None, None

    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_target = target_vector[sorted_indices]

    thresholds = (sorted_features[1:] + sorted_features[:-1]) / 2.0

    distinct_mask = np.diff(sorted_features) != 0
    if not np.any(distinct_mask):
        return np.array([]), np.array([]), None, None

    unique_thresholds = thresholds[distinct_mask]

    left_masks = sorted_features[:, np.newaxis] < unique_thresholds

    n_left = np.sum(left_masks, axis=0)
    n_right = len(sorted_features) - n_left

    valid_mask = (n_left > 0) & (n_right > 0)
    if not np.any(valid_mask):
        return np.array([]), np.array([]), None, None

    n_left = n_left[valid_mask]
    n_right = n_right[valid_mask]
    unique_thresholds = unique_thresholds[valid_mask]
    left_masks = left_masks[:, valid_mask]
    n_total = len(sorted_features)

    sum_left = np.dot(sorted_target, left_masks)
    p1_left = sum_left / n_left
    p0_left = 1 - p1_left

    sum_right = np.sum(sorted_target) - sum_left
    p1_right = sum_right / n_right
    p0_right = 1 - p1_right

    H_left = 1 - p1_left ** 2 - p0_left ** 2
    H_right = 1 - p1_right ** 2 - p0_right ** 2

    valid_h = ~np.isnan(H_left) & ~np.isnan(H_right)
    n_left = n_left[valid_h]
    n_right = n_right[valid_h]
    H_left = H_left[valid_h]
    H_right = H_right[valid_h]
    unique_thresholds = unique_thresholds[valid_h]

    if len(unique_thresholds) == 0:
        return np.array([]), np.array([]), None, None

    ginis = - (n_left / n_total) * H_left - (n_right / n_total) * H_right

    best_idx = np.argmax(ginis)
    threshold_best = unique_thresholds[best_idx]
    gini_best = ginis[best_idx]

    return unique_thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if (self._max_depth is not None and depth >= self._max_depth) or \
                (self._min_samples_split is not None and len(sub_y) < self._min_samples_split) or \
                (len(np.unique(sub_y)) == 1):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            feature_vector = sub_X[:, feature]

            if len(np.unique(feature_vector)) < 2:
                continue

            if feature_type == "real":
                thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)

                if gini is None or np.isnan(gini):
                    continue

                current_split = feature_vector < threshold

                if self._min_samples_leaf is not None and \
                        (np.sum(current_split) < self._min_samples_leaf or np.sum(
                            ~current_split) < self._min_samples_leaf):
                    continue

            elif feature_type == "categorical":
                feature_vector = sub_X[:, feature].copy()

                unique_categories = np.unique(feature_vector)
                category_probs = {}
                for cat in unique_categories:
                    mask = (feature_vector == cat)
                    if np.sum(mask) > 0:
                        prob = np.mean(sub_y[mask])
                        category_probs[cat] = prob

                sorted_categories = sorted(category_probs.keys(), key=lambda x: category_probs[x])
                category_map = {cat: idx for idx, cat in enumerate(sorted_categories)}

                numeric_vector = np.array([category_map[x] for x in feature_vector])

                thresholds, ginis, threshold, gini = find_best_split(numeric_vector, sub_y)

                if gini is None or np.isnan(gini):
                    continue

                numeric_values = np.array([category_map[x] for x in feature_vector])
                current_split = numeric_values < threshold

                if self._min_samples_leaf is not None and \
                        (np.sum(current_split) < self._min_samples_leaf or np.sum(
                            ~current_split) < self._min_samples_leaf):
                    continue

                self._left_categories = [cat for cat in sorted_categories if category_map[cat] < threshold]
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                threshold_best = threshold
                split = current_split

                if feature_type == "categorical":
                    node["left_categories"] = self._left_categories.copy()

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["feature_type"] = self._feature_types[feature_best]

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["left_categories"] = self._left_categories

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_type = node["feature_type"]

        if feature_type == "real":
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature_idx] in node["left_categories"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)


# Обертка для нашего дерева, чтобы оно работало с cross_val_score
class DecisionTreeWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types):
        self.feature_types = feature_types

    def fit(self, X, y):
        self.tree = DecisionTree(feature_types=self.feature_types)
        self.tree.fit(X, y)
        return self

    def predict(self, X):
        return self.tree.predict(X)