import numpy as np
from collections import Counter


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
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    sorted_indices = np.argsort(feature_vector)
    sorted_feature = feature_vector[sorted_indices]
    sorted_target = target_vector[sorted_indices]

    n = len(sorted_feature)

    unique_values = np.unique(sorted_feature)

    if len(unique_values) == 1:
        return np.array([]), np.array([]), None, None

    thresholds = (unique_values[1:] + unique_values[:-1]) / 2.0

    left_counts = np.searchsorted(sorted_feature, thresholds, side='left')

    valid_mask = (left_counts > 0) & (left_counts < n)
    left_counts = left_counts[valid_mask]
    thresholds = thresholds[valid_mask]

    if len(thresholds) == 0:
        return np.array([]), np.array([]), None, None

    cumsum = np.cumsum(sorted_target)
    total_sum = cumsum[-1]

    left_sums = cumsum[left_counts - 1]

    left_sizes = left_counts
    right_sizes = n - left_counts

    right_sums = total_sum - left_sums

    p1_left = left_sums / left_sizes
    p1_right = right_sums / right_sizes

    H_left = 1 - p1_left ** 2 - (1 - p1_left) ** 2
    H_right = 1 - p1_right ** 2 - (1 - p1_right) ** 2

    ginis = - (left_sizes / n) * H_left - (right_sizes / n) * H_right

    best_index = np.argmax(ginis)
    gini_best = ginis[best_index]
    threshold_best = thresholds[best_index]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
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
                (self._min_samples_leaf is not None and len(sub_y) < 2 * self._min_samples_leaf):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks and clicks[key] != 0:
                        current_click = clicks[key]
                        ratio[key] = current_click / current_count
                    else:
                        ratio[key] = 0

                sorted_categories = [x[0] for x in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = {cat: idx for idx, cat in enumerate(sorted_categories)}

                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError

            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if threshold is None or gini is None:
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [cat for cat in categories_map.keys()
                                      if categories_map[cat] < threshold]
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)],
                       node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        """
        Рекурсивное предсказание для одного объекта

        :param x: вектор признаков объекта
        :param node: текущий узел дерева
        :return: предсказанный класс
        """
        if node["type"] == "terminal":
            return node["class"]
        else:
            feature_idx = node["feature_split"]
            feature_type = self._feature_types[feature_idx]

            if feature_type == "real":
                if x[feature_idx] < node["threshold"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])
            elif feature_type == "categorical":
                if x[feature_idx] in node["categories_split"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)