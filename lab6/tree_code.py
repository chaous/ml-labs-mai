import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Находит оптимальный порог для разбиения вектора признака по критерию Джини.

    Критерий Джини определяется следующим образом:
    .. math::
        Q(R) = -\\frac {|R_l|}{|R|}H(R_l) -\\frac {|R_r|}{|R|}H(R_r),

    где:
    * :math:`R` — множество всех объектов,
    * :math:`R_l` и :math:`R_r` — объекты, попавшие в левое и правое поддерево соответственно.

    Функция энтропии :math:`H(R)`:
    .. math::
        H(R) = 1 - p_1^2 - p_0^2,

    где:
    * :math:`p_1` и :math:`p_0` — доля объектов класса 1 и 0 соответственно.

    Указания:
    - Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    - В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака.
    - Поведение функции в случае константного признака может быть любым.
    - При одинаковых приростах Джини нужно выбирать минимальный сплит.
    - Для оптимизации рекомендуется использовать векторизацию вместо циклов.

    Parameters
    ----------
    feature_vector : np.ndarray
        Вектор вещественнозначных значений признака.
    target_vector : np.ndarray
        Вектор классов объектов (0 или 1), длина `feature_vector` равна длине `target_vector`.

    Returns
    -------
    thresholds : np.ndarray
        Отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на
        два различных поддерева.
    ginis : np.ndarray
        Вектор со значениями критерия Джини для каждого порога в `thresholds`.
    threshold_best : float
        Оптимальный порог для разбиения.
    gini_best : float
        Оптимальное значение критерия Джини.

    """
    # ╰( ͡☉ ͜ʖ ͡☉ )つ──☆*:・ﾟ   ฅ^•ﻌ•^ฅ   ʕ•ᴥ•ʔ

    x = np.asarray(feature_vector)
    y = np.asarray(target_vector)
    n = x.shape[0]

    # Сортируем по признаку
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # Выбираем точки, где значение признака меняется
    diffs = x_sorted[1:] != x_sorted[:-1]
    boundaries = np.nonzero(diffs)[0]
    if boundaries.size == 0:
        # Константный признак — возвращаем пустые результаты
        return np.array([]), np.array([]), None, None

    # Все возможные пороги — середины между соседними уникальными значениями
    thresholds = (x_sorted[boundaries] + x_sorted[boundaries + 1]) / 2.0

    # Кумулятивная сумма меток класса '1'
    cum1 = np.cumsum(y_sorted)

    # Размеры левого и правого подмножества для каждого порога
    n_left  = boundaries + 1
    n_right = n - n_left

    # Число единиц и нулей в каждой части
    ones_left  = cum1[boundaries]
    ones_right = cum1[-1] - ones_left
    zeros_left  = n_left  - ones_left
    zeros_right = n_right - ones_right

    # Доли классов
    p1_left  = ones_left  / n_left
    p0_left  = zeros_left  / n_left
    p1_right = ones_right / n_right
    p0_right = zeros_right / n_right

    # Энтропия Джини для каждого подмножества
    H_left  = 1 - p1_left**2  - p0_left**2
    H_right = 1 - p1_right**2 - p0_right**2

    # Критерий Q = -(|Rl|/|R| * H_left + |Rr|/|R| * H_right)
    ginis = - (n_left / n) * H_left - (n_right / n) * H_right

    # Находим лучший порог: максимальное значение ginis, при равенстве — минимальный порог
    best_idx = np.where(ginis == ginis.max())[0][0]
    threshold_best = thresholds[best_idx]
    gini_best      = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
    ):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        """
        Обучение узла дерева решений.

        Если все элементы в подвыборке принадлежат одному классу, узел становится терминальным.

        Parameters
        ----------
        sub_X : np.ndarray
            Подвыборка признаков.
        sub_y : np.ndarray
            Подвыборка меток классов.
        node : dict
            Узел дерева, который будет заполнен информацией о разбиении.

        """
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        # Остановка по глубине или по минимальному числу для сплита
        if (self._max_depth is not None and depth >= self._max_depth) or \
        (self._min_samples_split is not None and len(sub_y) < self._min_samples_split):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # Поиск лучшего разбиения
        feature_best = None
        threshold_best = None
        gini_best = -np.inf
        split_best = None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                fv = sub_X[:, feature]
            else:  # categorical
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {k: clicks.get(k, 0) / v for k, v in counts.items()}
                sorted_cats = sorted(ratio, key=ratio.get)
                cat_map = {cat: i for i, cat in enumerate(sorted_cats)}
                fv = np.vectorize(cat_map.get)(sub_X[:, feature])

            if len(np.unique(fv)) <= 1:
                continue

            thresholds, ginis, thr, gini = find_best_split(fv, sub_y)
            if gini <= gini_best:
                continue

            # Проверяем маску разбиения
            if feature_type == "real":
                mask = fv < thr
                threshold_value = thr
            else:
                cats_left = [cat for cat, idx in cat_map.items() if idx < thr]
                mask = np.isin(sub_X[:, feature], cats_left)
                threshold_value = cats_left

            # Проверяем min_samples_leaf
            left_count = mask.sum()
            right_count = len(mask) - left_count
            if (self._min_samples_leaf is not None and
                (left_count < self._min_samples_leaf or right_count < self._min_samples_leaf)):
                continue

            feature_best = feature
            threshold_best = threshold_value
            gini_best = gini
            split_best = mask

        # Если лучший сплит не найден — терминальный узел
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # Создаем нетерминальный узел
        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = threshold_best

        node["left_child"] = {}
        node["right_child"] = {}
        # Рекурсивно строим потомков с увеличением глубины
        self._fit_node(sub_X[split_best], sub_y[split_best], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split_best], sub_y[~split_best], node["right_child"], depth + 1)



    def _predict_node(self, x, node):
        """
        Рекурсивное предсказание класса для одного объекта по узлу дерева решений.

        Если узел терминальный, возвращается предсказанный класс.
        Если узел не терминальный, выборка передается в соответствующее поддерево для дальнейшего предсказания.

        Parameters
        ----------
        x : np.ndarray
            Вектор признаков одного объекта.
        node : dict
            Узел дерева решений.

        Returns
        -------
        int
            Предсказанный класс объекта.
        """
        if node["type"] == "terminal":
            return node["class"]

        feat = node["feature_split"]
        if self._feature_types[feat] == "real":
            if x[feat] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[feat] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])


    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        preds = [self._predict_node(x, self._tree) for x in X]
        return np.array(preds, dtype=int)