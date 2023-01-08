import random

import numpy as np

from baes_generation import InputStyle


# рассчет вероятности для каждого атрибута по отдельности
def calculation_probabilities(config):
    probability = []
    for value in config.styles_count.values():
        p = [(x + 1) / (config.N + value.count(0)) if x == 0 else x / config.N for x in value]
        probability.append(p)
    return probability


# генератор стиля
def probability_generator(probability, config):
    count = 0
    out_generetor_dict = {}
    for key, value in config.styles.items():
        data = random.choices(value, weights=probability[count], k=1)
        count += 1
        out_generetor_dict[f"{key}"] = data[0]
    return out_generetor_dict


def naive_bayes(probability, config):
    res = []
    # type_style 0 - прически, 1 - цвет волос, 2 - аксесуар, 3 - одежда, 4 - цвет одежды
    type_style = 0
    mul = probability[0][type_style]
    # черный, нет очков, худи, черный
    count = 0
    for x1 in probability:
        res.append((x1[count]))
    result = np.prod(np.array(res))
    print(result)


if __name__ == "__main__":
    config = InputStyle()

    probability = calculation_probabilities(config)
    result = probability_generator(probability, config)
    print(f"MLE \n {result}")

    # probability_nb = naive_bayes(probability, config)
