import functools
import random
from itertools import product

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


def generator_combination(probability, config):
    count = 0
    new_dict = {}
    new = []
    # создаем новый словарь {'прическа': ['нет волос-0.14', 'длинные в пучок-0.0196078431372549',
    for key, value in config.styles.items():
        for i in range(len(value)):
            new_parametr = value[i] + "_" + str(probability[count][i])
            new.append(new_parametr)
            new_dict[f"{key}"] = new
        new = []
        count += 1

    # поплучаем комбинации
    generator = (dict(zip(new_dict.keys(), values)) for values in (product(*new_dict.values())))
    return generator


def naive_bayes(generator):
    r = next(generator)
    proba = []
    out = {}
    for key, value in r.items():
        nv = float(value.split("_")[1])
        vk = value.split("_")[0]
        out[f"{key}"] = vk
        proba.append(nv)

    proba = functools.reduce(lambda a, b: a * b, proba)
    return out, proba


if __name__ == "__main__":
    config = InputStyle()

    # 1 способ
    probability = calculation_probabilities(config)
    result = probability_generator(probability, config)
    print(f"полиномиальный метод \n {result}")

    # 2 способ
    generator = generator_combination(probability, config)
    for element in generator:
        result, proba = naive_bayes(generator)
        if proba > 0.005:
            print(f"наивная байесовская модель \n {result}")
            print(proba)
