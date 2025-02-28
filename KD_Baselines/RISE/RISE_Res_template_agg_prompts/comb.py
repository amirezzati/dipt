import itertools

numbers = range(5)  # 0, 1, 2, 3, 4
combinations = list(itertools.combinations(numbers, 3))

for comb in combinations:
    print(" ".join(map(str, comb)))