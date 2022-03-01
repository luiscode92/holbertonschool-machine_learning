#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# your code here

people = ("Farrah", "Fred", "Felicia")
colors = ("red", "yellow", "#ff8000", "#ffe5b4")
kind_fruits = len(fruit)

# Starting level for the bars
bottom = [0, 0, 0]

# Bar plot
# fruit[idx] is the height
for idx in range(kind_fruits):
    plt.bar(
        people,
        fruit[idx],
        color=colors[idx],
        width=0.5,
        bottom=bottom
    )
    bottom += fruit[idx]

plt.ylabel("Quantity of Fruit")
plt.yticks(range(0, 81, 10))
plt.ylim(0, 80)
plt.title("Number of Fruit per Person")
plt.legend(["apples", "bananas", "oranges", "peaches"])

plt.show()
