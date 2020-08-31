import numpy as np
import matplotlib.pyplot as plt


def function(x):
    return (x**2) + (2*x) + 2


x = np.linspace(-3, 3, 500)

plt.plot(x, function(x))
plt.show()


def deriv(x):
    return (2*x) + 2


def gradient_descent(x_new, x_old, precision, l_r):
    x_list, y_list = [x_new], [function(x_new)]
    while abs(x_new - x_old) > precision:
        x_prev = x_new
        d_x = -deriv(x_prev)
        x_new = x_prev + (l_r * d_x)
        x_list.append(x_new)
        y_list.append(function(x_new))
    print('Local Minimum Occurs at: ', x_new)
    plt.subplot(1, 2, 2)
    plt.scatter(x_list, y_list, c="g")
    plt.plot(x_list, y_list, c="g")
    plt.plot(x, function(x), c="r")
    plt.title("Gradient descent")
    plt.show()

    plt.subplot(1, 2, 1)
    plt.scatter(x_list, y_list, c="g")
    plt.plot(x_list, y_list, c="g")
    plt.plot(x, function(x), c="r")
    plt.xlim([1.0, 2.1])
    plt.title("Zoomed in Gradient descent to Key Area")
    plt.show()


print(gradient_descent(-2, 2, 0.001, 1))
