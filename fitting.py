#!/usr/bin/python3

from scipy.optimize import curve_fit
from matplotlib import pyplot
import numpy as np
import pandas as pd
import sys


PLOTFONTSIZE = 12
CURVEPOINTS = 1000


# defining a function for approximation
def func(x, N_0, x_0, a):
    return N_0 / (np.exp((x - x_0) / a) + 1)


def func_derivative(x, N_0, x_0, a):
    ex = np.exp((x - x_0) / a)
    return -(N_0 / a) * (ex / (ex + 1) ** 2)


def linear(x, k, b):
    return (k * x) + b


# wrapper for curve_fit
def curve_fit_wr(func, x, y, lborder=0, rborder=0):

    # curve fit
    popt, pcov = curve_fit(func, x, y)
    perr = np.sqrt(np.diag(pcov))

    # showing fit coefficients and errors
    for element, error in zip(popt, perr):
        print(f"{element} +- {error}")

    # creating points of approx curve
    if lborder != rborder:
        x_line = np.linspace(lborder, rborder, CURVEPOINTS)
    else:
        x_line = np.linspace(min(x), max(x), CURVEPOINTS)

    y_line = func(x_line, *popt)

    return x_line, y_line, popt


def main():

    # creating Pandas data frame from csv file defined by command line argument
    DataFrame = pd.read_csv(sys.argv[1])

    # Verification printout of the first lines of PcapFataFrame
    print(DataFrame.head())

    # creating numpy data arrays from data frame
    x = DataFrame['x'].to_numpy()
    y = DataFrame['y'].to_numpy()
    x_err = DataFrame['Err x'].to_numpy()
    y_err = DataFrame['Err y'].to_numpy()

    # getting approx curve
    print("Approximation of defined function: ")
    x_line, y_line, popt = curve_fit_wr(func, x, y)

    # getting derivative curve
    y_deriv_line = -2 * func_derivative(x_line, *popt)

    # --plot section--
    # plotting experimental points
    pyplot.scatter(x, y, label="Экспериментальные точки", marker='x')

    # adding errorbars for experimental points
    pyplot.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='none', 
                    elinewidth=0.5, capsize=4)

    # plotting approx curve
    pyplot.plot(x_line, y_line, color='purple',
                label="Нелинейное приближение N(x)")

    # plotting derivative of approx curve
    pyplot.plot(x_line, y_deriv_line, color='red',
                label=r"-2 * dN/dx")

    # plotting the line to the maximum of the derivative
    pyplot.vlines(x_line[y_deriv_line.argmax(axis=0)],
                  ymin=0, ymax=y_deriv_line.max(),
                  linestyles='dashed', color='red')

    # creating linear approximation of curve's section
    # all magic constants define particular curve's section
    print("Approximation of linear polynomial")
    x_line, y_line, popt = curve_fit_wr(linear, x[8:13], y[8:13],
                                        lborder=16.5, rborder=19.25)
    pyplot.plot(x_line, y_line, color='blue',
                linestyle='--', label="Линейное приближение отрезка N(x)")

    # creating plot grid
    pyplot.grid()

    # creating legend and naming axis
    pyplot.xlabel("x, мм")
    pyplot.ylabel("N'")
    pyplot.legend(framealpha=1)

    # show plot
    pyplot.show()



if __name__ == '__main__':
    main()
