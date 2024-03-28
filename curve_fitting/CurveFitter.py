import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class CurveFitter:
    def __init__(self):
        pass

    def fit_curve(self, model_func, x, y, initial_guess):
        """
        Fit a curve to the given data using nonlinear least squares optimization.

        Args:
            model_func (callable): The model function to fit the data.
            x (array-like): The independent variable values.
            y (array-like): The dependent variable values.
            initial_guess (array-like): Initial guess for the parameters of the model function.

        Returns:
            tuple: Optimal parameters for the model function, covariance matrix.
        """
        # Perform curve fitting
        params, cov_matrix = curve_fit(model_func, x, y, p0=initial_guess)
        return params, cov_matrix

    def calculate_r_squared(self, y_true, y_pred):
        """
        Calculate the coefficient of determination (R^2) for the fitted curve.

        Args:
            y_true (array-like): The true values of the dependent variable.
            y_pred (array-like): The predicted values of the dependent variable.

        Returns:
            float: The coefficient of determination (R^2).
        """
        # Calculate the mean of the observed data
        mean_y = np.mean(y_true)
        # Calculate total sum of squares
        total_sum_squares = np.sum((y_true - mean_y) ** 2)
        # Calculate residual sum of squares
        residual_sum_squares = np.sum((y_true - y_pred) ** 2)
        # Calculate R^2
        r_squared = 1 - (residual_sum_squares / total_sum_squares)
        return r_squared

    def plot_curve_fit(self, x, y, model_func, params, xlabel='', ylabel='', title=''):
        """
        Plot the original data and the fitted curve.

        Args:
            x (array-like): The independent variable values.
            y (array-like): The observed dependent variable values.
            model_func (callable): The model function used for fitting.
            params (array-like): Optimal parameters for the model function.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            title (str): Title of the plot.
        """
        plt.figure()
        plt.scatter(x, y, label='Data')
        plt.plot(x, model_func(x, *params), color='red', label='Fitted Curve')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()

        
    def plot_curve_comparison(self, x, y, model_func, params, xlabel='', ylabel='', title=''):
        """
        Plot a comparison between the original data and the fitted curve.

        Args:
            x (array-like): The independent variable values.
            y (array-like): The observed dependent variable values.
            model_func (callable): The model function used for fitting.
            params (array-like): Optimal parameters for the model function.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            title (str): Title of the plot.
        """
        # Plot unfitted data
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(x, y, label='Data')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title + ' (Unfitted)')

        # Plot fitted curve
        plt.subplot(1, 2, 2)
        plt.scatter(x, y, label='Data')
        plt.plot(x, model_func(x, *params), color='red', label='Fitted Curve')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title + ' (Fitted)')
        plt.legend()
        plt.show()

    def plot_curve_overlap(self, x, y, model_func, params, xlabel='', ylabel='', title=''):
        """
        Plot the overlap of the original data, fitted curve, and their difference.

        Args:
            x (array-like): The independent variable values.
            y (array-like): The observed dependent variable values.
            model_func (callable): The model function used for fitting.
            params (array-like): Optimal parameters for the model function.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            title (str): Title of the plot.
        """
        plt.figure()
        plt.scatter(x, y, label='Data')
        plt.plot(x, model_func(x, *params), color='red', label='Fitted Curve')
        plt.plot(x, y - model_func(x, *params), color='black', linestyle='--', label='Difference')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()