import random
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def metrics(predicted_values, Y_test, nvar):
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((predicted_values - Y_test) ** 2)
    print("Mean Squared Error (MSE):", mse)

    # Calculate R-squared Error
    ssr = np.sum((predicted_values - Y_test) ** 2)
    sst = np.sum((Y_test - np.mean(Y_test)) ** 2)
    r_sqErr = 0.0
    if sst == 0:
        print("The value of sst is 0. which means the independent variable/s is not good for prediction!")
    else:
        r_sqErr = 1 - (ssr / sst)

    print("R-squared Error:", r_sqErr)

    # Calculate Adjusted R-squared Errors
    m = len(Y_test)
    adj_r_sqErr = 1 - (1 - r_sqErr) * (m - 1) / (m - nvar - 1)
    print("Adjusted R-squared Error:", adj_r_sqErr)
    return mse, r_sqErr, adj_r_sqErr


class LR_1var:
    def __init__(self, data, X, Y, split_ratio=0.7, learning_rate=0.01, lambda_tp=0, max_iterations=1000,
                 convergence_threshold=0.000001):
        self.data = data
        self.X = X
        self.Y = Y
        self.split_ratio = split_ratio
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.theta0 = 0
        self.theta1 = 0
        self.cost_history = []
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.lambda_tp = lambda_tp

    def v_cost_function(self, X, Y):
        m = len(X)
        summation1 = self.theta0 + self.theta1 * X
        regularization = (1 / (2 * m)) * self.lambda_tp * (self.theta1 ** 2)
        return ((1 / (2 * m)) * np.sum((summation1 - Y) ** 2)) + regularization

    def v_theta0_slope(self, X, Y):
        m = len(X)
        summation = self.theta0 + self.theta1 * X - Y
        regularization = 0  # No regularization for theta0
        return ((1 / m) * np.sum(summation)) + regularization

    def v_theta1_slope(self, X, Y):
        m = len(X)
        summation = X * (self.theta0 + self.theta1 * X - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta1
        return ((1 / m) * np.sum(summation)) + regularization

    def split_data(self, X: np.ndarray, Y: np.ndarray):
        split_index = int(len(X) * self.split_ratio)
        self.X_train = X[:split_index]
        self.Y_train = Y[:split_index]
        self.X_test = X[split_index:]
        self.Y_test = Y[split_index:]
        if len(self.X_train) == 0 or len(self.Y_train) == 0:
            raise ValueError("Training set is empty. Modify the split ratio.")

    def train(self):
        self.split_data(self.X, self.Y)

        for i in range(self.max_iterations):
            cost = self.v_cost_function(self.X_train, self.Y_train)
            self.cost_history.append(cost)
            temp0 = self.theta0 - self.learning_rate * self.v_theta0_slope(self.X_train, self.Y_train)
            temp1 = self.theta1 - self.learning_rate * self.v_theta1_slope(self.X_train, self.Y_train)

            if i > 0 and abs(self.cost_history[i] - self.cost_history[i - 1]) < self.convergence_threshold:
                print(f"Converged after {i + 1} iterations.")
                break

            self.theta0 = temp0
            self.theta1 = temp1

    def predict(self):
        predicted_values = [self.theta0 + self.theta1 * i for i in self.X_test]
        mse, r2_err, adj_r2_err = metrics(predicted_values, self.Y_test, 1)
        return mse, r2_err, adj_r2_err

    def plot_cost_history(self):
        plt.figure(figsize=(5, 5))
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost History')
        plt.show()


class LR_2var:
    def __init__(self, data, X1, X2, Y, split_ratio=0.7, learning_rate=0.01, lambda_tp=0, max_iterations=1000,
                 convergence_threshold=0.000001):
        self.data = data
        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.split_ratio = split_ratio
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.theta0 = 0
        self.theta1 = 0
        self.theta2 = 0
        self.cost_history = []
        self.X1_train = None
        self.X2_train = None
        self.Y_train = None
        self.X1_test = None
        self.X2_test = None
        self.Y_test = None
        self.lambda_tp = lambda_tp

    def v_cost_function(self, X1, X2, Y):
        m = len(Y)
        summation = self.theta0 + self.theta1 * X1 + self.theta2 * X2
        regularization = (1 / (2 * m)) * self.lambda_tp * (self.theta1 ** 2 + self.theta2 ** 2)
        return ((1 / (2 * m)) * np.sum((summation - Y) ** 2)) + regularization

    def v_theta0_slope(self, X1, X2, Y):
        m = len(Y)
        summation = self.theta0 + self.theta1 * X1 + self.theta2 * X2 - Y
        regularization = 0  # No regularization for theta0
        return (1 / m) * np.sum(summation)

    def v_theta1_slope(self, X1, X2, Y):
        m = len(Y)
        summation = X1 * (self.theta0 + self.theta1 * X1 + self.theta2 * X2 - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta1
        return (1 / m) * np.sum(summation) + regularization

    def v_theta2_slope(self, X1, X2, Y):
        m = len(Y)
        summation = X2 * (self.theta0 + self.theta1 * X1 + self.theta2 * X2 - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta2
        return (1 / m) * np.sum(summation) + regularization

    def split_data(self, X1: np.ndarray, X2: np.ndarray, Y: np.ndarray):
        split_index = int(len(X1) * self.split_ratio)
        self.X1_train = X1[:split_index]
        self.X2_train = X2[:split_index]
        self.Y_train = Y[:split_index]
        self.X1_test = X1[split_index:]
        self.X2_test = X2[split_index:]
        self.Y_test = Y[split_index:]
        if len(self.X1_train) == 0 or len(self.X2_train) == 0 or len(self.Y_train) == 0:
            raise ValueError("Training set is empty. Modify the split ratio.")

    def train(self):
        self.split_data(self.X1, self.X2, self.Y)

        for i in range(self.max_iterations):
            cost = self.v_cost_function(self.X1_train, self.X2_train, self.Y_train)
            self.cost_history.append(cost)
            temp0 = self.theta0 - self.learning_rate * self.v_theta0_slope(self.X1_train, self.X2_train, self.Y_train)
            temp1 = self.theta1 - self.learning_rate * self.v_theta1_slope(self.X1_train, self.X2_train, self.Y_train)
            temp2 = self.theta2 - self.learning_rate * self.v_theta2_slope(self.X1_train, self.X2_train, self.Y_train)

            if i > 0 and abs(self.cost_history[i] - self.cost_history[i - 1]) < self.convergence_threshold:
                print(f"Converged after {i + 1} iterations.")
                break

            self.theta0 = temp0
            self.theta1 = temp1

    def predict(self):
        predicted_values = [self.theta0 + self.theta1 * i1 + self.theta2 * i2 for i1, i2 in
                            zip(self.X1_test, self.X2_test)]
        mse, r2_err, adj_r2_err = metrics(predicted_values, self.Y_test, 2)
        return mse, r2_err, adj_r2_err

    def plot_cost_history(self):
        plt.figure(figsize=(5, 5))
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost History')
        plt.show()


class LR_3var:
    def __init__(self, data, X1, X2, X3, Y, split_ratio=0.7, learning_rate=0.01, lambda_tp=0, max_iterations=1000,
                 convergence_threshold=0.000001):
        self.data = data
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.Y = Y
        self.split_ratio = split_ratio
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.theta0 = 0
        self.theta1 = 0
        self.theta2 = 0
        self.theta3 = 0
        self.cost_history = []
        self.X1_train = None
        self.X2_train = None
        self.X3_train = None
        self.Y_train = None
        self.X1_test = None
        self.X2_test = None
        self.X3_test = None
        self.Y_test = None
        self.lambda_tp = lambda_tp


    def v_cost_function(self, X1, X2, X3, Y):
        m = len(Y)
        summation = self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3
        regularization = (1 / (2 * m)) * self.lambda_tp * (self.theta1 ** 2 + self.theta2 ** 2 + self.theta3 ** 2)
        return ((1 / (2 * m)) * np.sum((summation - Y) ** 2)) + regularization

    def v_theta0_slope(self, X1, X2, X3, Y):
        m = len(Y)
        summation = self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 - Y
        regularization = 0  # No regularization for theta0
        return (1 / m) * np.sum(summation)

    def v_theta1_slope(self, X1, X2, X3, Y):
        m = len(Y)
        summation = X1 * (self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta1
        return (1 / m) * np.sum(summation) + regularization

    def v_theta2_slope(self, X1, X2, X3, Y):
        m = len(Y)
        summation = X2 * (self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta2
        return (1 / m) * np.sum(summation) + regularization

    def v_theta3_slope(self, X1, X2, X3, Y):
        m = len(Y)
        summation = X3 * (self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta3
        return (1 / m) * np.sum(summation) + regularization

    def split_data(self, X1: np.ndarray, X2: np.ndarray, X3: np.ndarray, Y: np.ndarray):
        split_index = int(len(X1) * self.split_ratio)
        self.X1_train = X1[:split_index]
        self.X2_train = X2[:split_index]
        self.X3_train = X3[:split_index]
        self.Y_train = Y[:split_index]
        self.X1_test = X1[split_index:]
        self.X2_test = X2[split_index:]
        self.X3_test = X3[split_index:]
        self.Y_test = Y[split_index:]
        if len(self.X1_train) == 0 or len(self.Y_train) == 0:
            raise ValueError("Training set is empty. Modify the split ratio.")

    def train(self):
        self.split_data(self.X1, self.X2, self.X3, self.Y)

        for i in range(self.max_iterations):
            cost = self.v_cost_function(self.X1_train, self.X2_train, self.X3_train, self.Y_train)
            self.cost_history.append(cost)
            temp0 = self.theta0 - self.learning_rate * self.v_theta0_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.Y_train)
            temp1 = self.theta1 - self.learning_rate * self.v_theta1_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.Y_train)
            temp2 = self.theta2 - self.learning_rate * self.v_theta2_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.Y_train)
            temp3 = self.theta3 - self.learning_rate * self.v_theta3_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.Y_train)

            if i > 0 and abs(self.cost_history[i] - self.cost_history[i - 1]) < self.convergence_threshold:
                print(f"Converged after {i + 1} iterations.")
                break

            self.theta0 = temp0
            self.theta1 = temp1
            self.theta2 = temp2
            self.theta3 = temp3

    def predict(self):
        predicted_values = [self.theta0 + self.theta1 * i1 + self.theta2 * i2 + self.theta3 * i3 for i1, i2, i3 in
                            zip(self.X1_test, self.X2_test, self.X3_test)]
        mse, r_sqErr, adj_r_sqErr = metrics(predicted_values, self.Y_test, 3)
        return mse, r_sqErr, adj_r_sqErr

    def plot_cost_history(self):
        plt.figure(figsize=(5, 5))
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost History')
        plt.show()


class LR_4var:
    def __init__(self, data, X1, X2, X3, X4, Y, split_ratio=0.7, learning_rate=0.01, lambda_tp=0,
                 max_iterations=1000, convergence_threshold=0.000001):
        self.data = data
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.X4 = X4
        self.Y = Y
        self.split_ratio = split_ratio
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.theta0 = 0
        self.theta1 = 0
        self.theta2 = 0
        self.theta3 = 0
        self.theta4 = 0
        self.cost_history = []
        self.X1_train = None
        self.X2_train = None
        self.X3_train = None
        self.X4_train = None
        self.Y_train = None
        self.X1_test = None
        self.X2_test = None
        self.X3_test = None
        self.X4_test = None
        self.Y_test = None
        self.lambda_tp = lambda_tp

    def v_cost_function(self, X1, X2, X3, X4, Y):
        m = len(Y)
        summation = self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 + self.theta4 * X4
        regularization = (1 / (2 * m)) * self.lambda_tp * (
                    self.theta1 ** 2 + self.theta2 ** 2 + self.theta3 ** 2 + self.theta4 ** 2)
        return ((1 / (2 * m)) * np.sum((summation - Y) ** 2)) + regularization

    def v_theta0_slope(self, X1, X2, X3, X4, Y):
        m = len(Y)
        summation = self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 + self.theta4 * X4 - Y
        regularization = 0  # No regularization for theta0
        return (1 / m) * np.sum(summation)

    def v_theta1_slope(self, X1, X2, X3, X4, Y):
        m = len(Y)
        summation = X1 * (self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 + self.theta4 * X4 - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta1
        return (1 / m) * np.sum(summation) + regularization

    def v_theta2_slope(self, X1, X2, X3, X4, Y):
        m = len(Y)
        summation = X2 * (self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 + self.theta4 * X4 - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta2
        return (1 / m) * np.sum(summation) + regularization

    def v_theta3_slope(self, X1, X2, X3, X4, Y):
        m = len(Y)
        summation = X3 * (self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 + self.theta4 * X4 - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta3
        return (1 / m) * np.sum(summation) + regularization

    def v_theta4_slope(self, X1, X2, X3, X4, Y):
        m = len(Y)
        summation = X4 * (self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 + self.theta4 * X4 - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta4
        return (1 / m) * np.sum(summation) + regularization

    def split_data(self, X1: np.ndarray, X2: np.ndarray, X3: np.ndarray, X4: np.ndarray, Y: np.ndarray):
        split_index = int(len(X1) * self.split_ratio)
        self.X1_train = X1[:split_index]
        self.X2_train = X2[:split_index]
        self.X3_train = X3[:split_index]
        self.X4_train = X4[:split_index]
        self.Y_train = Y[:split_index]
        self.X1_test = X1[split_index:]
        self.X2_test = X2[split_index:]
        self.X3_test = X3[split_index:]
        self.X4_test = X4[split_index:]
        self.Y_test = Y[split_index:]
        if len(self.X1_train) == 0 or len(self.Y_train) == 0:
            raise ValueError("Training set is empty. Modify the split ratio.")

    def train(self):
        self.split_data(self.X1, self.X2, self.X3, self.X4, self.Y)

        for i in range(self.max_iterations):
            cost = self.v_cost_function(self.X1_train, self.X2_train, self.X3_train, self.X4_train, self.Y_train)
            self.cost_history.append(cost)
            temp0 = self.theta0 - self.learning_rate * self.v_theta0_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.X4_train, self.Y_train)
            temp1 = self.theta1 - self.learning_rate * self.v_theta1_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.X4_train, self.Y_train)
            temp2 = self.theta2 - self.learning_rate * self.v_theta2_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.X4_train, self.Y_train)
            temp3 = self.theta3 - self.learning_rate * self.v_theta3_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.X4_train, self.Y_train)
            temp4 = self.theta4 - self.learning_rate * self.v_theta4_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.X4_train, self.Y_train)

            if i > 0 and abs(self.cost_history[i] - self.cost_history[i - 1]) < self.convergence_threshold:
                print(f"Converged after {i + 1} iterations.")
                break

            self.theta0 = temp0
            self.theta1 = temp1
            self.theta2 = temp2
            self.theta3 = temp3
            self.theta4 = temp4

    def predict(self):
        predicted_values = [self.theta0 + self.theta1 * i1 + self.theta2 * i2 + self.theta3 * i3 + self.theta4 * i4 for
                            i1, i2, i3, i4 in zip(self.X1_test, self.X2_test, self.X3_test, self.X4_test)]
        mse, r_sqErr, adj_r_sqErr = metrics(predicted_values, self.Y_test, 4)
        return mse, r_sqErr, adj_r_sqErr

    def plot_cost_history(self):
        plt.figure(figsize=(5, 5))
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost History')
        plt.show()


class LR_5var:
    def __init__(self, data, X1, X2, X3, X4, X5, Y, split_ratio=0.7, learning_rate=0.01,lambda_tp=0,
                 max_iterations=1000, convergence_threshold=0.000001):
        self.data = data
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.X4 = X4
        self.X5 = X5
        self.Y = Y
        self.split_ratio = split_ratio
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.theta0 = 0
        self.theta1 = 0
        self.theta2 = 0
        self.theta3 = 0
        self.theta4 = 0
        self.theta5 = 0
        self.cost_history = []
        self.X1_train = None
        self.X2_train = None
        self.X3_train = None
        self.X4_train = None
        self.X5_train = None
        self.Y_train = None
        self.X1_test = None
        self.X2_test = None
        self.X3_test = None
        self.X4_test = None
        self.X5_test = None
        self.Y_test = None
        self.lambda_tp = lambda_tp

    def v_cost_function(self, X1, X2, X3, X4, X5, Y):
        m = len(Y)
        summation = self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 + self.theta4 * X4 + self.theta5 * X5
        regularization = (1 / (2 * m)) * self.lambda_tp * (
                    self.theta1 ** 2 + self.theta2 ** 2 + self.theta3 ** 2 + self.theta4 ** 2 + self.theta5 ** 2)
        return ((1 / (2 * m)) * np.sum((summation - Y) ** 2)) + regularization

    def v_theta0_slope(self, X1, X2, X3, X4, X5, Y):
        m = len(Y)
        summation = self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 + self.theta4 * X4 + self.theta5 * X5 - Y
        regularization = 0  # No regularization for theta0
        return (1 / m) * np.sum(summation)

    def v_theta1_slope(self, X1, X2, X3, X4, X5, Y):
        m = len(Y)
        summation = X1 * (
                self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 + self.theta4 * X4 + self.theta5 * X5 - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta1
        return (1 / m) * np.sum(summation) + regularization

    def v_theta2_slope(self, X1, X2, X3, X4, X5, Y):
        m = len(Y)
        summation = X2 * (
                self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 + self.theta4 * X4 + self.theta5 * X5 - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta2
        return (1 / m) * np.sum(summation) + regularization

    def v_theta3_slope(self, X1, X2, X3, X4, X5, Y):
        m = len(Y)
        summation = X3 * (
                self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 + self.theta4 * X4 + self.theta5 * X5 - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta3
        return (1 / m) * np.sum(summation) + regularization

    def v_theta4_slope(self, X1, X2, X3, X4, X5, Y):
        m = len(Y)
        summation = X4 * (
                self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 + self.theta4 * X4 + self.theta5 * X5 - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta4
        return (1 / m) * np.sum(summation) + regularization

    def v_theta5_slope(self, X1, X2, X3, X4, X5, Y):
        m = len(Y)
        summation = X5 * (
                self.theta0 + self.theta1 * X1 + self.theta2 * X2 + self.theta3 * X3 + self.theta4 * X4 + self.theta5 * X5 - Y)
        regularization = (1 / m) * self.lambda_tp * self.theta5
        return (1 / m) * np.sum(summation) + regularization

    def split_data(self, X1: np.ndarray, X2: np.ndarray, X3: np.ndarray, X4: np.ndarray, X5: np.ndarray, Y: np.ndarray):
        split_index = int(len(X1) * self.split_ratio)
        self.X1_train = X1[:split_index]
        self.X2_train = X2[:split_index]
        self.X3_train = X3[:split_index]
        self.X4_train = X4[:split_index]
        self.X5_train = X5[:split_index]
        self.Y_train = Y[:split_index]
        self.X1_test = X1[split_index:]
        self.X2_test = X2[split_index:]
        self.X3_test = X3[split_index:]
        self.X4_test = X4[split_index:]
        self.X5_test = X5[split_index:]
        self.Y_test = Y[split_index:]
        if len(self.X1_train) == 0 or len(self.Y_train) == 0:
            raise ValueError("Training set is empty. Modify the split ratio.")

    def train(self):
        self.split_data(self.X1, self.X2, self.X3, self.X4, self.X5, self.Y)

        for i in range(self.max_iterations):
            cost = self.v_cost_function(self.X1_train, self.X2_train, self.X3_train, self.X4_train, self.X5_train,
                                        self.Y_train)
            self.cost_history.append(cost)
            temp0 = self.theta0 - self.learning_rate * self.v_theta0_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.X4_train, self.X5_train, self.Y_train)
            temp1 = self.theta1 - self.learning_rate * self.v_theta1_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.X4_train, self.X5_train, self.Y_train)
            temp2 = self.theta2 - self.learning_rate * self.v_theta2_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.X4_train, self.X5_train, self.Y_train)
            temp3 = self.theta3 - self.learning_rate * self.v_theta3_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.X4_train, self.X5_train, self.Y_train)
            temp4 = self.theta4 - self.learning_rate * self.v_theta4_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.X4_train, self.X5_train, self.Y_train)
            temp5 = self.theta5 - self.learning_rate * self.v_theta5_slope(self.X1_train, self.X2_train, self.X3_train,
                                                                           self.X4_train, self.X5_train, self.Y_train)

            if i > 0 and abs(self.cost_history[i] - self.cost_history[i - 1]) < self.convergence_threshold:
                print(f"Converged after {i + 1} iterations.")
                break

            self.theta0 = temp0
            self.theta1 = temp1
            self.theta2 = temp2
            self.theta3 = temp3
            self.theta4 = temp4
            self.theta5 = temp5

    def predict(self):
        predicted_values = [
            self.theta0 + self.theta1 * i1 + self.theta2 * i2 + self.theta3 * i3 + self.theta4 * i4 + self.theta5 * i5
            for i1, i2, i3, i4, i5 in zip(self.X1_test, self.X2_test, self.X3_test, self.X4_test, self.X5_test)]
        mse, r_sqErr, adj_r_sqErr = metrics(predicted_values, self.Y_test, 5)
        return mse, r_sqErr, adj_r_sqErr

    def plot_cost_history(self):
        plt.figure(figsize=(5, 5))
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost History')
        plt.show()
