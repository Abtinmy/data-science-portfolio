import numpy as np


class Regressor:

    def __init__(self) -> None: 
        self.X, self.y = self.generate_dataset(n_samples=200, n_features=1)
        n, d = self.X.shape
        self.w = np.zeros((d, 1))
        self.v = np.zeros((d, 1)) #SGD with momentum

    def generate_dataset(self, n_samples, n_features):
        """
        Generates a regression dataset
        Returns:
            X: a numpy.ndarray of shape (100, 2) containing the dataset
            y: a numpy.ndarray of shape (100, 1) containing the labels
        """
        from sklearn.datasets import make_regression
        
        np.random.seed(42)
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=30)
        y = y.reshape(n_samples, 1)
        return X, y


    def linear_regression(self):
        """
        Performs linear regression on a dataset
        Returns:
            y: a numpy.ndarray of shape (n, 1) containing the predictions
        """
        y = np.dot(self.X, self.w)
        return y


    def predict(self, X):
        """
        Predicts the labels for a given dataset
        X: a numpy.ndarray of shape (n, d) containing the dataset
        Returns:
            y: a numpy.ndarray of shape (n,) containing the predictions
        """
        y = np.dot(X, self.w).reshape(X.shape[0])
        return y


    def compute_loss(self):
        """
        Computes the MSE loss of a prediction
        Returns:
            loss: the loss of the prediction
        """
        predictions = self.linear_regression()
        loss = np.mean((predictions - self.y)**2)
        return loss


    def compute_gradient(self):
        """
        Computes the gradient of the MSE loss
        Returns:
            grad: the gradient of the loss with respect to w
        """
        predictions = self.linear_regression()
        dif = (predictions - self.y)
        grad = 2 * np.dot(self.X.T, dif)
        return grad


    def fit(self, optimizer='gd', n_iters=1000, tol=0.001, render_animation=False, monitor=True, **kwargs):
        """
        Trains the model
        optimizer: the optimization algorithm to use
        X: a numpy.ndarray of shape (n, d) containing the dataset
        y: a numpy.ndarray of shape (n, 1) containing the labels
        n_iters: the number of iterations to train for
        """        

        figs = []
        prev_w = 0
        loss = None
        alpha = kwargs.get('alpha')
        g = np.zeros((self.w.shape[0], 1))
        m = 0
        v = 0

        for i in range(1, n_iters+1):
            iter = i
            prev_w = self.w

            if optimizer == 'gd':
                self.w = self.gradient_descent(alpha)
            elif optimizer == "sgd":
                self.w = self.sgd_optimizer(alpha, kwargs.get('batch_size'))
            elif optimizer == "sgdMomentum":
                self.w = self.sgd_momentum(alpha, kwargs.get('momentum'), kwargs.get('batch_size'))
            elif optimizer == "adagrad":
                self.w, g = self.adagrad_optimizer(g, alpha, epsilon=1e-8,
                                                       batch_size=kwargs.get('batch_size'))
            elif optimizer == "rmsprop":
                self.w, g = self.rmsprop_optimizer(g, alpha, beta=kwargs.get('momentum'),
                                                   epsilon=1e-8, batch_size=kwargs.get('batch_size'))
            elif optimizer == "adam":
                self.w, m, v = self.adam_optimizer(m, v, iter, alpha, beta1=kwargs.get('beta1'),
                                                   beta2=kwargs.get('beta2'), epsilon=1e-8,
                                                   batch_size=kwargs.get('batch_size'))

            #stop criterion
            loss = self.compute_loss()
            if np.sum(abs(prev_w - self.w)) < tol:
                if monitor:
                    print("Iteration: ", i)
                    print("Loss: ", loss)
                break

            if i % 10 == 0 and monitor:
                print("Iteration: ", i)
                print("Loss: ", loss)
            
            if render_animation:
                import matplotlib.pyplot as plt
                from moviepy.video.io.bindings import mplfig_to_npimage

                fig = plt.figure()
                plt.scatter(self.X, self.y, color='red')
                plt.plot(self.X, self.predict(self.X), color='blue')
                plt.xlim(self.X.min(), self.X.max())
                plt.ylim(self.y.min(), self.y.max())
                plt.title(f'Optimizer:{optimizer}\nIteration: {i}')
                plt.close()
                figs.append(mplfig_to_npimage(fig))
            
        
        if render_animation and len(figs) > 0:
            from moviepy.editor import ImageSequenceClip
            clip = ImageSequenceClip(figs, fps=5)
            clip.write_gif(f'{optimizer}_animation.gif', fps=5)
        
        return loss, iter


    def gradient_descent(self, alpha):
        """
        Performs gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        return self.w - (alpha / self.X.shape[0]) * np.dot(self.X.T, (self.linear_regression() - self.y))


    def sgd_optimizer(self, alpha, batch_size=1):
        """
        Performs stochastic gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        indexes = np.random.randint(0, len(self.X), batch_size)
        temp_x = np.take(self.X, indexes).reshape(batch_size, self.X.shape[1])
        temp_y = np.take(self.y, indexes).reshape((batch_size, 1))

        return self.w - (alpha / temp_x.shape[0]) * np.dot(temp_x.T, (np.dot(temp_x, self.w) - temp_y))


    def sgd_momentum(self, alpha=0.01, momentum=0.9, batch_size=1):
        """
        Performs SGD with momentum to optimize the weights
        alpha: the learning rate
        momentum: the momentum
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        indexes = np.random.randint(0, len(self.X), batch_size)
        temp_x = np.take(self.X, indexes).reshape(batch_size, self.X.shape[1])
        temp_y = np.take(self.y, indexes).reshape((batch_size, 1))

        grad = (1 / temp_x.shape[0]) * np.dot(temp_x.T, (np.dot(temp_x, self.w) - temp_y))
        self.v = momentum * self.v  + grad
        return self.w - alpha * self.v

    
    def adagrad_optimizer(self, g, alpha, epsilon, batch_size):
        """
        Performs Adagrad optimization to optimize the weights
        alpha: the learning rate
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        indexes = np.random.randint(0, len(self.X), batch_size)
        temp_x = np.take(self.X, indexes).reshape(batch_size, self.X.shape[1])
        temp_y = np.take(self.y, indexes).reshape((batch_size, 1))

        grad = (1 / temp_x.shape[0]) * np.dot(temp_x.T, (np.dot(temp_x, self.w) - temp_y))
        g = g + grad ** 2
        grad = grad / (np.sqrt(g) + epsilon)
        return self.w - alpha * grad, g

    
    def rmsprop_optimizer(self, g, alpha, beta, epsilon, batch_size):
        """
        Performs RMSProp optimization to optimize the weights
        g: sum of squared gradients
        alpha: the learning rate
        beta: the momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        indexes = np.random.randint(0, len(self.X), batch_size)
        temp_x = np.take(self.X, indexes).reshape(batch_size, self.X.shape[1])
        temp_y = np.take(self.y, indexes).reshape((batch_size, 1))

        grad = (1 / temp_x.shape[0]) * np.dot(temp_x.T, (np.dot(temp_x, self.w) - temp_y))
        g = beta * g + (1 - beta) * (grad ** 2)
        grad = grad / (np.sqrt(g) + epsilon)
        return self.w - alpha * grad, g

    def adam_optimizer(self, m, v, i, alpha, beta1, beta2, epsilon, batch_size):
        """
        Performs Adam optimization to optimize the weights
        m: the first moment vector
        v: the second moment vector
        alpha: the learning rate
        beta1: the first momentum
        beta2: the second momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        indexes = np.random.randint(0, len(self.X), batch_size)
        temp_x = np.take(self.X, indexes).reshape(batch_size, self.X.shape[1])
        temp_y = np.take(self.y, indexes).reshape((batch_size, 1))

        grad = (1 / temp_x.shape[0]) * np.dot(temp_x.T, (np.dot(temp_x, self.w) - temp_y))
        m = beta1 * m + (1 - beta1) * grad
        m_corrected = m / (1 - (beta1 ** i))
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        v_corrected = v / (1 - (beta2 ** i))

        return self.w - (alpha * m_corrected) / (np.sqrt(v_corrected + epsilon)), m, v

    def plot_gradient():
        """
        Plots the gradient descent path for the loss function
        Useful links: 
        -   http://www.adeveloperdiary.com/data-science/how-to-visualize-gradient-descent-using-contour-plot-in-python/
        -   https://www.youtube.com/watch?v=zvp8K4iX2Cs&list=LL&index=2
        """
        # TODO: Bonus!
        pass


def main():
    regressor = Regressor()
    
    # regressor.fit(optimizer='gd', n_iters=300, tol=0.1, render_animation=True, alpha=.8)
    
    alpha = [.000001, .00001, .0001, .001, .01, .03, .05, .08, .1, .3, .5, .6, .7, .8, .9]

    # for i in alpha:
    #     regressor = Regressor()
    #     loss, iter = regressor.fit(optimizer='gd', n_iters=500, tol=.0001,
    #                                render_animation=False, monitor=False, alpha=i)
    #     print(f'for alpha = {i}, loss = {loss}, iterations = {iter}')

    ### gradient descent -> alpha = 0.8, loss = 933.0111694023019, iterations = 9


    # regressor.fit(optimizer='sgd', n_iters=1000, tol=0.1, render_animation=True, alpha=.3, batch_size=1)

    # for i in alpha:
    #     regressor = Regressor()
    #     loss, iter = regressor.fit(optimizer='sgd', n_iters=1000, tol=.1,
    #                                render_animation=False, monitor=False, alpha=i, batch_size=1)
    #     print(f'for alpha = {i}, loss = {loss}, iterations = {iter}')

    ### SGD -> alpha = 0.3, loss = 933.0125863887486, iterations = 29


    # regressor.fit(optimizer='sgdMomentum', n_iters=1000, tol=.1,
    #               render_animation=True, alpha=.01, momentum=.9, batch_size=1)

    # for i in alpha:
    #     regressor = Regressor()
    #     loss, iter = regressor.fit(optimizer='sgdMomentum', n_iters=1000, tol=.1,
    #                             render_animation=False, monitor=False, alpha=i, momentum=.9, batch_size=1)
    #     print(f'for alpha = {i}, loss = {loss}, iterations = {iter}')

    ### SGD with momentum -> alpha = 0.01, loss = 933.4152928973236, iterations = 48


    # regressor.fit(optimizer='adagrad', n_iters=150, tol=.1,
    #               render_animation=True, alpha=.99, batch_size=64)

    ### AdaGrad -> alpha = 0.99, loss = 933.0609790809508, iterations = 10436 


    # regressor.fit(optimizer='rmsprop', n_iters=1000, tol=0.1, momentum=.9,
    #               render_animation=True, alpha=.9, batch_size=64)

    # for i in alpha:
    #     regressor = Regressor()
    #     loss, iter = regressor.fit(optimizer='rmsprop', n_iters=1000, tol=.1,
    #                             render_animation=False, monitor=False, alpha=i, momentum=.9, batch_size=64)
    #     print(f'for alpha = {i}, loss = {loss}, iterations = {iter}')

    ### RMSProp -> alpha = 0.9, loss = 933.078476440518, iterations = 126

    # regressor.fit(optimizer='adam', n_iters=1000, tol=0.1, beta1=.9, beta2=.999,
    #               render_animation=False, alpha=.5, batch_size=64)

    # for i in alpha:
    #     regressor = Regressor()
    #     loss, iter = regressor.fit(optimizer='adam', n_iters=1000, tol=.1, beta1=.9, beta2=.999,
    #                                render_animation=False, monitor=False, alpha=i, batch_size=64)
    #     print(f'for alpha = {i}, loss = {loss}, iterations = {iter}')

    ### Adam -> alpha = 0.5, loss = 933.0930430063877, iterations = 40


if __name__ == '__main__':
    main()