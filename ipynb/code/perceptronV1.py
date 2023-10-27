def perceptronV1(X, y, w = None, eta=0.1, epochs=10, random_seed=1):
    if w is None:
        randnum = np.random.RandomState(random_seed) 
        w = randnum.normal(loc=0.0, scale=0.01, size=X.shape[1])
    maxy, miny = y.max(), y.min()

    for _ in range(epochs): 
        for xi, yi in zip(X, y):
            z = np.dot(xi, w)                              # Compute net input, same as np.dot(w.T, x)
            yhat = np.where(z >= 0.0, maxy, miny)          # Apply step func and get output
            delta = eta * (yi - yhat) * xi                 # Compute delta    
            w += delta                                     # Adjust weight
            print('{},{}={}'.format(xi, yi, w))
    return w
