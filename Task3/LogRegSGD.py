class LogRegSGD:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def logloss(self, y_true, y_pred):
        _sum = 0
        for i in range(len(y_true)):
            _sum += (y_true[i] * np.log10(y_pred[i])) + ((1 - y_true[i]) * np.log10(1 - y_pred[i]))
        loss = -1 * (1 / len(y_true)) * _sum
    
        return loss
    
    def gradient_dw(self, x, y, w, b, alpha, N):
        dw = x * (y - sigmoid(np.dot(w,x) + b) - (alpha / N) * w)
        return dw
    
    def gradient_db(self, x, y, w, b):
        db = y - sigmoid(np.dot(w,x) + b)
        return db
    
    def pred(w,b, X):
        N = len(X)
        predict = []
        for i in range(N):
            z=np.dot(w,X[i])+b
            if sigmoid(z) >= 0.5: # sigmoid(w,x,b) returns 1/(1+exp(-(dot(x,w)+b)))
                predict.append(1)
            else:
                predict.append(0)
        return np.array(predict)
    
    