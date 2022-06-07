#필요한 모듈들을 호출한다
import copy, numpy as np

#활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#인공신경망 클래스
class ann:
    def __init__(self, shape):
        self.ann_shape = shape
        self.weight = [np.random.normal(0, 1, (self.ann_shape[i], self.ann_shape[i + 1])) for i in range(0, len(self.ann_shape) - 1)]

    def calcul(self, input_data):
        output_data = np.array(input_data)
        for w in self.weight:
            output_data = sigmoid(np.dot(output_data, w))
        return output_data

    def mutate(self, m):
        for i in range(0, len(self.ann_shape) - 1):
            self.weight[i] += np.random.normal(0, m, (self.ann_shape[i], self.ann_shape[i + 1]))
