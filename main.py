import neural_network as nn
import train as t
from numpy import array


def main():
    # Тренируем нашу нейронную сеть!
    network = nn.NeuralNetwork()
    network.train(t.data, t.all_y_trues)

    # Делаем предсказания
    emily = array([-7, -3])  # 128 фунтов, 63 дюйма
    frank = array([20, 2])  # 155 фунтов, 68 дюймов
    print("Emily: %.3f" % network.feed_forward(emily))  # 0.951 - F
    print("Frank: %.3f" % network.feed_forward(frank))  # 0.039 - M


if __name__ == '__main__':
    main()
