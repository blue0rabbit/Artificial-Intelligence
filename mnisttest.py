# Lokalizacja nauczonej sieci neuronow
SAVE_PATH = "tmp/model.ckpt"

import tensorflow as tf
import sys

def main():
    # inicjalizacja zmiennych
    init()
    # przygotowanie modelu obliczen
    #model_prepare()
    example()
    # uczenie sie sieci lub wczytanie juz nauczonego modelu
    # preferuje korzystanie z juz nauczonej sieci, poniewaz z moim CPU trwa to nawet pare godzin (:
    if len(sys.argv) > 1 and sys.argv[1] == "-learn":
        learn()
        save_checkpoint()
    else:
        restore_checkpoint()

# inicjalizacja srodowiska
def init():
    #pobieranie baze danych MNIST
    from tensorflow.examples.tutorials.mnist import input_data

    global mnist, sess, x, y_
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # one_hot=True oznacza ze dany obraz nalezy do dokladnie jednej klasy

    # inicjalizacja sesji
    sess = tf.InteractiveSession()

    # przygotowanie modelu do obliczen, funkcja placeholder inicjalizuje miejsce na zmienna, w tym przypadku typu float32
    x = tf.placeholder(tf.float32, shape=[None, 784])
    # x jest tensorem danych wejsciowych.
    # Kazdy obraz wejsciowy jest rozmiaru 28x28 pikseli -> wektor 2d mozemy zamienic na wektor jednowymiarowy o dlugosci 28*28 = 784
    # None oznacza, ze pierwszy wymiar nie jest okreslony -> chcemy moc przypisac do x dowolna liczbe wejsciowych obrazow w trakcie wykonania programu

    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    # y_ jest wektorem dlugosci 10, ktory bedzie wskazywal do ktorej klasy dany obraz nalezy.
    # Bedzie sie skladal z dokladnie jednej jedynki,

# funkcje pomocnicze do inicjalizowania zmiennych
# poczatkowe wartosci dobiera sie empirycznie, dla weight_variable sa to liczby bliskie zera, dla bias_variable stala 0.1
def weight_variable(shape):
    # truncated_normal zwaraca zmienna z rozkladu normalnego, ale obcietego do 2 odchylen standardowych od sredniej, czyli dla stddev = 0.1 mamy rozklad normalny N(0, 0.1) obciety do odcinka [-0.2, 0.2] (jesli wylosowana jest liczba spoza odcinka, to losujemy ponownie)
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial, name="weight")

def bias_variable(shape):
    # inicjalizujemy stala wartoscia 0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')

# operacja konwolucji
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# funkcja max pooling -

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# przyklad sieci neuronowej
def example():
    # definicja zmiennych, nazwy sa opcjonalne, ale nadajemy je po to, by moc zapisac model sieci i pozniej go wczytac
    W = tf.Variable(tf.zeros([784, 10]), name="W")
    # W - macierz wag
    b = tf.Variable(tf.zeros([10]), name='b')
    # b - bias
    # -> sekcja "Softmax Regressions"

    # inicjalizacja sesji
    sess.run(tf.global_variables_initializer())

    y = tf.matmul(x, W) + b
    # y = Wx + b, operacja na macierzach, matmul - matrix multiplication, x z W zamienione miejscami, by dalo sie pomnozyc - x ma rozmiar (None, 784), W - (784,10)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # ocena bledu szacowania
    # funkcja softmax_cross_entropy_with_logits 
    # ogolnie za pomoca powyzszej funkcji obliczamy odleglosc miedzy y - prawdziwe etykiety obrazow, a y_ - przewidziane przez siec neurownowa, wewnetrznie uzywana jest funkcja softmax - opisana tutaj: 
    # reduce_mean oznacza, ze chcemy zminimalizowac srednia po wszystkich 10 odleglosciach

    train_step = tf.train.GradientDescentOptimizer(
        learning_rate=0.5).minimize(cross_entropy)
    # definiujemy krok treningowy - kazemy za pomoca algorytmu  zminimalizowac wartosc zmiennej cross_entropy
    # ogolnie problem polega na dobraniu odpowiednich wartosci dla W i b, tak zeby nasza siec charakteryzowala sie jak najwieksza poprawnoscia - sprowadza sie to do problemu optymalizacji / znalezienia takich wartosci, dla ktorych blad jest najmniejszy
    # GradientDescentOptimizer jest jednym z algorytmow optymalizacji zaimplementowanych w tensorflow,
    # empirycznie dobiera sie wartosc learning_rate, od ktorej zalezy zbieznosc algorytmu do ekstremum

    # wykonujemy 1000 krokow uczacych
    for i in range(1000):
        batch = mnist.train.next_batch(100)
        # pobieramy probke rozmiaru 100 ze zbioru treningowego
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        # wykonujemy krok uczacy, okreslajac parametrem feed_dict co jest wejsciem, a co wyjsciem dla
        # naszego problemu

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # dokladnosc naszej sieci
    # tf.argmax - indeks w wektorze dla ktorego wektor przyjmuja najwieksza wartosc -> wektor jest rozkladem prawdopodobienstwa, wiec wartosc argmax jest tutaj przewidywana/prawdziwa klasa obrazu
    # sprawdzamy w ilu miejscach nasz algorym zgadzal sie z prawdziwa klasa
    # powyzszy wektor jest postaci [True, False, True, ..., False]
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # rzutujemy powyzszy wektor na {0,1} i obliczamy srednia dla calego wektora, czyli poprawnosc naszej sieci
    print("test accuracy: %s" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    # WAZNE: obliczamy tutaj dokladnosc na zbiorze testowym, nie treningowym.
    # Przypomnienie: do tej pory nasza siec uczyla sie wylacznie na rozdzielnym zbiorze treningowym. Zbior testowy jest uzywany wylacznie do sprawdzenia czy to czego siec nauczy sie na zbiorze treningowym mozna generalizowac na przypadek ogolny.


# przygotowanie modelu obliczen

# funkcja do zapisania wyuczonego modelu
def save_checkpoint():
    saver = tf.train.Saver()
    saver.save(sess, SAVE_PATH)
    print("Model saved in file: %s" % SAVE_PATH)

# wczytywanie modelu
def restore_checkpoint():
    try:
        saver = tf.train.Saver()
        saver.restore(sess, SAVE_PATH)
        print("Model restored from: %s" % SAVE_PATH)
        print_accuracy()
    except:
        print "Checkpoint file not found"

# proces uczenia sie sieci neuronowej, po 200 000 krokach dokladnosc okolo 99,2%
# trwa to 5 godzin
def learn(number_of_steps=200000):
    # 200 000 razy wykonujemy krok uczacy
    for i in range(number_of_steps):
        # wylosowanie probki rozmiaru 50 ze zbioru treningowego
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            # co 100 krok wyswietlamy postep/dokladnosc
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            # uwaga! jest to dokladnosc na zbiorze uczacym, a dokladnie na wylosowanej probce, wiec moze mocno oscylowac w trakcie calego procesu uczenia
        # wykonanie kroku uczacego, keep_prob = 0.5 - prawdopodobienstwo odrzucenia neuronu
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print_accuracy()

def print_accuracy():
    # obliczenie dokladnosci na zbiorze testowym, keep_prob = 1, poniewaz nie odrzucamy tu zadnych neuronow
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# wykonanie programu
main()
