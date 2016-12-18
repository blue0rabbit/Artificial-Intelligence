# Lokalizacja nauczonej sieci neuronow
SAVE_PATH = "tmp/model.ckpt"

import tensorflow as tf
import sys

def main():
    # inicjalizacja zmiennych
    init()
    # przygotowanie modelu obliczen
    model_prepare()
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

# przygotowanie modelu obliczen
def model_prepare():
    ##############################################
    # pierwsza warstwa konwolucyjna, inicjalizacja zmiennych za pomoca naszych funkcji pomocniczych
    ##############################################
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    # zmieniamy rozmiary tensora, tak by mozna bylo zastosowac konwolucje,
    # 28x28 to odpowiednio szerokosc i wysokosc obrazu, ostatnia zmienna 1 - ocznacza ilosc kanalow obrazu, w tym przypadku 1 bo obrazy sa czarnobiale, wartosc -1 oznacza, ze konkretny wymiar zostanie obliczony automatycznie
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # stosujemy nasza konwolucje, funkcja relu tworzy siec neuronowa.
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # stosujemy downsampling, zeby otrzymac obraz o wymiarach 14x14
    h_pool1 = max_pool_2x2(h_conv1)

    ###################################################
    # druga warstwa konwolucyjna
    # jak poprzednio, podobrazy 5x5, 32 kanaly/wartosci z poprzedniej warstwy, tym razem obliczamy 64 wartosci
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    # jak wyzej, stosujemy konwolucje i max pooling
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    ###############################################
    # fully connected layer,
    # -> w tej warstwie kazdy neuron wnioskuje na podstawie wszystkich wartosci obliczonych w poprzedniej warstwie, czyli obraz o rozmiarze 7x7 ma 64 wartosci, obliczamy w tej warstwie 1024 nowe wartosci
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    # w tym celu zmieniamy odpowiednio wymiary tensorow z poprzedniej warstwy
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # przeciwdzialanie overfitting'owi, 
    # miejsce na prawdopodobienstwo odrzucenia neuronu metoda dropout
    # 
    # -> sekcja 'Dropout'
    global keep_prob, train_step, accuracy
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    # ostateczny wynik sieci po odrzuceniu niektorych neuronow metoda dropout
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # zdefiniowanie bledu, ktory chcemy minimalizowac,
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

    # okreslenie kroku uczacego, algorytm AdamOptimizer
    # 
    # mozna testowac na innych algorytmach optymalizacji, wystarczy zmienic
    # ponizsza linie na inny algorytm optymalizacji
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # obliczenie dokladnosci
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

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
