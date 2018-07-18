import tensorflow as tf
from data import handle_data
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

x = tf.placeholder('float', shape=[None, 6, 7])
y_ = tf.placeholder('float', shape=[None, 928])

x = tf.reshape(x, [-1, 6, 7, 1])

# convolution layer 1
w_conv1 = tf.Variable(tf.truncated_normal([2, 2, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
h1 = tf.nn.relu(tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

#convolution layer 2
w_conv2 = tf.Variable(tf.truncated_normal([2, 2, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h2 = tf.nn.relu(tf.nn.conv2d(h1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h2, keep_prob)

h3 = tf.reshape(h_fc1_drop, [-1, 6 * 7 * 64])
w_fc = tf.Variable(tf.truncated_normal([6 * 7 * 64, 928], stddev=0.1))
b_fc = tf.Variable(tf.constant(0.1, shape=[928]))
y = tf.nn.softmax(tf.matmul(h3, w_fc) + b_fc)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
sess.run(tf.global_variables_initializer())


accuracy_list = []
dataset, label = handle_data(True)
for r in range(400):
    print('round %d : ' % r)
    temp_accuracy = []
    for i in range(61):
        start = i * 100
        end = (i + 1) * 100
        if i is 60:
            end = 6096
        test_data = dataset[start:end]
        test_label = label[start:end]
        train.run(feed_dict={x: test_data, y_: test_label, keep_prob: 0.8})
        train_accuracy = accuracy.eval(feed_dict={
            x: test_data, y_: test_label, keep_prob: 1.0})
        if i % 10 is 0:
            temp_accuracy.append(train_accuracy)
            print("step %d, training accuracy %g" % (i, train_accuracy))
    accuracy_list.append(sum(temp_accuracy) / len(temp_accuracy))

plt.plot([i for i in range(400)], accuracy_list, 'r--')
plt.xlim([-1, 401])
plt.ylim([0, 1])
plt.xlabel('learning period')
plt.ylabel('prediction accuracy')
plt.title('Learning rate - time / slicing window size: 2 x 2')
plt.show()

# print(len(dataset))