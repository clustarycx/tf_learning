import tensorflow as tf
import numpy as np


summary_dir = 'logs'
#
x_data = np.float32(np.random.rand(2, 100)) #  2 row * 100 column array, 
y_data = np.dot([0.100, 0.200], x_data) + 0.300  #  [1,2] * [2, 100] = [1, 100]

#
# 
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

#
loss = tf.reduce_mean(tf.square(y - y_data))

tf.summary.scalar('loss', loss)

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 
init = tf.initialize_all_variables()



merge = tf.summary.merge_all()




# 
sess = tf.Session()
sess.run(init)

summary_writer = tf.summary.FileWriter(logdir=summary_dir, graph=sess.graph)

# 
for step in range(0, 201):
    summary, _ = sess.run([merge, train])
    #sess.run(train)
    summary_writer.add_summary(summary, global_step=step)

    if step % 20 == 0:
        print("step:", step)
        print("W:", sess.run(W))
        print("b:", sess.run(b))

summary_writer.close()		
#		print step, sess.run(W), sess.run(b)