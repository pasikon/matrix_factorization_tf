import tensorflow as tf
import pandas as pd
import numpy as np

d = {
    'U1': [5, 3, np.nan, 1],
    'U2': [4, np.nan, np.nan, 1],
    'U3': [1, 1, np.nan, 5],
    'U4': [1, np.nan, np.nan, 4],
    'U5': [np.nan, 1, 5, 4]
}
df = pd.DataFrame(data=d, index=['D1', 'D2', 'D3', 'D4']).transpose()
mtx = df.fillna(0).as_matrix().astype(np.float32)
np_mask = df.notnull()

n_dim = mtx.shape[0]
m_dim = mtx.shape[1]
k = 2

p = np.random.rand(n_dim, k).astype(np.float32)
q = np.random.rand(m_dim, k).astype(np.float32)

assert mtx.dtype == np.float32
assert mtx.shape == (5, 4)

tf_mask = tf.Variable(np_mask.values)

mtx_ph = tf.placeholder(dtype=tf.float32, shape=mtx.shape, name='inp_mtx_ph')

p_t = tf.Variable(initial_value=p, trainable=True, name='p_matrix')
q_t = tf.Variable(initial_value=q, trainable=True, name='q_matrix')
pqdot_t = tf.matmul(p_t, q_t, transpose_a=False, transpose_b=True)
loss = tf.reduce_sum(tf.square((tf.boolean_mask(mtx_ph, tf_mask) - tf.boolean_mask(pqdot_t, tf_mask))))

gradient_step = tf.train.GradientDescentOptimizer(learning_rate=.0002)
train = gradient_step.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(5000):
        sess.run(train, feed_dict={mtx_ph: mtx})
        # sess.run(train)
        if step % 500 == 0:
            print('Step: %i loss: %f' % (step, sess.run(loss, feed_dict={mtx_ph: mtx})))

    a = sess.run(pqdot_t)
    print(a)
