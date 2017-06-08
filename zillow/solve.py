import numpy as np
import random
import tensorflow as tf

def calc(data_set):
    def create_feature_sets_and_labels(test_size = 0.5):
        # known patterns (5 features) output of [1] of positions [0,4]==1
        # water, soylent, food
        features = data_set
        '''
        features = [
                [[1, 1, 1], [0, 1]],
                [[0, 0, 1], [1, 0]],
                [[1, 0, 0], [0, 1]],
                [[0, 1, 1], [0, 1]],
                [[0, 1, 0], [0, 1]],
                [[1, 1, 0], [0, 1]],
                [[1, 0, 1], [0, 1]],
                [[0, 0, 1], [1, 0]],
                [[0, 0, 1], [1, 0]],
                [[0, 0, 1], [1, 0]],
                [[0, 0, 1], [1, 0]],
                [[0, 0, 1], [1, 0]]
                ]
                '''

        # shuffle out features and turn into np.array
        random.shuffle(features)
        features = np.array(features)

        # split a portion of the features into tests
        testing_size = int(test_size*len(features))

        # create train and test lists
        train_x = list(features[:,0][:-testing_size])
        train_y = list(features[:,1][:-testing_size])
        test_x = list(features[:,0][-testing_size:])
        test_y = list(features[:,1][-testing_size:])

        return train_x, train_y, test_x, test_y

    train_x, train_y, test_x, test_y = create_feature_sets_and_labels()

# hidden layers and their nodes
    n_nodes_hl1 = 20
    n_nodes_hl2 = 20
    n_nodes_hl3 = 20

# classes in our output
    n_classes = 2
# iterations and batch-size to build out model
    hm_epochs = 400
    batch_size = 5
        
    x = tf.placeholder('float')
    y = tf.placeholder('float')

# random weights and bias for our layers
    hidden_1_layer = {'f_fum':n_nodes_hl1,
                      'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'f_fum':n_nodes_hl2,
                      'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'f_fum':n_nodes_hl3,
                      'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'f_fum':None,
                    'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'bias':tf.Variable(tf.random_normal([n_classes])),}

# our predictive model's definition
    def neural_network_model(data):

        # hidden layer 1: (data * W) + b
        l1 = tf.nn.relu(tf.matmul(data,hidden_1_layer['weight']) + hidden_1_layer['bias'])
        l1 = tf.sigmoid(l1)

        # hidden layer 2: (hidden_layer_1 * W) + b
        l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
        l2 = tf.sigmoid(l2)

        l3 = tf.nn.relu(tf.matmul(l2,hidden_3_layer['weight']) + hidden_3_layer['bias'])
        l3 = tf.sigmoid(l3)

        # output: (hidden_layer_2 * W) + b
        output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

        return output

# training our model
    def train_neural_network(x):
        # use the model definition
        prediction = neural_network_model(x)

        # formula for cost (error)
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
        # optimize for cost using GradientDescent
        optimizer = tf.train.GradientDescentOptimizer(1).minimize(cost)

        # Tensorflow session
        with tf.Session() as sess:
            # initialize our variables
            sess.run(tf.global_variables_initializer())

            # loop through specified number of iterations
            for epoch in range(hm_epochs):
                epoch_loss = 0
                i=0
                # handle batch sized chunks of training data
                while i < len(train_x):
                    start = i
                    end = i+batch_size
                    batch_x = np.array(train_x[start:end])
                    batch_y = np.array(train_y[start:end])

                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                    epoch_loss += c
                    i+=batch_size
                    last_cost = c

                # print cost updates along the way
                if (epoch% (hm_epochs/5)) == 0:
                    print('Epoch', epoch, 'completed out of',hm_epochs,'cost:', last_cost)
            
            # print accuracy of our model
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

            # print predictions using our model
            outcome = [0, 0]
            for i,t in enumerate(test_x):
                success = False
                '''
                print ('prediction for:', test_x[i])
                '''
                output = prediction.eval(feed_dict = {x: [test_x[i]]})
                # normalize the prediction values
                '''
                print('Predicted: ', tf.sigmoid(output[0][0]).eval(), tf.sigmoid(output[0][1]).eval())
                print('Actual: ',test_y[i])
                '''
                if (test_y[i][0] > test_y[i][1]) and (tf.sigmoid(output[0][0]).eval() > tf.sigmoid(output[0][1]).eval()):
                        success = True
                elif (test_y[i][0] < test_y[i][1]) and (tf.sigmoid(output[0][0]).eval() < tf.sigmoid(output[0][1]).eval()):
                    success = True
                print(success)
                if success:
                    outcome[0] += 1
                else:
                    outcome[1] += 1
            print('')
            print(t)
            print(output)
            print(test_y[i])
            print('Success: ', outcome[0])
            print('Failures: ', outcome[1])

            
    train_neural_network(x)

if __name__ == '__main__':
    calc([
                [[1, 1, 1], [0, 1]],
                [[0, 0, 1], [1, 0]],
                [[1, 0, 0], [0, 1]],
                [[0, 1, 1], [0, 1]],
                [[0, 1, 0], [0, 1]],
                [[1, 1, 0], [0, 1]],
                [[1, 0, 1], [0, 1]],
                [[0, 0, 1], [1, 0]],
                [[0, 0, 1], [1, 0]],
                [[0, 0, 1], [1, 0]],
                [[0, 0, 1], [1, 0]],
                [[0, 0, 1], [1, 0]],
                [[1, 1, 1], [0, 1]],
                [[0, 0, 1], [1, 0]],
                [[1, 0, 0], [0, 1]],
                [[0, 1, 1], [0, 1]],
                [[0, 1, 0], [0, 1]],
                [[1, 1, 0], [0, 1]],
                [[1, 0, 1], [0, 1]],
                [[0, 0, 1], [1, 0]],
                [[0, 0, 1], [1, 0]],
                [[0, 0, 1], [1, 0]],
                [[0, 0, 1], [1, 0]],
                [[0, 0, 1], [1, 0]]
                ])
