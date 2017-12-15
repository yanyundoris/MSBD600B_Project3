import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score


# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# For simplicity, I reduce the original classification problem to a binary classification task.
circle_digit = ((y_train==8)|(y_train==0))


x_train = x_train[circle_digit]
y_train = y_train[circle_digit]

y_train[y_train==8] = 1
y_train[y_train==0] = 0

circle_digit = ((y_test==8)|(y_test==0))

x_test = x_test[circle_digit]
y_test = y_test[circle_digit]

y_test[y_test==8] = 1
y_test[y_test==0] = 0


x_test = x_test/255
x_train = x_train/255





def im2col_2d(input_layer, kernel_size):


    field_width, field_lenght, field_channel = input_layer.shape
    input_padding = np.zeros((field_width+2, field_lenght+2, field_channel))
    input_padding[1:field_width+1,1:field_lenght+1,:] = input_layer

    input_layer = input_padding
    field_width, field_lenght, field_channel = input_layer.shape

    count = 0
    resize_list = []

    for i in range(0,(field_lenght-kernel_size+1)):
        for j in range(0,(field_width-kernel_size+1)):
            feature_row = []

            for k in range(0, field_channel):
                feature_row.append(input_layer[i:i+kernel_size,j:j+kernel_size,k])

            feature_row = np.concatenate(feature_row)

            count = count + 1

            feature_row = feature_row.reshape(1,(kernel_size**2)*field_channel)
            resize_list.append(feature_row)


    output_img = np.concatenate(resize_list)
    output_img = output_img.reshape((field_lenght-kernel_size+1)**2, (kernel_size**2)*field_channel)

    return np.transpose(output_img)


def conv_2d(input_layer, kernel_m,kernel_size):

    kernel_m = kernel_m.reshape(kernel_size**2,1)
    input_layer2 = np.matmul(input_layer, kernel_m)

    return input_layer2



def col2im_2d(input_layer,input_size, kernel_size, feat_num):

    # print(input_layer.shape)

    new_image = []

    for index, row in enumerate(np.transpose(input_layer)):
        new_image.append(row.reshape(input_size,input_size, 1))

    return np.concatenate(new_image, axis=2)


def max_pooling(input_layer, stride_num = 2):
    row, col, features = input_layer.shape

    pool_row = row // stride_num
    pool_col = col // stride_num

    output_layer =  input_layer[:pool_row * stride_num, :pool_col * stride_num,:]\
        .reshape(pool_row, stride_num, pool_col, stride_num, features).max(axis=(1, 3))

    return output_layer



def relu_layer(input_layer):
    input_layer[input_layer < 0] = 0
    return input_layer


def avg_pooling(input_layer, stride_num = 2):

    row, col = input_layer.shape

    pool_row = row // stride_num
    pool_col = col // stride_num

    output_layer = input_layer[:pool_row * stride_num, :pool_col * stride_num]\
        .reshape(pool_row, stride_num, pool_col, stride_num).max(axis=(1, 3))

    return output_layer


def softmax_layer(input_layer):

    input_layer = input_layer - np.max(input_layer)

    input_layer = np.array(list(map(lambda x: round(np.exp(x),8), input_layer.ravel())))
    input_layer = input_layer/sum(input_layer)

    return input_layer


def activation(i):
    if i > 0:
        return i
    else:
        return 0

def get_z_l(a,w_a):

    def f_z(z):
        return gamma*(a-activation(z))**2 + beta*(z-w_a)**2

    z1 = max((a*gamma+w_a*beta)/(beta+gamma),0)
    result1 = f_z(z1)

    z2 = min(w_a,0)
    result2 = f_z(z2)

    if result1 <= result2:
        return z1
    else:
        return z2

def get_z_L(y,w_a,reduce_params):
    if y== 0:
        def f_z(z):
            return beta*z**2 - (2*beta*w_a-reduce_params)*z + max(1+z,0)

        z1 = min((2*beta*w_a - reduce_params)/(2*beta),-1)
        z2 = max((2*beta*w_a-reduce_params-1)/(2*beta),-1)
        if f_z(z1) < f_z(z2):
            return z1
        else:
            return z2

    if y==1:
        def f_z(z):
            return beta*z**2 - (2*beta*w_a -reduce_params)*z + max(1-z,0)
        z1 = min((2*beta*w_a -reduce_params+1)/(2*beta),1)
        z2 = max((2*beta*w_a - reduce_params)/(2*beta),1)

        if f_z(z1) < f_z(z2):
            return z1
        else:
            return z2

    else:
        print('error!')
        exit()

def get_predict(pre):
    if pre > 0:
        return 1
    else:
        return 0

def get_predict_label(test, input_size):


    # test = np.random.randn(32, 32)
    # test = test.reshape(32, 32, 1)
    test = im2col_2d(test, 3)

    test = np.matmul(np.transpose(test), np.transpose(W_1))

    test = col2im_2d(test, input_size, kernel_size=3, feat_num=layer_1_feature_num)
    test = relu_layer(test)

    test = max_pooling(test, stride_num=2)

    test = im2col_2d(test, 3)
    test = np.matmul(np.transpose(test), np.transpose(W_2))

    test = col2im_2d(test, int(input_size / 2), kernel_size=3, feat_num=layer_2_feature_num)
    test = relu_layer(test)

    test = max_pooling(test, stride_num=2)

    print(test.flatten().shape)
    test = test.flatten()

    test = np.matmul(np.transpose(test), np.transpose(W_3))
    test = relu_layer(test)

    test = np.matmul(np.transpose(test), np.transpose(W_4))
    test = relu_layer(test)

    test = np.matmul(np.transpose(test), np.transpose(W_5))
    test = relu_layer(test)
    print(test)

    pre = vget_predict(test)

    # test = softmax_layer(test)
    # test_label = test.argmax()

    print(pre)

    return pre


if __name__ == '__main__':



    # initialize

    beta = 10
    layer_1_feature_num = 10
    layer_2_feature_num = 10
    layer_3_feature_num = 128
    layer_4_feature_num = 128
    layer_5_feature_num = 1

    kernel_size = 3
    gamma = 1
    grow_rate = 1
    warm_start = 1
    err_tol = 1e-8

    input_size = 28
    input_size_pool1 = int(input_size/2)
    input_size_pool2 = int(input_size_pool1 / 2)


    train_data = [np.random.randn(input_size,input_size),np.random.randn(input_size,input_size),np.random.randn(input_size,input_size)]

    global z_1, z_2, z_3, reduce_params, W_1, W_2, W_3

    W_1 = np.zeros((layer_1_feature_num, kernel_size ** 2))
    W_2 = np.zeros((layer_2_feature_num, layer_1_feature_num * 9))
    W_3 = np.zeros((layer_3_feature_num, input_size_pool2 * input_size_pool2 * layer_1_feature_num))
    W_4 = np.zeros((layer_3_feature_num, layer_4_feature_num))
    W_5 = np.zeros((layer_5_feature_num, layer_4_feature_num))

    # W_1 = np.random.randn(layer_1_feature_num, kernel_size ** 2)
    # W_2 = np.random.randn(layer_2_feature_num, layer_1_feature_num * 9)
    # W_3 = np.random.randn(layer_3_feature_num, input_size_pool2 * input_size_pool2 * layer_1_feature_num)
    # W_4 = np.random.randn(layer_3_feature_num, layer_4_feature_num)
    # W_5 = np.random.randn(layer_5_feature_num, layer_4_feature_num)

    for interation in range(0,100):


        w1_list,w2_list,w3_list,w4_list,w5_list = [] , [],[] ,[] ,[]


        for index, a_0 in enumerate(x_train):

            old_W_1 = W_1
            old_W_2 = W_2
            old_W_3 = W_3
            old_W_4 = W_4
            old_W_5 = W_5

            a_0 = a_0.reshape((input_size,input_size,1))

            z_1 = np.random.randn(layer_1_feature_num, input_size**2)
            z_2 = np.random.randn(layer_2_feature_num, input_size_pool1*input_size_pool1)
            z_3 = np.random.randn(layer_3_feature_num, 1)
            # get im2col:

            a_0 = im2col_2d(a_0, kernel_size)

            a_0_pinv = np.linalg.pinv(a_0)

            W_1 = np.matmul(z_1, a_0_pinv)

            a_1_left = np.linalg.inv((beta * np.matmul(np.transpose(W_2), W_2) + gamma * np.eye(layer_1_feature_num*kernel_size*kernel_size, dtype=float)))

            z_1_change = col2im_2d(np.transpose(z_1),input_size, kernel_size,layer_1_feature_num)
            z_1_change = max_pooling(z_1_change, 2)
            z_1_change = z_1_change.reshape(input_size_pool1**2,layer_1_feature_num)
            z_1_change = np.transpose(np.repeat(z_1_change,9, axis=1))


            a_1_right = (beta * np.dot(np.transpose(W_2), z_2) + gamma * relu_layer(z_1_change))
            a_1 = np.matmul(a_1_left,a_1_right)


            find_min = np.vectorize(get_z_l)
            z_1 = find_min(a_1, z_1_change)

            # try:
            a_1_pinv = np.linalg.pinv(a_1)
            W_2 = np.matmul(z_2, a_1_pinv)


            a_2_left = np.linalg.inv((beta * np.dot(np.transpose(W_3), W_3) + gamma * np.eye((input_size_pool2**2)*layer_1_feature_num, dtype=float)))
            z_2_change = col2im_2d(np.transpose(z_2), input_size_pool1, kernel_size,layer_2_feature_num)
            z_2_change = max_pooling(z_2_change, 2)
            z_2_change = z_2_change.reshape(input_size_pool2*input_size_pool2*layer_2_feature_num,1)

            a_2_right = (beta * np.matmul(np.transpose(W_3), z_3) + gamma * relu_layer(z_2_change))
            a_2 = np.matmul(a_2_left, a_2_right)
            z_2 = find_min(a_2, z_2_change)

            a_2_pinv = np.linalg.pinv(a_2)
            W_3 = np.matmul(z_3, a_2_pinv)

            a_3_left = np.linalg.inv((beta * np.dot(np.transpose(W_4), W_4) + gamma * np.eye(layer_4_feature_num, dtype=float)))
            z_4 = np.random.randn(layer_4_feature_num, 1)
            a_3_right = (beta * np.matmul(np.transpose(W_4), z_4) + gamma * relu_layer(z_3))

            a_3 = np.matmul(a_3_left, a_3_right)
            z_3 = find_min(a_3, z_3)

            a_3_pinv = np.linalg.pinv(a_3)
            W_4 = np.matmul(z_4, a_3_pinv)
            a_4_left = np.linalg.inv((beta * np.dot(np.transpose(W_5), W_5) + gamma * np.eye(layer_4_feature_num, dtype=float)))
            z_5 = np.random.randn(layer_5_feature_num, 1)
            a_4_right = (beta * np.matmul(np.transpose(W_5), z_5) + gamma * relu_layer(z_4))



            a_4 = np.matmul(a_4_left, a_4_right)
            z_4 = find_min(a_4, z_4)

            a_4_pinv = np.linalg.pinv(a_4)

            W_5 = np.matmul(z_5, a_4_pinv)

            find_min_z = np.vectorize(get_z_L)

            reduce_params = np.zeros((1, 1))
            # print(index, y_train[index])
            z_5 = find_min_z(int(y_train[index]), np.dot(W_5, a_4),reduce_params)

            reduce_params = reduce_params + beta * (z_5 - np.dot(W_5, a_4))

            w1_list.append(W_1), w2_list.append(W_2), w3_list.append(W_3), w4_list.append(W_4),w5_list.append(W_5)
            # except:
            #     print("some data are not invertable")


            if index % 1000 == 0:

                print(index)

                W_1 = np.sum(w1_list, axis = 0)/len(w1_list)
                W_2 = np.sum(w2_list,axis = 0)/len(w2_list)
                W_3 = np.sum(w3_list,axis = 0)/len(w3_list)
                W_4 = np.sum(w4_list,axis = 0)/len(w4_list)
                W_5 = np.sum(w5_list,axis = 0)/len(w5_list)

                print(np.linalg.norm(W_1, 1))
                print(np.linalg.norm(W_2, 1))
                print(np.linalg.norm(W_3, 1))
                print(np.linalg.norm(W_4, 1))
                print(np.linalg.norm(W_5, 1))

                truth_array = y_test[0:20]
                pre_list = []
                for i in range(0,20):
                    test = x_test[i].reshape(input_size, input_size, 1)
                    vget_predict = np.vectorize(get_predict)

                    get_predict_label(test, input_size)
                    pre_list.append(get_predict_label(test, input_size))
                    print(y_test[i])

                pre_list = np.array(pre_list)
                print(pre_list.tolist())
                print(accuracy_score(truth_array, pre_list))













