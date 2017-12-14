import numpy as np

def im2col_2d(input_layer, kernel_size):


    field_width, field_lenght, field_channel = input_layer.shape
    input_padding = np.zeros((field_width+2, field_lenght+2, field_channel))
    input_padding[1:field_width+1,1:field_lenght+1,:] = input_layer

    input_layer = input_padding
    field_width, field_lenght, field_channel = input_layer.shape

    print(field_width, field_lenght, field_channel)
    # input_layer = input_layer[:,:,0]
    # print(input_layer.shape)

    count = 0
    resize_list = []

    for i in range(0,(field_lenght-kernel_size+1)):
        for j in range(0,(field_width-kernel_size+1)):
            feature_row = []
            print(i, i + kernel_size, j, j + kernel_size)
            for k in range(0, field_channel):

                feature_row.append(input_layer[i:i+kernel_size,j:j+kernel_size,k])
            feature_row = np.concatenate(feature_row)
                # print(input_layer[i:i+kernel_size,j:j+kernel_size])
            count = count + 1
                # print(input_layer[i:i+kernel_size,j:j+kernel_size].reshape(1,kernel_size**2))
            feature_row = feature_row.reshape(1,(kernel_size**2)*field_channel)
            print(feature_row.shape)

            resize_list.append(feature_row)

    return np.transpose(np.concatenate(resize_list).reshape((field_lenght-kernel_size+1)**2, (kernel_size**2)*field_channel))


def conv_2d(input_layer, kernel_m,kernel_size):

    kernel_m = kernel_m.reshape(kernel_size**2,1)
    input_layer2 = np.matmul(input_layer, kernel_m)

    return input_layer2



def col2im_2d(input_layer,input_size, kernel_size, feat_num):

    print(input_layer.shape)

    new_image = []

    for index, row in enumerate(np.transpose(input_layer)):
        # print(row.reshape(input_size-kernel_size+1,input_size-kernel_size+1))
        new_image.append(row.reshape(input_size,input_size, 1))
        print(index, row.reshape(input_size,input_size, 1).shape)

    print(np.concatenate(new_image, axis=2).shape)

    print(input_layer.reshape(input_size,input_size, feat_num))

    return np.concatenate(new_image, axis=2)


def max_pooling(input_layer, stride_num = 2):
    row, col, features = input_layer.shape

    pool_row = row // stride_num
    pool_col = col // stride_num

    print(input_layer[:pool_row * stride_num, :pool_col * stride_num,:].reshape(pool_row, stride_num, pool_col, stride_num, features).max(axis=(1, 3)))

    return input_layer[:pool_row * stride_num, :pool_col * stride_num,:].reshape(pool_row, stride_num, pool_col, stride_num, features).max(axis=(1, 3))



def relu_layer(input_layer):
    input_layer[input_layer < 0] = 0
    return input_layer


def avg_pooling(input_layer, stride_num = 2):

    row, col = input_layer.shape

    pool_row = row // stride_num
    pool_col = col // stride_num

    print(input_layer[:pool_row * stride_num, :pool_col * stride_num].reshape(pool_row, stride_num, pool_col, stride_num).max(axis=(1, 3)))

    return input_layer[:pool_row * stride_num, :pool_col * stride_num].reshape(pool_row, stride_num, pool_col, stride_num).max(axis=(1, 3))

def full_connect_layer(input_size, output_size, input_layer):

    weight_m = np.random.rand(input_size, output_size)
    print(weight_m)

    print(np.matmul(input_layer,weight_m))

    return np.matmul(input_layer,weight_m)

def softmax_layer(input_layer):

    print("value before softmax")
    print(input_layer)

    input_layer = input_layer - np.max(input_layer)

    input_layer = np.array(list(map(lambda x: round(np.exp(x),8), input_layer.ravel())))
    print(input_layer)
    print(sum(input_layer))
    input_layer = input_layer/sum(input_layer)

    print(input_layer)

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

def get_z_L(y,w_a,_lambda):
    if y==-1:
        def f_z(z):
            return beta*z**2 - (2*beta*w_a-_lambda)*z + max(1+z,0)

        z1 = min((2*beta*w_a - _lambda)/(2*beta),-1)
        z2 = max((2*beta*w_a-_lambda-1)/(2*beta),-1)
        if f_z(z1) < f_z(z2):
            return z1
        else:
            return z2

    if y==1:
        def f_z(z):
            return beta*z**2 - (2*beta*w_a - _lambda)*z + max(1-z,0)
        z1 = min((2*beta*w_a - _lambda+1)/(2*beta),1)
        z2 = max((2*beta*w_a - _lambda)/(2*beta),1)

        if f_z(z1) < f_z(z2):
            return z1
        else:
            return z2

    else:
        print("error class: {}".format(y))
        exit()




if __name__ == '__main__':
    """

    input_layer = np.random.randn(32,32)

    kernel_1 = np.random.rand(3,3).reshape(9,1)
    print(kernel_1.shape)
    input_layer = input_layer.reshape((32,32,1))

    # input_layer = im2col_2d(input_layer, 3)
    # print(input_layer.shape)
    # input_layer2 = conv_2d(input_layer, kernel_1,3)
    # input_layer2 = col2im_2d(input_layer2, 32,3)
    # input_layer2 = relu_layer(input_layer2)
    # input_layer3 = avg_pooling(input_layer2, stride_num=2)



    # print(input_layer3.shape)


    #
    # print(input_layer3.reshape(1,input_layer3.shape[0]*input_layer3.shape[1]))
    #
    # input_layer4 = input_layer3.reshape(1,input_layer3.shape[0]*input_layer3.shape[1])
    # print(input_layer4.shape)
    # weight_4 = np.random.rand(input_layer4.shape[0],input_layer4.shape[1])
    #
    # print(weight_4)
    #
    # print(np.matmul(input_layer4, np.transpose(weight_4)))
    #
    # input_layer5 = full_connect_layer(15**2, 5,input_layer4)
    # input_layer5 = relu_layer(input_layer5)
    #
    # print(input_layer.shape)
    # softmax_layer(5, input_layer5)


    input_size = 32


    data_num = (input_size -2 )** 2

    input_num = 3**2
    layer_1_num = 36

    # layer_1_units = 15**2



    # W_1 = np.zeros((input_num,layer_1_num))
    W_1 = np.random.randn(input_num,layer_1_num)
    W_2 = np.random.randn(36*9,64*9)
    init_var = 1
    z_1 = init_var * np.random.randn(layer_1_num, data_num)
    a_1 = init_var * np.random.randn(layer_1_num, data_num)

    input_layer = im2col_2d(input_layer, 3)


    print(input_layer.shape)

    print(np.matmul(input_layer, W_1).shape)
    input_layer = np.matmul(input_layer, W_1)
    input_layer = col2im_2d(input_layer, input_size, kernel_size=3, feat_num=layer_1_num)

    input_layer = relu_layer(input_layer)

    # for x in np.nditer(input_layer, order='F'):
    #     print(x, x.shape)

    input_layer = max_pooling(input_layer, stride_num=2)
    #
    print(input_layer.shape)

    input_layer = im2col_2d(input_layer,3)
    print('W shape')
    print(W_2.shape)
    print('input shape')
    print(input_layer.shape)
    input_layer = np.matmul(input_layer,W_2)
    # input_layer = np.tensordot(W_2,input_layer, axes=1)
    print(input_layer.shape)
    input_layer = col2im_2d(input_layer, 16,3,64*9)
    input_layer = relu_layer(input_layer)

    # for x in np.nditer(input_layer, order='F'):
    #     print(x, x.shape)

    input_layer = max_pooling(input_layer, stride_num=2)
    print(input_layer.shape)

    print(input_layer.flatten().shape[0])
    input_layer = input_layer.flatten()

    W_3 = np.random.rand(input_layer.flatten().shape[0], 1024) * 0.01

    input_layer = np.matmul(input_layer, W_3)
    input_layer = relu_layer(input_layer)

    W_4 = np.random.rand(1024, 128) * 0.01
    input_layer = np.matmul(input_layer, W_4)
    input_layer = relu_layer(input_layer)

    W_5 = np.random.rand(128,10) * 0.01
    input_layer = np.matmul(input_layer, W_5)
    input_layer = relu_layer(input_layer)
    print(input_layer)

    print(input_layer.shape)
    print(input_layer)

    input_layer = softmax_layer(input_layer)

    print(input_layer)

    init_var = 1





    a0 = np.random.randn(32, 32)
    a0 = a0.reshape((32,32,1))
    a0 = im2col_2d(a0, 3)
    print(a0.shape)
    z1 = init_var * np.random.randn(1024, 9)
    z2 = init_var * np.random.randn(256, 324)
    a0_pinv = np.linalg.pinv(a0)

    W_1 = np.dot(z1, a0_pinv)
    beta = 10
    gamma = 1

    print(np.dot(np.transpose(W_2), W_2).shape)

    # print(np.matmul(np.transpose(W_2), z2).shape)

    a1_left = np.linalg.inv((beta * np.matmul( W_2,np.transpose(W_2)) + gamma * np.eye(324, dtype=float)))
    print(a1_left.shape)
    print(z1.shape)
    a1_right = (beta * np.dot(np.transpose(W_2), z2) + gamma * relu_layer(z1))
    # a1 = np.matmul(a1_left,a1_right)












    W_2 = np.zeros((layer_2_units, layer_1_units))
    z_2 = init_var * np.random.randn(layer_2_units, data_num)
    a_2 = init_var * np.random.randn(layer_1_units, data_num)

    W_3 = np.zeros((1, layer_2_units))
    z_3 = init_var * np.random.randn(1, data_num)

    """

    # a0 = np.random.randn(32, 32)
    # a0 = a0.reshape((32, 32, 1))
    # a0 = im2col_2d(a0, 3)
    #
    # print(a0.shape)
    #
    # input_num = 3**2
    # layer_1_num = 36
    # W_1 = np.zeros((input_num, layer_1_num))
    input_size = 32

    input_layer = np.random.randn(input_size,input_size)
    input_layer = input_layer.reshape((input_size,input_size,1))

    # input_layer = im2col_2d(input_layer, 3)
    # print(input_layer.shape)
    # input_layer2 = conv_2d(input_layer, kernel_1,3)
    # input_layer2 = col2im_2d(input_layer2, 32,3)
    # input_layer2 = relu_layer(input_layer2)
    # input_layer3 = avg_pooling(input_layer2, stride_num=2)



    # print(input_layer3.shape)


    #
    # print(input_layer3.reshape(1,input_layer3.shape[0]*input_layer3.shape[1]))
    #
    # input_layer4 = input_layer3.reshape(1,input_layer3.shape[0]*input_layer3.shape[1])
    # print(input_layer4.shape)
    # weight_4 = np.random.rand(input_layer4.shape[0],input_layer4.shape[1])
    #
    # print(weight_4)
    #
    # print(np.matmul(input_layer4, np.transpose(weight_4)))
    #
    # input_layer5 = full_connect_layer(15**2, 5,input_layer4)
    # input_layer5 = relu_layer(input_layer5)
    #
    # print(input_layer.shape)
    # softmax_layer(5, input_layer5)


    #
    # data_num = (input_size -2 )** 2
    #
    # kernel_size = 3
    #
    #
    # # layer_1_units = 15**2
    #
    #
    #
    # # W_1 = np.zeros((input_num,layer_1_num))
    # # W_2 = np.random.randn(36*9,64*9)
    # # init_var = 1
    # # z_1 = init_var * np.random.randn(layer_1_num, data_num)
    # # a_1 = init_var * np.random.randn(layer_1_num, data_num)
    #
    #
    #
    # input_layer = im2col_2d(input_layer, 3)
    # print('inital a0 shape')
    # print(input_layer.shape)
    #
    #
    #
    # layer_1_feature_num = 32
    # W_1 = np.random.randn(layer_1_feature_num, kernel_size**2)
    # print("W1 shape")
    # print(W_1.shape)
    #
    #
    # print('z1 shape')
    # print(np.matmul(W_1,input_layer).shape)
    # z_1 = np.matmul( W_1, input_layer)
    #
    # print(z_1.shape)
    #
    #
    # z_1 = col2im_2d(np.transpose(z_1), input_size, kernel_size=3, feat_num=layer_1_feature_num)
    #
    # print('after put it back to image')
    #
    # print(z_1.shape)
    #
    #
    # z_1 = relu_layer(z_1)
    # a_1 = max_pooling(z_1, stride_num=2)
    #
    # print("original a1")
    #
    # print(a_1.shape)
    # a_1 = im2col_2d(a_1,3)
    #
    # print("a1 after im2col")
    # print(a_1.shape)
    #
    # W_2 = np.random.randn(64, 32 * 9)
    #
    # print("W2 shape")
    # print(W_2.shape)
    #
    #
    # print('z2 shape')
    # print(np.matmul(W_2,a_1).shape)
    # z_2 = np.matmul( W_2, a_1)
    #
    # z_2 = col2im_2d(np.transpose(z_2), 16, kernel_size=3, feat_num=64)
    #
    # z_2 = relu_layer(z_2)
    # a_2 = max_pooling(z_2, stride_num=2)
    #
    # print(a_2.shape)


    # initialize

    beta = 10
    layer_1_feature_num = 10
    layer_2_feature_num = 10
    layer_3_feature_num = 128
    layer_4_feature_num = 128
    layer_5_feature_num = 10

    kernel_size = 3
    gamma = 1
    grow_rate = 5
    warm_start = 1
    err_tol = 1e-8

    input_size = 32

    a_0 = np.random.randn(input_size,input_size)
    a_0 = a_0.reshape((input_size,input_size,1))

    W_1 = np.random.randn(layer_1_feature_num, kernel_size**2)

    z_1 = np.random.randn(layer_1_feature_num, input_size**2)
    a_1 = np.random.randn(kernel_size**2, input_size**2)

    z_2 = np.random.randn(layer_2_feature_num, 16*16)

    z_3 = np.random.randn(layer_3_feature_num, 1)


    # get im2col:

    a_0 = im2col_2d(a_0, kernel_size)

    print('a0 size after im2col')
    print(a_0.shape)


    a_0_pinv = np.linalg.pinv(a_0)

    print('a0 inv size')
    print(a_0_pinv.shape)

    W_1 = np.matmul(z_1, a_0_pinv)
    print('w1 size')
    print(W_1.shape)

    W_2 = np.random.randn(layer_2_feature_num, layer_1_feature_num*9)

    a_1_left = np.linalg.inv((beta * np.matmul(np.transpose(W_2), W_2) + gamma * np.eye(layer_1_feature_num*kernel_size*kernel_size, dtype=float)))

    z_1_change = col2im_2d(np.transpose(z_1),input_size, kernel_size,layer_1_feature_num)
    print(z_1_change.shape)
    z_1_change = max_pooling(z_1_change, 2)
    print(z_1_change.shape)
    z_1_change = z_1_change.reshape(256,layer_1_feature_num)
    print(np.matmul(np.transpose(W_2), z_2).shape)
    print(np.repeat(z_1_change,9, axis=1).shape)
    z_1_change = np.transpose(np.repeat(z_1_change,9, axis=1))


    a_1_right = (beta * np.dot(np.transpose(W_2), z_2) + gamma * relu_layer(z_1_change))
    a_1 = np.matmul(a_1_left,a_1_right)
    print(a_1_left.shape,a_1_right.shape)
    print(a_1.shape)

    print(a_1)

    vget_z_l = np.vectorize(get_z_l)
    # print(np.matmul(W_1, a_0).shape)
    z_1 = vget_z_l(a_1, z_1_change)
    print(z_1)


    a_1_pinv = np.linalg.pinv(a_1)
    print(a_1_pinv.shape)
    print(z_2.shape)
    print(W_2.shape)

    W_2 = np.matmul(z_2, a_1_pinv)

    W_3 = np.random.randn(layer_3_feature_num,8*8*layer_1_feature_num)
    a_2_left = np.linalg.inv((beta * np.dot(np.transpose(W_3), W_3) + gamma * np.eye(64*layer_1_feature_num, dtype=float)))


    z_2_change = col2im_2d(np.transpose(z_2),16, kernel_size,layer_2_feature_num)
    print(z_2_change.shape)
    z_2_change = max_pooling(z_2_change, 2)
    print(z_2_change.shape)
    z_2_change = z_2_change.reshape(8*8*layer_2_feature_num,1)
    print(np.matmul(np.transpose(W_3), z_3).shape)
    # print(np.repeat(z_1_change,9, axis=1).shape)
    # z_1_change = np.transpose(np.repeat(z_1_change,9, axis=1))


    a_2_right = (beta * np.matmul(np.transpose(W_3), z_3) + gamma * relu_layer(z_2_change))

    print(a_2_left.shape, a_2_right.shape)

    a_2 = np.matmul(a_2_left, a_2_right)
    z_2 = vget_z_l(a_2, z_2_change)

    a_2_pinv = np.linalg.pinv(a_2)
    print(a_2_pinv.shape, z_3.shape)

    W_3 = np.matmul(z_3, a_2_pinv)
    W_4 = np.random.randn(layer_3_feature_num, layer_4_feature_num)

    print(np.dot(np.transpose(W_4), W_4).shape)
    a_3_left = np.linalg.inv((beta * np.dot(np.transpose(W_4), W_4) + gamma * np.eye(layer_4_feature_num, dtype=float)))

    print(W_4.shape, z_3.shape)
    z_4 = np.random.randn(layer_4_feature_num, 1)
    a_3_right = (beta * np.matmul(np.transpose(W_4), z_4) + gamma * relu_layer(z_3))

    a_3 = np.matmul(a_3_left, a_3_right)
    z_3 = vget_z_l(a_3, z_3)

    print(z_3.shape)

    # final layer

    a_3_pinv = np.linalg.pinv(a_3)
    print(a_3_pinv.shape, z_4.shape)

    W_4 = np.matmul(z_4, a_3_pinv)
    print(W_4.shape)
    W_5 = np.random.randn(layer_5_feature_num,layer_4_feature_num)
    #
    print(np.dot(np.transpose(W_5), W_5).shape)
    a_4_left = np.linalg.inv((beta * np.dot(np.transpose(W_5), W_5) + gamma * np.eye(layer_4_feature_num, dtype=float)))
    #

    z_5 = np.random.randn(layer_5_feature_num, 1)
    print(W_5.shape, z_5.shape, z_4.shape)

    a_4_right = (beta * np.matmul(np.transpose(W_5), z_5) + gamma * relu_layer(z_4))

    print(a_4_left.shape, a_4_right.shape)
    #
    a_4 = np.matmul(a_4_left, a_4_right)
    z_4 = vget_z_l(a_4, z_4)

    a_4_pinv = np.linalg.pinv(a_4)
    print(a_4.shape)

    W_5 = np.matmul(z_5, a_4_pinv)
    print(W_5.shape)

    vget_z_L = np.vectorize(get_z_L)

    # train_label = np.array([0,0,0,0,0,0,0,0,0,1])
    _lambda = np.zeros((1, 1))
    z_5 = vget_z_L(1, np.dot(W_5, a_4), _lambda)

    _lambda = _lambda + beta * (z_5 - np.dot(W_5, a_4))

    print(z_4)

    print(_lambda)












    # print(z_2.shape, a)

    # W_2 = np.zeros((layer_2_units, layer_1_units))
    # z_2 = np.random.randn(layer_2_units, data_num)
    # a_2 = np.random.randn(layer_1_units, data_num)
    #
    # W_3 = np.zeros((1, layer_2_units))
    # z_3 = np.random.randn(1, data_num)

    """







    print(input_layer.shape)

    print(np.matmul(input_layer, W_1).shape)
    input_layer = np.matmul(input_layer, W_1)
    input_layer = col2im_2d(input_layer, input_size, kernel_size=3, feat_num=layer_1_num)

    input_layer = relu_layer(input_layer)

    # for x in np.nditer(input_layer, order='F'):
    #     print(x, x.shape)

    input_layer = max_pooling(input_layer, stride_num=2)
    #
    print(input_layer.shape)

    input_layer = im2col_2d(input_layer,3)
    print('W shape')
    print(W_2.shape)
    print('input shape')
    print(input_layer.shape)
    input_layer = np.matmul(input_layer,W_2)
    # input_layer = np.tensordot(W_2,input_layer, axes=1)
    print(input_layer.shape)
    input_layer = col2im_2d(input_layer, 16,3,64*9)
    input_layer = relu_layer(input_layer)

    # for x in np.nditer(input_layer, order='F'):
    #     print(x, x.shape)

    input_layer = max_pooling(input_layer, stride_num=2)
    print(input_layer.shape)

    print(input_layer.flatten().shape[0])
    input_layer = input_layer.flatten()

    W_3 = np.random.rand(input_layer.flatten().shape[0], 1024) * 0.01

    input_layer = np.matmul(input_layer, W_3)
    input_layer = relu_layer(input_layer)

    W_4 = np.random.rand(1024, 128) * 0.01
    input_layer = np.matmul(input_layer, W_4)
    input_layer = relu_layer(input_layer)

    W_5 = np.random.rand(128,10) * 0.01
    input_layer = np.matmul(input_layer, W_5)
    input_layer = relu_layer(input_layer)
    print(input_layer)

    print(input_layer.shape)
    print(input_layer)

    input_layer = softmax_layer(input_layer)

    print(input_layer)

    """








