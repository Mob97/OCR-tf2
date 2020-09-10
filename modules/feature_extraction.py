import tensorflow as tf

class BatchNormReluConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride=(1,1), padding='same', use_relu=True, use_bn=True):
        super(BatchNormReluConv2D, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.conv2d = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding, use_bias=False)
        if use_bn:
            self.bn = tf.keras.layers.BatchNormalization()
        if use_relu:
            self.relu = tf.keras.layers.ReLU()

            
    def call(self, input_tensor, training=False):
        net = self.conv2d(input_tensor)
        if self.use_bn:
            net = self.bn(net, training)
        if self.use_relu:
            net = self.relu(net)
        return net


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(BasicBlock, self).__init__()
        
        self.conv_block_1 = BatchNormReluConv2D(filters, kernel_size=(3, 3))
        self.conv_block_2 = BatchNormReluConv2D(filters, kernel_size=(3, 3), use_relu=False)

        self.conv2d_sc = tf.keras.layers.Conv2D(filters=filters, 
                                                kernel_size=(1, 1), 
                                                use_bias=False, 
                                                padding='valid')        
        self.bn_sc = tf.keras.layers.BatchNormalization()
        
        self.relu_2 = tf.keras.layers.ReLU()
        self.filters = filters
        
    def call(self, input_tensor, training=False):
        if (input_tensor.shape[3] != self.filters):
            residual = self.conv2d_sc(input_tensor)
            residual = self.bn_sc(residual, training)
        else:
            residual = input_tensor        
            
        net = self.conv_block_1(input_tensor, training)        
        net = self.conv_block_2(net, training)        
        
        net += residual
        net = self.relu_2(net)
        return net
    
    
class Resnet(tf.keras.layers.Layer):
    def __init__(self, output_channel, layer_num_list=[1, 2, 5, 3]):
        super(Resnet, self).__init__()
        self.inplanes = output_channel // 8
        
        self.layer_num_list = layer_num_list
        
        self.output_channel_block = [output_channel // 4, output_channel // 2, output_channel, output_channel]
        self.block_list = [
            [ 
                BasicBlock(self.output_channel_block[i]) 
                for _ in range(layer_num_list[i])
            ] 
            for i in range(len(layer_num_list))
        ]
        
        self.output_channel = int(output_channel)
        
        self.conv2d_1 = BatchNormReluConv2D(self.output_channel//16, kernel_size=(3, 3))
        self.conv2d_2 = BatchNormReluConv2D(self.output_channel//8, kernel_size=(3, 3))
        
        self.max_pooling_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')        
        self.conv2d_3 = BatchNormReluConv2D(self.output_channel_block[0], kernel_size=(3, 3))
        
        self.max_pooling_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.conv2d_4 = BatchNormReluConv2D(self.output_channel_block[1], kernel_size=(3, 3))
        
        self.max_pooling_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid')
        self.conv2d_5 = BatchNormReluConv2D(self.output_channel_block[2], kernel_size=(3, 3))
        
        self.conv2d_6 = BatchNormReluConv2D(self.output_channel_block[3], kernel_size=(2, 2), stride=(2,1), padding='valid')
        self.conv2d_7 = BatchNormReluConv2D(self.output_channel_block[3], kernel_size=(2, 2), padding='valid')
        
    @tf.function
    def call(self, input_tensor, training=False):
        net = self.conv2d_1(input_tensor, training)
        net = self.conv2d_2(net, training)
        net = self.max_pooling_1(net)
        for elem in self.block_list[0]:
            net = elem(net, training)
        net = self.conv2d_3(net, training)
        net = self.max_pooling_2(net)
        for elem in self.block_list[1]:
            net = elem(net, training)        
        
        net = self.conv2d_4(net, training)
        net = tf.pad(net, paddings=[[0,0], [0,0], [1,1], [0,0]])
        net = self.max_pooling_3(net)
        for elem in self.block_list[2]:
            net = elem(net, training)  
        net = self.conv2d_5(net, training)
        net = tf.pad(net, paddings=[[0,0], [0,0], [1,1], [0,0]])
        for elem in self.block_list[3]:
            net = elem(net, training)  
        net = self.conv2d_6(net, training)
        net = self.conv2d_7(net, training)
        net = tf.squeeze(net, axis=1)
        return net