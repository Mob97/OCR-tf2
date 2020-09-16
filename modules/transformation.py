import numpy as np 
import tensorflow as tf 

class Conv2D_BatchNormalization(tf.keras.Model):
    def __init__(self,filter, kernel_size, padding='same', stride = (1,1), use_bias = False, use_batchnorm = True, use_relu = True):
        super(Conv2D_BatchNormalization, self).__init__(name='')
        self.use_batchnorm = use_batchnorm
        self.use_bias = use_bias
        self.use_relu = use_relu
        self.conv2d = tf.keras.layers.Conv2D(filter, kernel_size, stride, padding, use_bias = use_bias)
        self.bn = tf.keras.layers.BatchNormalization()

    
    @tf.function
    def call(self,input_tensor, training = False):
        x = self.conv2d(input_tensor)
        if self.use_batchnorm:
            x = self.bn(x)
        if self.use_relu:
            x = tf.nn.relu(x)
        
        return x

class TPS_SpatialTransformerNetwork(tf.keras.Model):
    def __init__(self,F, I_size, I_r_size, I_channel_num = 1):
        '''
        input:
            F: numbers of fiducial point in a input image boundary
            I_size: (height, width ) of the input image I
            I_channel_num: the number of channels of the input image I
            I_r_size: (height, width) of the recified image_I_r
        output:
            batch_I_r: rectified image [batch_size * I_channel_num * I_r_height * I_r_width]
    
        '''
        super(TPS_SpatialTransformerNetwork, self).__init__(name = '')
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size 
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    @tf.function
    def call(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I) # batch_size x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime) # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime_reshape = tf.reshape(build_P_prime,(build_P_prime.shape[0], self.I_r_size[0], self.I_r_size[1], 2))
        batch_I_r = self.GridGenerator.grid_sample_2d(batch_I, build_P_prime_reshape)

        return batch_I_r



class LocalizationNetwork(tf.keras.Model):
    def __init__(self,F, I_channel_num):
        '''
        input:
            F: number of fiducial point in a input image boundary
            I_channel_num: the number of channels of the input image I 
        '''
        super(LocalizationNetwork, self).__init__(name = '')
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv1 = Conv2D_BatchNormalization(64,3,padding='same', stride = (1,1),use_bias = False,use_batchnorm= True,use_relu= True)
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')
        self.conv2 = Conv2D_BatchNormalization(128,3,padding='same', stride = (1,1),use_bias = False,use_batchnorm= True,use_relu= True)
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')
        self.conv3 = Conv2D_BatchNormalization(256,3,padding='same', stride = (1,1),use_bias = False,use_batchnorm= True,use_relu= True)
        self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same') # batch_size * 256 * I_height/8  * I_width/8
        self.conv4 = Conv2D_BatchNormalization(512,3,padding='same', stride = (1,1),use_bias = False,use_batchnorm= True,use_relu= True)
        self.localization_fc1 = tf.keras.layers.Dense(256)
        self.localization_fc2 = tf.keras.layers.Dense(self.F * 2)
    @tf.function
    def call(self, batch_I):
        '''
        input:
            batch_I: Batch input Image [batch_size * I_channel_num * I_height * I_width ]
        output:
            batch_C_prime: Predicted coordinates of fiducial point for input batch [batch_size * F * 2]        
        '''
        batch_size = batch_I.shape[0]
        x = batch_I
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = tf.nn.avg_pool2d(x,ksize=(x.shape[1],x.shape[2]),strides=1, padding='VALID')
        x = tf.reshape(x,(batch_size,512))
        x = self.localization_fc1(x)
        x = tf.nn.relu(x)
        x = self.localization_fc2(x)
        batch_C_prime = tf.reshape(x,(batch_size,self.F,2))
        return batch_C_prime

class GridGenerator():
    def __init__(self, F, I_r_size):
        '''
        input: 
            F: number of fiducial point in a input image boundary
            I_r_size: (height, width) of the recified image_I_r
        '''
        self.epsilon = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F) # return F * 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        self.inv_delta_C = self._build_inv_delta_C(self.F, self.C)
        self.P_hat = self._build_P_hat(self.F, self.C, self.P)

    def _build_C(self, F):
        '''
        Return coordinates of fiducial points in I_r; C
        '''
        control_pts_x = np.linspace(-1.0, 1.0, int(F /2))
        control_pts_y_top = -1 * np.ones(int(F / 2))
        control_pts_y_bottom = np.ones(int(F / 2))
        control_pts_top = np.stack([control_pts_x, control_pts_y_top], axis = 1)
        control_pts_bottom = np.stack([control_pts_x, control_pts_y_bottom], axis = 1)
        C = np.concatenate([control_pts_top, control_pts_bottom], axis = 0) #F * 2
        return C
    
    def _build_inv_delta_C(self, F, C):
        '''
        Return Inv_delta_C which is needed to calculate T
        '''
        hat_C = np.zeros((F, F), dtype = np.float)
        for i in range(0,F):
            for j in range(1,F):
                distance_vecto = np.linalg.norm(C[i] - C[j])
                hat_C[i,j] = distance_vecto
                hat_C[j,i] = distance_vecto
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)

        delta_C = np.concatenate(
        [
            np.concatenate([np.ones((F,1)), C, hat_C], axis =  1), # F * (F+3)
            np.concatenate([np.zeros((2,3)), np.transpose(C)], axis = 1), # 2 * (F+3)
            np.concatenate([np.zeros((1,3)), np.ones((1,F))], axis = 1), # ! * (F + 3)
        ], axis = 0)
        
        inv_delta_C = np.linalg.inv(delta_C)

        return inv_delta_C # (F+3)*(F+3)

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        )
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2
    
    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.epsilon))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        """ Generate Grid from batch_C_prime [batch_size x F x 2] """
        batch_size = batch_C_prime.shape[0]
        inv_delta_C = np.expand_dims(self.inv_delta_C, axis = 0)
        batch_inv_delta_C = np.tile(inv_delta_C, reps = (batch_size,1,1))
        P_hat = np.expand_dims(self.P_hat, axis = 0)
        batch_P_hat = np.tile(P_hat, reps = (batch_size,1,1))
        batch_C_prime_with_zeros = tf.cast(tf.concat([batch_C_prime, tf.zeros((batch_size, 3, 2))], axis = 1),dtype = tf.float64) # batch_size x F+3 x 2
        batch_T = tf.matmul(batch_inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2
        batch_P_prime = tf.matmul(batch_P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2
    
    def grid_sample_2d(self, inp, grid):
        in_shape = tf.shape(inp)
        in_h = in_shape[1]
        in_w = in_shape[2]

        # Find interpolation sides
        i, j = grid[..., 0], grid[..., 1]
        i = tf.cast(in_h - 1, grid.dtype) * (i + 1) / 2
        j = tf.cast(in_w - 1, grid.dtype) * (j + 1) / 2
        i_1 = tf.maximum(tf.cast(tf.floor(i), tf.int32), 0)
        i_2 = tf.minimum(i_1 + 1, in_h - 1)
        j_1 = tf.maximum(tf.cast(tf.floor(j), tf.int32), 0)
        j_2 = tf.minimum(j_1 + 1, in_w - 1)

        # Gather pixel values
        n_idx = tf.tile(tf.range(in_shape[0])[:, tf.newaxis, tf.newaxis], tf.concat([[1], tf.shape(i)[1:]], axis=0))
        q_11 = tf.gather_nd(inp, tf.stack([n_idx, i_1, j_1], axis=-1))
        q_12 = tf.gather_nd(inp, tf.stack([n_idx, i_1, j_2], axis=-1))
        q_21 = tf.gather_nd(inp, tf.stack([n_idx, i_2, j_1], axis=-1))
        q_22 = tf.gather_nd(inp, tf.stack([n_idx, i_2, j_2], axis=-1))

        # Interpolation coefficients
        di = tf.cast(i, inp.dtype) - tf.cast(i_1, inp.dtype)
        di = tf.expand_dims(di, -1)
        dj = tf.cast(j, inp.dtype) - tf.cast(j_1, inp.dtype)
        dj = tf.expand_dims(dj, -1)

        # Compute interpolations
        q_i1 = q_11 * (1 - di) + q_21 * di
        q_i2 = q_12 * (1 - di) + q_22 * di
        q_ij = q_i1 * (1 - dj) + q_i2 * dj

        return q_ij
        
if __name__ == '__main__':
    input_tensor = tf.zeros([10,100,60,1])
    TPA = TPS_SpatialTransformerNetwork(14,(100,60),(100,60),1)
    image_out = TPA(input_tensor)
    print(image_out.shape)
    