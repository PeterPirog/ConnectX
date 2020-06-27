import tensorflow as tf
from tensorflow.keras import layers,models
from  tensorflow.keras.optimizers import Adam
import os


class DQNAgent:
    def __init__(self,rows,columns,action_size):
        self.rows=rows
        self.columns=columns
        self.state_size=(self.rows,self.columns,1)
        self.action_size=action_size
        #self.memory=deque()
        self.gamma=0.95 #discount rate
        self.epsilon=1.0 #exploration rate
        self.epsilon_min=0.01
        self.epsilon_decay=0.995
        self.learning_rate=0.001
        #Net data
        self.output_shape=self.columns
        self.modelQ=self.__build_model_Q()
        self.modelQ.save('model_action_predictor.h5')
        self.modelQ.save_weights('model_weights_Q.h5')
        #save model to Cloud bucket
        cmd1="gsutil cp *.h5 gs://bert-pl/ConnectX/"
        os.system(cmd1)
        #self.entropy=0


    def __build_model_Q(self):
        #Neural Net for Deep-Q learning Model
        inputs = tf.keras.Input(shape=(self.state_size), name='input_states')
        #Convolution part
        conv_1 = layers.Conv2D(32, (3, 3), activation='relu',padding='same',name='conv_1')(inputs)
        maxpool_1 =layers.MaxPooling2D((2, 2),name='maxpool_1')(conv_1)
        conv_2 = layers.Conv2D(64, (3, 3), activation='relu',name='conv_2')(maxpool_1)

        #Dense part
        flatten =layers.Flatten()(conv_2)
        x = layers.Dense(64, activation='elu',kernel_initializer='glorot_normal',name='hidden_layer2')(flatten)
        outputs = layers.Dense(self.output_shape,activation='linear',kernel_initializer='glorot_normal',name='output_layer')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Q_values_model')
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

