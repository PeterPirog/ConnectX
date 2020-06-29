import tensorflow as tf
from tensorflow.keras import layers,models
from  tensorflow.keras.optimizers import Adam


class DQNAgent:
    def __init__(self,rows,columns,action_size):
        self.rows=rows
        self.columns=columns
        self.state_size=(self.rows,self.columns)
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
        self.modelQ.save('model_Q.h5')
        self.modelQ.save_weights('model_weights_Q.h5')
        #self.entropy=0


    def __build_model_Q(self):
        #Neural Net for Deep-Q learning Model
        inputs = tf.keras.Input(shape=(self.state_size), name='input_states')

        x=tf.keras.layers.Flatten()(inputs)
        x=layers.LayerNormalization(name='normalization_layer')(x)
        x = layers.Dense(24, activation='elu',kernel_initializer='glorot_normal',name='hidden_layer1')(x)
        x = layers.Dense(64, activation='elu',kernel_initializer='glorot_normal',name='hidden_layer2')(x)
        outputs = layers.Dense(self.output_shape,activation='linear',kernel_initializer='glorot_normal',name='output_layer')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Q_values_model')
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

