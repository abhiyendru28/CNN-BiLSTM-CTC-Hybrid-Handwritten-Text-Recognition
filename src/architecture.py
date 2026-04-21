import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, LSTM, Bidirectional, Activation, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from src.config import IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS, TOTAL_CLASSES, SEQUENCE_TIME_STEPS
from src.logger import initialize_logger

log = initialize_logger(__name__)

class CTCLossLayer(tf.keras.layers.Layer):
    """
    Custom topological layer to compute the CTC Negative Log Likelihood during forward passes.
    Bypasses standard categorical cross-entropy limits for unaligned sequences.
    """
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None, mask=None):

        y_true, y_pred, input_length, label_length = inputs
        y_true = tf.cast(y_true, tf.int32)
        input_length = tf.cast(input_length, tf.int32)
        label_length = tf.cast(label_length, tf.int32)

        loss = tf.keras.backend.ctc_batch_cost(
            y_true, y_pred, input_length, label_length
        )
        self.add_loss(tf.reduce_mean(loss))
        return y_pred

def compile_hybrid_network():
    """
    Assembles the hybrid CNN+BiLSTM+CTC topology.
    Executes specific dimensional reductions to achieve exactly 32 time-steps.
    """
    try:
        
        tensor_input = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS), name="image_input") #128,32,1
        
        tensor_labels = Input(shape=(SEQUENCE_TIME_STEPS,), name="labels", dtype="int32")
        tensor_input_len = Input(shape=(1,), name="input_length", dtype="int32")
        tensor_label_len = Input(shape=(1,), name="label_length", dtype="int32")

        # CNN Layer
        # Layer 1
        conv_1 = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(tensor_input)
        conv_1 = BatchNormalization()(conv_1)
        conv_1 = Activation("relu")(conv_1)
        pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1) # Output dimensions: (64, 16, 32)

        # Layer 2
        conv_2 = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(pool_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_2 = Activation("relu")(conv_2)
        pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2) # Output dimensions: (32, 8, 64)

        # Layer 3
        conv_3 = Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal")(pool_2)
        conv_3 = BatchNormalization()(conv_3)
        conv_3 = Activation("relu")(conv_3)
        pool_3 = MaxPooling2D(pool_size=(1, 2))(conv_3) # Output dimensions: (32, 4, 128)

        # Flatenning
        reshape_tensor = Reshape(target_shape=(SEQUENCE_TIME_STEPS, 512))(pool_3)
        
        dense_projection = Dense(256, activation="relu", name="cnn_dense_projection")(reshape_tensor)
        dropout_layer = Dropout(0.3)(dense_projection)

        # BiLSTM Layer
        # Layer 1
        bilstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25))(dropout_layer)

        # Layer 2
        bilstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25))(bilstm_1)

        # CTC 
        # Prediction Layer
        softmax_predictions = Dense(TOTAL_CLASSES, activation="softmax", name="softmax_classifier")(bilstm_2)

        ctc_output = CTCLossLayer(name="ctc_nll_loss")(
            [tensor_labels, softmax_predictions, tensor_input_len, tensor_label_len]
        )

        training_architecture = Model(
            inputs=[tensor_input, tensor_labels, tensor_input_len, tensor_label_len], 
            outputs=ctc_output, 
            name="End_to_End_HTR_Trainer"
        )
        
        inference_architecture = Model(
            inputs=tensor_input, 
            outputs=softmax_predictions, 
            name="Production_Inference_Engine"
        )

        log.info("CNN-BiLSTM-CTC architecture compiled. Spatial tensor constrained to (32, 256).")
        return training_architecture, inference_architecture

    except Exception as exception_trace:
        log.critical(f"Graph compilation failure: {str(exception_trace)}")
        raise