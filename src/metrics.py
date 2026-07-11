import tensorflow as tf
from tensorflow.keras.metrics import Metric


class SparseMeanIoU(Metric):
    """IoU metric compatible with sparse labels and softmax outputs."""

    def __init__(self, num_classes, name="IoU", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.conf_mat = self.add_weight(
            name="conf_mat", shape=(num_classes, num_classes),
            initializer="zeros", dtype=tf.int64
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.reshape(y_true, [-1])
        y_pred_labels = tf.reshape(y_pred_labels, [-1])
        cm = tf.math.confusion_matrix(
            y_true, y_pred_labels, num_classes=self.num_classes, dtype=tf.int64
        )
        self.conf_mat.assign_add(cm)

    def result(self):
        cm = tf.cast(self.conf_mat, tf.float32)
        inter = tf.linalg.tensor_diag_part(cm)
        union = tf.reduce_sum(cm, 0) + tf.reduce_sum(cm, 1) - inter
        iou = inter / (union + 1e-7)
        return tf.reduce_mean(iou)

    def reset_states(self):
        self.conf_mat.assign(tf.zeros_like(self.conf_mat))
