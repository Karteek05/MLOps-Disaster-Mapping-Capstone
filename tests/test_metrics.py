import tensorflow as tf

from src.metrics import SparseMeanIoU


def test_sparse_mean_iou_matches_known_confusion_matrix():
    # true labels: [0, 0, 1, 1]; predicted labels: [0, 1, 1, 1]
    # confusion matrix (true x pred) = [[1, 1], [0, 2]]
    # iou_0 = 1 / (2 + 1 - 1) = 0.5
    # iou_1 = 2 / (2 + 3 - 2) = 2/3
    # mean  = (0.5 + 2/3) / 2 = 7/12
    y_true = tf.constant([0, 0, 1, 1])
    y_pred = tf.constant(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    metric = SparseMeanIoU(num_classes=2)
    metric.update_state(y_true, y_pred)

    result = float(metric.result().numpy())
    assert abs(result - 7 / 12) < 1e-5
