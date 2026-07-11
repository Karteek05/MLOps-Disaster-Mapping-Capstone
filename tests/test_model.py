import tensorflow as tf

from src.model import get_resnet_unet_model


def test_model_output_shape_matches_img_size_and_num_classes():
    img_size = 64
    num_classes = 5

    model = get_resnet_unet_model(
        img_size=img_size,
        num_channels=6,
        num_classes=num_classes,
        pretrained=False,  # no network access in tests
    )

    output = model(tf.zeros((1, img_size, img_size, 6)))

    assert output.shape == (1, img_size, img_size, num_classes)
