import json

import numpy as np
from PIL import Image

from src.data_prep import create_mask_from_json


def test_create_mask_from_json_paints_damage_grade(tmp_path):
    image_path = tmp_path / "sample_post_disaster.png"
    json_path = tmp_path / "sample_post_disaster.json"
    mask_path = tmp_path / "sample_mask.png"

    Image.new("RGB", (10, 10), (0, 0, 0)).save(image_path)

    annotation = {
        "features": {
            "xy": [
                {
                    "wkt": "POLYGON ((2 2, 8 2, 8 8, 2 8, 2 2))",
                    "properties": {"damage_grade": "major-damage"},
                }
            ]
        }
    }
    json_path.write_text(json.dumps(annotation))

    ok = create_mask_from_json(str(image_path), str(json_path), str(mask_path))
    assert ok is True

    mask = np.array(Image.open(mask_path))
    assert mask[5, 5] == 3  # inside the polygon: major-damage
    assert mask[0, 0] == 0  # outside the polygon: background
