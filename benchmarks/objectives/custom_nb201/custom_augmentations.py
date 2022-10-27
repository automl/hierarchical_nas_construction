import numpy as np
import torch
from PIL import Image


class CUTOUT:
    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return "{name}(length={length})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


imagenet_pca = {
    "eigval": np.asarray([0.2175, 0.0188, 0.0045]),
    "eigvec": np.asarray(
        [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ]
    ),
}


class Lighting:
    def __init__(
        self, alphastd, eigval=imagenet_pca["eigval"], eigvec=imagenet_pca["eigvec"]
    ):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.0:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype("float32")
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), "RGB")
        return img

    def __repr__(self):
        return self.__class__.__name__ + "()"
