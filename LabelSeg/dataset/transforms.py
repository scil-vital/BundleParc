from monai.transforms import RandomizableTransform


class RandDownSampleFODFOrderD(RandomizableTransform):
    """
    TODO

    """

    def randomize(self):
        super().randomize(None)
        self._nb_coeff = self.R.choice([6, 15, 28, 45])

    def __call__(self, img):
        self.randomize()
        if self._do_transform:
            img[self._nb_coeff:] = 0.
        return img
