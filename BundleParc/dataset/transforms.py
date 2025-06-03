import numpy as np

from monai.transforms import RandomizableTransform


class RandDownSampleFODFOrderD(RandomizableTransform):
    """
    Randomly downsample the number of coefficients describing
    FODFs. Presumes the volume has 45 coefficients (SH order 8)
    and downsamples to 6, 15, 28 or 45 (order 2, 4, 6 or 8).
    "Downsampling" occurs by padding the truncated volume with 0.
    """

    def randomize(self):
        """ Select the SH order """

        super().randomize(None)
        self._nb_coeff = self.R.choice([6, 15, 28, 45])

    def __call__(self, img):
        """ Downsample the volume.

        Parameters
        ----------
        img: np.ndarray
            The input FODF volume.

        Returns
        -------
        img: np.ndarray
            The downsampled (in terms of coeffs.) FODF volume.
        """

        self.randomize()
        if self._do_transform:
            img[self._nb_coeff:] = 0.
        return np.ascontiguousarray(img)
