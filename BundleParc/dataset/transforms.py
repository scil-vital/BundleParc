import numpy as np

from dipy.data import get_sphere
from dipy.reconst.shm import (
    convert_sh_from_legacy, convert_sh_descoteaux_tournier)
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


class RandSHBasisD(RandomizableTransform):
    """ Randomly change the SH basis of FODF coefficients between
    'tournier07' and 'descoteaux07', including 'legacy' versions.
    """

    def randomize(self):
        """ Select the SH basis """

        super().randomize(None)
        self._basis = self.R.choice(['tournier07', 'descoteaux07'])
        self.sphere = get_sphere(name='repulsion724').subdivide(n=1)

    def __call__(self, img):
        """ Change the SH basis of the FODF coefficients.

        Parameters
        ----------
        img: np.ndarray
        The input FODF volume.

        Returns
        -------
        img: np.ndarray
        The FODF volume with changed SH basis.
        """

        self.randomize()
        if self._do_transform:
            # Swap from C,H,W,D H,W,D,C
            new_img = img.transpose(1, 2, 3, 0)
            new_img = convert_sh_descoteaux_tournier(new_img)
            new_img = convert_sh_from_legacy(new_img, self._basis)
            new_img = new_img.transpose(3, 0, 1, 2)
            return np.ascontiguousarray(new_img.astype(np.float32))
        return img
