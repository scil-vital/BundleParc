import numpy as np
import os
import requests
import torch

DEFAULT_CKPT = os.path.join('checkpoints', 'bundleparc.ckpt')


def to_numpy(tensor: torch.Tensor, dtype=np.float32) -> np.ndarray:
    """ Helper function to convert a torch GPU tensor
    to numpy.
    """

    return tensor.cpu().numpy().astype(dtype)


def get_data(fodf, n_coefs):
    """ Get the data from the input files and prepare it for the model.
    This function truncates or pad the number of coefficients to fit the
    model's input and z-score normalizes the fODF data.

    Parameters
    ----------
    fodf : nibabel.Nifti1Image
        fODF data.
    n_coefs : int
        Number of SH coefficients to use.

    Returns
    -------
    fodf_data : np.ndarray
        fODF data.
    """

    # Select the first n_coefs coefficients from the fodf data and put it in
    # the first dimension. This truncates the number of coefficients if there
    # are more than n_coefs.
    input_fodf_data = fodf.get_fdata().transpose(
        (3, 0, 1, 2))[:n_coefs, ...].astype(dtype=np.float32)

    # Shape of the input fODF data
    fodf_shape = input_fodf_data.shape

    # If the input fODF has fewer than n_coefs coefficients, pad with zeros
    fodf_data = np.zeros((n_coefs, *fodf_shape[1:]), dtype=np.float32)
    fodf_data[:n_coefs, ...] = input_fodf_data

    # z-score norm
    mean = np.mean(fodf_data)
    std = np.std(fodf_data)
    fodf_data = (fodf_data - mean) / std

    return fodf_data


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def download_weights(path=DEFAULT_CKPT):
    url = 'https://zenodo.org/records/15579498/files/123_4_5_bundleparc.ckpt'
    os.makedirs(os.path.dirname(path))
    print('Downloading weights ...')
    with requests.get(url, stream=True) as r:
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(r.content)
    print('Done !')
