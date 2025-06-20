import json
import numpy as np

from h5py import File


from BundleParc.dataset.bundleparc_dataset import BundleParcDataset


class BundleParcPretrainDataset(BundleParcDataset):
    """ Dataset for FODF volumes stored in a HDF5 file.
    """

    def _compute_length(self, config_file: str):
        # Load subject ids to map them to an index
        subject_ids = []
        subject_bundles = {}
        train_idx, val_idx, test_idx = [], [], []
        with File(self.file_path, 'r') as f:
            with open(config_file, 'r') as g:
                config = json.load(g)

                train_split = config['train']
                val_split = config['valid']
                test_split = config['test']

                for i, subject_id in enumerate(f.keys()):
                    subject_ids.append((subject_id))

                    train_split = config['train']
                    val_split = config['valid']
                    test_split = config['test']

                    bundles = list(self.f[subject_id]['bundles'].keys())

                    subject_bundles[subject_id] = bundles

                    if subject_id in train_split:
                        train_idx.append(i)

                    elif subject_id in val_split:
                        val_idx.append(i)

                    elif subject_id in test_split:
                        test_idx.append(i)

        self.subject_bundles = subject_bundles
        return subject_ids, train_idx, val_idx, test_idx

    def _get_datum(self, i):
        # TODO: misnomer, change it for like data or something
        s = self.bundle_ids[i]
        b = np.random.choice(self.subject_bundles[s])

        return s, b
