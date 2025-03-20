import json
import numpy as np

from h5py import File


from LabelSeg.dataset.labelseg_dataset import LabelSegDataset


class LabelSegPretrainDataset(LabelSegDataset):
    """ Dataset for FODF volumes stored in a HDF5 file.
    """

    def _compute_length(self, config_file: str):
        # Load subject ids to map them to an index
        subject_ids = []
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

                    if subject_id in train_split:
                        train_idx.append(i)

                    elif subject_id in val_split:
                        val_idx.append(i)

                    elif subject_id in test_split:
                        test_idx.append(i)

        return subject_ids, train_idx, val_idx, test_idx

    def _get_datum(self, i):
        # TODO: misnomer, change it for like data or something
        s = self.bundle_ids[i]
        b_i = np.random.choice(len(self.bundle_set))
        b = self.bundle_set[b_i]

        return s, b
