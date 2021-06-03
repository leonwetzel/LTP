__author__ = "Jantina Schakel, Marieke Weultjes, Leon Wetzel," \
             " and Dion Theodoridis"
__copyright__ = "Copyright 2021, Jantina Schakel, Marieke Weultjes," \
                " Leon Wetzel, and Dion Theodoridis"
__credits__ = ["Jantina Schakel", "Marieke Weultjes", "Leon Wetzel",
                    "Dion Theodoridis"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Leon Wetzel"
__email__ = "l.f.a.wetzel@student.rug.nl"
__status__ = "Development"

import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class SentenceDataset(Dataset):
    def __init__(self, data_file):
        super().__init__()

        self.corpus_frame = pd.read_csv(data_file)
        # TODO: juiste gegevens inladen

        # TODO: onderscheid maken in data en labels
        self.data = np.array()
        self.labels = np.array()

    def __len__(self):
        return len(self.corpus_frame)

    def __getitem__(self, item):
        # TODO: ervoor zorgen dat een Tensor-object geretourneerd wordt
        return item
