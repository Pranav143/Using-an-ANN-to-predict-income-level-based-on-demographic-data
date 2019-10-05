import torch.utils.data as data


class AdultDataset(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        ######

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        ######


        sample_train = self.X[index]
        sample_label = self.y[index]

        return sample_train, sample_label

        ######
