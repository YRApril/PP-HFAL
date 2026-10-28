from abc import ABC
from torch.utils.data import Dataset


def dataset(conf, data, net):
    if conf["data_form"] == 'integrated':
        return DataSetIntegrated(data[0], data[1], )
    if conf["data_form"] == 'sliced':
        return DataSetSliced(data[0], data[1], net, data[0].shape[1],
                             conf["num_steps"], )
    if conf["data_form"] == 'unequal_sliced':
        return DataSetUnequalSliced(data[0], data[1], data[3], net, data[0].shape[1], conf[
            "num_steps"], )


class DataSetIntegrated(Dataset, ABC):

    def __init__(self, sparse_x, m):
        self.sparse_X = sparse_x
        self.M = m


class DataSetSliced(Dataset, ABC):

    def __init__(self, sparse_x, m, net, num_slices, num_steps):
        self.sparse_X = sparse_x
        self.M = m
        self.z = net.z
        self.num_slices = num_slices
        self.num_steps = num_steps

    def __iter__(self):
        for i in range(0, int(self.num_slices / self.num_steps)):
            x = self.sparse_X[:, i * self.num_steps:(i + 1) * self.num_steps]
            z = self.z[:, i * self.num_steps:(i + 1) * self.num_steps]
            m = self.M[:, i * self.num_steps:(i + 1) * self.num_steps]
            yield x, z, m, None


class DataSetUnequalSliced(Dataset, ABC):
    def __init__(self, sparse_x, m, timestamp, net, num_slices, num_steps):
        self.sparse_X = sparse_x
        self.M = m
        self.z = net.z
        self.timestamp = timestamp
        self.num_slices = num_slices
        self.num_steps = num_steps

    def __iter__(self, ):
        for i in range(0, int(self.num_slices / self.num_steps)):
            x_steps = self.sparse_X[:, i * self.num_steps:(i + 1) * self.num_steps]
            z_steps = self.z[:, i * self.num_steps:(i + 1) * self.num_steps]
            m_steps = self.M[:, i * self.num_steps:(i + 1) * self.num_steps]
            t_steps = self.timestamp[i * self.num_steps:(i + 1) * self.num_steps]
            yield x_steps, z_steps, m_steps, t_steps
