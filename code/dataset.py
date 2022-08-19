import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import dgl
import random


class OriginDataset(object):
    '''Base class for posting dataset (for demand) and work experience dataset (for supply)
    '''

    def __init__(self, data):
        super().__init__()
        self.data: pd.DataFrame = data
        # standardize time
        time_range = self.data[self.time_attr_name].value_counts().sort_index().index.tolist()
        time_map = dict(zip(time_range, list(range(len(time_range)))))
        self.data[self.time_attr_name] = self.data[self.time_attr_name].map(time_map)
        self._time_range = list(range(len(time_range)))

    @property
    def time_attr_name(self):
        raise NotImplementedError

    @property
    def time_range(self):
        return self._time_range

    @property
    def companies(self):
        return self.data['Company'].value_counts().sort_index().index.tolist()

    @property
    def positions(self):
        return self.data['Position'].value_counts().sort_index().index.tolist()

    @property
    def count(self):
        raise NotImplementedError


class PostingDataset(OriginDataset):
    '''Job posting dataset (for demand)
    '''

    def __init__(self, data_path: str):
        self.posting: pd.DataFrame = pd.read_csv(
            data_path,
            sep='\t',
            encoding='utf-8',
            dtype={'Company': str, 'Time': str, 'Position': str, 'Location': str},
            header=0,
            on_bad_lines='warn',
            index_col=False
        )
        super().__init__(self.posting)

    @property
    def time_attr_name(self):
        return 'Time'

    @property
    def count(self):
        count = self.data.groupby(by=['Company', 'Position', self.time_attr_name]).count()
        count = count[count.columns.tolist()[0]]  # choose first(any) column
        count = count.unstack(fill_value=0).stack()  # fill missing
        return count


class WorkExperienceDataset(OriginDataset):
    '''Work experience dataset (for supply)
    '''

    def __init__(self, data_path: str):
        self.work_experience: pd.DataFrame = pd.read_csv(
            data_path,
            sep='\t',
            encoding='utf-8',
            dtype={'People': np.int64, 'Company': str, 'StartDate': str, 'EndDate': str, 'Position': str},
            index_col=None,
            on_bad_lines='warn'
        )
        super().__init__(self.work_experience)
        self._time_range = self._time_range[:-1]
        self.data = self.data[self.data[self.time_attr_name].isin(self._time_range)]
        self.data = self.data.reset_index()

    @property
    def time_attr_name(self):
        return 'EndDate'

    @property
    def count(self):
        count = self.data.groupby(by=['Company', 'Position', self.time_attr_name]).count()
        count = count[count.columns.tolist()[0]]  # choose first(any) column
        count = count.unstack(fill_value=0).stack()  # fill missing
        return count

    @property
    def peoples(self):
        return self.data['People']

    def get_flow_matrix(self, type: str):
        if type == 'Company':
            n = len(self.companies)
        elif type == 'Position':
            n = len(self.positions)
        sorted_data = self.data[['People', type, 'EndDate']].sort_values(by=['People', 'EndDate'])
        m = len(sorted_data)
        d = dict(zip(self.companies if type == 'Company' else self.positions, list(range(n))))
        flow_matrix = [[[0 for _ in range(n)] for __ in range(n)] for ____ in self._time_range]
        i = 0
        j = 0
        while i < m:
            while j + 1 < m and sorted_data['People'][j + 1] == sorted_data['People'][i]:
                j += 1
            for k in range(i, j + 1 - 1):
                com_src = sorted_data[type][k]
                com_tgt = sorted_data[type][k + 1]
                com_src = d[com_src]
                com_tgt = d[com_tgt]
                time = sorted_data['EndDate'][k]
                flow_matrix[time][com_src][com_tgt] += 1
            i = j + 1
            j = i
        return flow_matrix

    @property
    def company_flow_matrix(self):
        return self.get_flow_matrix('Company')

    @property
    def position_flow_matrix(self):
        return self.get_flow_matrix('Position')


class SingleTalentTrendDataset(Dataset):

    def __init__(self, data, max_length, * args, **kwargs):
        super().__init__()
        self.max_length = max_length
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        d_x, s_x = sample['X']
        d_y, s_y = sample['Y_Label']
        l = sample['L']
        c = sample['C']
        p = sample['P']
        t_s = sample['Sta']
        t_e = sample['End']
        # padding
        d_x_padded = F.pad(input=d_x, pad=(0, self.max_length - l))
        s_x_padded = F.pad(input=s_x, pad=(0, self.max_length - l))
        return (d_x_padded, s_x_padded), (d_y, s_y), l, c, p, t_s, t_e


class TalentTrendDataset(SingleTalentTrendDataset):
    '''Joint talent demand and supply dataset
    '''

    def __init__(self, demand: PostingDataset, supply: WorkExperienceDataset, max_length: int, class_num: int, min_length: int, *args, **kwargs):
        assert len(demand.time_range) == len(supply.time_range) and demand.time_range == supply.time_range and demand.companies == supply.companies and demand.positions == supply.positions
        self.time_range = demand.time_range
        self.companies = demand.companies
        self.positions = demand.positions
        self.demand: PostingDataset = demand
        self.supply: WorkExperienceDataset = supply
        self.max_length = max_length
        self.data, self.demand_matrices, self.supply_matrices = _generate_data(demand.count, supply.count, self.time_range, self.companies, self.positions, class_num, min_length, self.company_flow_matrix, self.position_flow_matrix)
        super().__init__(self.data, self.max_length)

    @property
    def company_flow_matrix(self):
        return self.supply.company_flow_matrix

    @property
    def position_flow_matrix(self):
        return self.supply.position_flow_matrix

    @property
    def com_pos_hg(self):
        company_flow_matrix = self.supply.company_flow_matrix
        position_flow_matrix = self.supply.position_flow_matrix
        return [_construct_composhg(len(self.companies), len(self.positions), self.demand_matrices, self.supply_matrices, company_flow_matrix[ti], position_flow_matrix[ti]) for ti in self.time_range]


class Taskset(object):
    '''Taskset for meta-learning, split tasks which respectively represent data within a company.
    '''

    def __init__(self, company_num: int, position_num: int, data: list, max_length: int, *args, **kwargs):
        super().__init__()
        self.company_num = company_num
        self.position_num = position_num
        self.data = [[] for _ in range(self.company_num)]
        for sample in data:
            self.data[sample['C']].append(sample)
        self.indices = list(range(self.company_num))
        for c in self.indices:
            dataset = SingleTalentTrendDataset(self.data[c], max_length)
            self.data[c] = DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=True)
        self.losses = torch.ones(self.company_num) / self.company_num
        self.sample_probabilities = [1.0 / self.company_num for _ in range(self.company_num)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.data[index]

    def sample(self):
        index = random.choices(self.indices, k=1, weights=self.sample_probabilities)[0]
        return index, self[index]

    def record(self, index, loss):
        self.losses[index] = loss

    def update_sample_prob(self):
        self.sample_probabilities = F.softmax(self.losses, dim=0).tolist()


# ----------------------Utilities----------------------

def _generate_data(demand_count: pd.DataFrame, supply_count: pd.DataFrame, time_range: list, companies: list, positions: list, class_num: int, min_length: int, company_flow_matrix: list, position_flow_matrix: list):
    '''generate training sequence data and demand/supply normalized company-position pairwise matrix data
    '''
    demand_matrices = _get_matrices(demand_count, time_range, companies, positions)
    supply_matrices = _get_matrices(supply_count, time_range, companies, positions)
    # sampling
    data = []
    assert demand_matrices.size() == supply_matrices.size()
    max_length = demand_matrices.size(0)
    min_length = min_length
    for l in range(min_length, 36 + 1, 6):  # l: x's length
        for i in range(0, max_length - l, 3):
            for ci in range(len(companies)):
                for pi in range(len(positions)):
                    x = (demand_matrices[i: i + l, ci, pi], supply_matrices[i: i + l, ci, pi])
                    # if x[0].sum() < 0.2 or x[1].sum() < 0.2:
                    #     continue
                    y = (demand_matrices[i + l, ci, pi], supply_matrices[i + l, ci, pi])
                    sample = {'X': x, 'Y': y, 'L': l, 'C': ci, 'P': pi, 'Sta': i, 'End': i + l - 1}
                    data.append(sample)
    # labeling
    ys = []
    for sample in data:
        d_y, s_y = sample['Y']
        ys.append(d_y)
        ys.append(s_y)
    val_range_list = _get_labels(torch.stack(ys), class_num)
    # print(val_range_list)
    for sample in data:
        d_y, s_y = sample['Y']
        d_y_label = _to_label(d_y, val_range_list, class_num)
        s_y_label = _to_label(s_y, val_range_list, class_num)
        sample['Y_Label'] = (d_y_label, s_y_label)
    return data, demand_matrices, supply_matrices


def _get_matrices(count, time_range, companies, positions):
    '''get normalized company-position pairwise matrix data
    '''
    matrices = {}
    # build matrix
    for time in time_range:
        matrix = count[:, :, time].unstack()
        assert matrix.index.tolist() == companies and matrix.columns.tolist() == positions
        matrix[matrix.isna()] = 0.0
        matrix += 1.0  # avoid divided by 0
        matrices[time] = torch.from_numpy(matrix.values)
    # stack data
    matrices = torch.stack(list(matrices.values())).float()
    # normalize data
    matrices = F.normalize(matrices)
    matrices = (matrices - matrices.min()) / (matrices.max() - matrices.min())
    return matrices


def _get_labels(vec: torch.Tensor, class_num: int):
    '''split all range for data and get value range for each labels
    '''
    vec_tmp: torch.Tensor = vec.reshape(-1)
    vec_tmp, _ = vec_tmp.sort()
    n = len(vec_tmp)
    val_range_list: list = []
    for i in range(class_num):
        val_range_list.append(vec_tmp[(n // class_num) * i])
    val_range_list.append(vec_tmp[-1])
    return val_range_list


def _to_label(vec: torch.Tensor, val_range_list: list, class_num: int):
    '''map continuous values to `class_num` discrete labels for `vec` using `val_range_list`
    '''

    def _to_label_(v: float, val_range_list: list, class_num: int):
        if v < val_range_list[0]:
            return 0
        for i in range(class_num):
            if val_range_list[i] <= v <= val_range_list[i + 1]:
                return i
        return class_num - 1

    return vec.clone().apply_(lambda x: _to_label_(x, val_range_list, class_num)).long()


def _construct_composhg(nc: int, np: int, demand_matrices: torch.Tensor, supply_matrices: torch.Tensor, company_flow_matrix_ti: list, position_flow_matrix_ti: list):
    '''construct company-position heterogeneous graph
    '''
    cids, pids, dvs = [], [], []
    ciss, piss, svs = [], [], []
    for ci in range(nc):
        for pi in range(np):
            # demand edges
            dv = demand_matrices[:, ci, pi].mean()
            if dv >= 0:
                cids.append(ci)
                pids.append(pi)
                dvs.append(dv)
            # supply edges
            sv = supply_matrices[:, ci, pi].mean()
            if sv >= 0:
                ciss.append(ci)
                piss.append(pi)
                svs.append(sv)
    cids, pids, dvs = torch.tensor(cids).long(), torch.tensor(pids).long(), torch.tensor(dvs).float()
    ciss, piss, svs = torch.tensor(ciss).long(), torch.tensor(piss).long(), torch.tensor(svs).float()
    # flow matrix
    com_flow_matrix = torch.tensor(company_flow_matrix_ti).float()
    pos_flow_matrix = torch.tensor(position_flow_matrix_ti).float()
    cfmax = com_flow_matrix.max()
    pfmax = pos_flow_matrix.max()
    cfmi, cfmj, cfmx = [], [], []
    pfmi, pfmj, pfmx = [], [], []
    for ci in range(nc):
        for cj in range(nc):
            if com_flow_matrix[ci][cj] > 0:
                cfmi.append(ci)
                cfmj.append(cj)
                cfmx.append(com_flow_matrix[ci][cj] / cfmax)
    for pi in range(np):
        for pj in range(np):
            if pos_flow_matrix[pi][pj] > 0:
                pfmi.append(pi)
                pfmj.append(pj)
                pfmx.append(pos_flow_matrix[pi][pj] / pfmax)
    cfmi, cfmj, cfmx = torch.tensor(cfmi).long(), torch.tensor(cfmj).long(), torch.tensor(cfmx).float()
    pfmi, pfmj, pfmx = torch.tensor(pfmi).long(), torch.tensor(pfmj).long(), torch.tensor(pfmx).float()
    # edge index
    graph_data = {
        ('Company', 'Demand', 'Position'): (cids, pids),
        ('Position', 'Supply', 'Company'): (piss, ciss),
        ('Company', 'CompanyFlow', 'Company'): (cfmi, cfmj),
        ('Position', 'PositionFlow', 'Position'): (pfmi, pfmj)
    }
    # init graph
    graph: dgl.DGLGraph = dgl.heterograph(graph_data)
    # edge attr
    graph.edata['val'] = {
        'Demand': dvs,
        'Supply': svs,
        'CompanyFlow': cfmx,
        'PositionFlow': pfmx
    }
    return graph
