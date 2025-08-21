import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import random
import anndata as ad
import os  
import pickle
warnings.filterwarnings("ignore")


class data_process(object):
    def __init__(self, type_list, tissue_name, sample_size=45, train_sample_num=6000, test_sample_num=1000,                    num_artificial_cells=10, random_type='CellType'):
        self.tissue_name = tissue_name
        self.random_type = random_type
        self.type_list = type_list
        self.train_sample_num = train_sample_num
        self.celltype_num = len(self.type_list)
        self.sample_size = sample_size
        self.test_sample_num = test_sample_num
        self.num_artificial_cells = num_artificial_cells

    def build_pseudo_bulk_no_noise(self, data, purpose):
        data_x = pd.DataFrame(data.X)
        data_x = data_x.fillna(0)
        data_x[data_x < 0] = 0
        data_y = pd.DataFrame(data.obs[self.random_type])
        data_y.reset_index(inplace=True, drop=True)

        x_sim = []
        y = []

        print(f"Generating {purpose} pseudo_bulk samples...")
        inx = 0
        if purpose == 'train':
            total_num = self.train_sample_num
        else:
            total_num = self.test_sample_num
        with tqdm(total=total_num, desc=f"{purpose} Samples") as pbar:
            while len(x_sim) < total_num:
                result = self.mix_cells(data_x, data_y, cell_type_list=self.type_list)
                if result is None:
                    continue
                sample, label = result
                x_sim.append(sample)
                y.append(label)
                inx += 1
                pbar.update(1)
                if inx >= total_num:  
                    break
        return x_sim, y

    def build_train_pseudo_bulk_with_noise(self, train_x, noise, noise_limit):
        data_x = pd.DataFrame(noise.X)
        data_x = data_x.fillna(0)
        data_x[data_x < 0] = 0

        train_with_noise = []

        num_noise_limit = max(1, int(np.floor(noise_limit * self.sample_size)))
        num_noise_list = list(range(1, num_noise_limit + 1))
        for x in train_x:
            num_noise = random.choice(num_noise_list)
            noise_fraction = np.random.randint(0, noise.shape[0], num_noise)
            noise_sub = noise.to_df().iloc[noise_fraction, :]
            noise_sub = noise_sub.sum(axis=0)
            train_with_noise.append(pd.Series(noise_sub.values + x.values))
        return train_with_noise

    def build_test_pseudo_bulk_with_noise(self, test_x, noise, noise_limit):
        data_x = pd.DataFrame(noise.X)
        data_x = data_x.fillna(0)
        data_x[data_x < 0] = 0

        test_with_noise_all = []

        num_noise_limit = int(np.floor(noise_limit * self.sample_size))
        num_noise_list = list(range(1, num_noise_limit + 1))

        for num_noise in num_noise_list:
            test_with_noise = []
            for x in test_x:
                noise_fraction = np.random.randint(0, noise.shape[0], num_noise)
                noise_sub = noise.to_df().iloc[noise_fraction, :]
                noise_sub = noise_sub.sum(axis=0)
                test_with_noise.append(pd.Series(noise_sub.values + x.values))
            test_with_noise_all.append(test_with_noise)
        return test_with_noise_all

    def build_artificial_cell(self, data, num):
        type_data = {
            ct: data[data.obs['CellType'] == ct].to_df().values
            for ct in self.type_list
        }
        num_metabolites = len(data.var_names)
        num_types = len(self.type_list)
        
        artificial_cells = []
        for _ in range(num):
            weights = np.random.dirichlet(np.ones(num_types), size=num_metabolites) 
            
            sampled_indices = {
                ct: np.random.choice(type_data[ct].shape[0], size=num_metabolites)
                for ct in self.type_list
            }
            
            sampled_values = np.array([
                type_data[ct][sampled_indices[ct], np.arange(num_metabolites)]
                for ct in self.type_list
            ])  
            
            weighted_sum = (weights * sampled_values.T).sum(axis=1) 
            artificial_cells.append(weighted_sum)
        
        return ad.AnnData(np.array(artificial_cells), var=pd.DataFrame(index=data.var_names))


    def mix_cells(self, x, y, cell_type_list):
        fracs = self.mixup_fraction(len(cell_type_list))
        samp_fracs = np.multiply(fracs, self.sample_size)
        samp_fracs = list(map(round, samp_fracs))
        fracs = np.divide(samp_fracs, sum(samp_fracs))
        # Make complete fracions

        fracs_complete = [0] * len(cell_type_list)

        for i, act in enumerate(cell_type_list):
            idx = cell_type_list.index(act)
            fracs_complete[idx] = fracs[i]

        artificial_samples = []
        for i, ct in enumerate(cell_type_list):
            cells_sub = x.loc[y[self.random_type] == ct]
            if cells_sub.shape[0] > 0 and samp_fracs[i] <= len(cells_sub):  
                cells_fraction = np.random.randint(0, cells_sub.shape[0], samp_fracs[i])
                cells_sub = cells_sub.iloc[cells_fraction, :]
                artificial_samples.append(cells_sub)
            else:
                return None

        df_samp = pd.concat(artificial_samples, axis=0)
        df_samp = df_samp.sum(axis=0)

        return df_samp, fracs_complete

    def mixup_fraction(self, cell_num):
        fracs = np.random.rand(cell_num)
        fracs_sum = np.sum(fracs)
        fracs = np.divide(fracs, fracs_sum)
        return fracs

    def normalize(self, series_list):
        normalized_series_list = []

        for series in series_list:
            max_value = series.max()
            normalized_series = series / max_value
            normalized_series_list.append(normalized_series)

        return normalized_series_list
    
    def fit(self, train_data, test_data, test_noise_cells = None):
        path = os.path.join('data', self.tissue_name, f'{self.tissue_name}{len(self.type_list)}cell.pkl')

        if os.path.exists(path):
            print('The data processing is complete')
        else:
            print('Generating artificial cells...')
            artificial_cells = self.build_artificial_cell(train_data, self.num_artificial_cells)
            train_x_sim, train_y = self.build_pseudo_bulk_no_noise(train_data, 'train')
            train_with_noise_1 = self.build_train_pseudo_bulk_with_noise(train_x_sim, noise=artificial_cells, noise_limit=0.1)
            train_with_noise_2 = self.build_train_pseudo_bulk_with_noise(train_x_sim, noise=artificial_cells, noise_limit=0.1)
            test_x_sim, test_y = self.build_pseudo_bulk_no_noise(test_data, 'test')
            train = [train_x_sim, train_with_noise_1, train_with_noise_2, train_y]
            test = [test_x_sim, test_y]

            if test_noise_cells:
                test_with_noise_all = self.build_test_pseudo_bulk_with_noise(test_x_sim, test_noise_cells, noise_limit=0.2)
            test_with_noise_all = []
            with open(f'data/{self.tissue_name}/{self.tissue_name}{len(self.type_list)}cell_nonorm.pkl', 'wb') as f:
                pickle.dump(train, f)
                pickle.dump(test, f)
                pickle.dump(test_with_noise_all, f)


            train_x_sim = self.normalize(train_x_sim)
            train_with_noise_1 = self.normalize(train_with_noise_1)
            train_with_noise_2 = self.normalize(train_with_noise_2)
            test_x_sim = self.normalize(test_x_sim)
            test_with_noise = []
            if test_noise_cells:
                for x in test_with_noise_all:
                    x = self.normalize(x)
                    test_with_noise.append(x)
            train = [train_x_sim, train_with_noise_1, train_with_noise_2, train_y]
            test = [test_x_sim, test_y]
            with open(path, 'wb') as f:
                pickle.dump(train, f)
                pickle.dump(test, f)
                pickle.dump(test_with_noise, f)
            train_data.write_h5ad(f'data/{self.tissue_name}/ref_cell.h5ad')
            print('The data processing is complete')