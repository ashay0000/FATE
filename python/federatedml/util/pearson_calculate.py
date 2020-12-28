#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import numpy as np

from federatedml.secureprotol.spdz.tensor.fixedpoint_table import FixedPointTensor, table_dot


class PearsonCalculate:

    def __init__(self):
        self.local_corr = None
        self.shapes = []
        self._summary = {}

    def _select_columns(self, data_instance):
        col_names = data_instance.schema["header"]
        return data_instance.mapValues(lambda inst: inst.features)

    @staticmethod
    def _standardized(data):
        n = data.count()
        sum_x, sum_square_x = data.mapValues(lambda x: (x, x ** 2)) \
            .reduce(lambda pair1, pair2: (pair1[0] + pair2[0], pair1[1] + pair2[1]))
        #
        sum_x = sum_x.astype('float')
        sum_square_x = sum_square_x.astype('float')
        # 原本输出结果为object类型 导致此处报错 具体原因待调研
        mu = sum_x / n  # 平均数
        sigma = np.sqrt(sum_square_x / n - mu ** 2)  # 标准差
        if (sigma <= 0).any():
            raise ValueError(f"zero standard deviation detected, sigma={sigma}")
        return n, data.mapValues(lambda x: (x - mu) / sigma)

    def fit(self, data_instance):
        # local
        data = self._select_columns(data_instance)
        n, normed = self._standardized(data)
        self.local_corr = table_dot(normed, normed)
        self.local_corr /= n
        self._summary["local_corr"] = self.local_corr.tolist()
        self._summary["num_local_features"] = n

        self.shapes.append(self.local_corr.shape[0])

        return self._summary


class PearsonFilter:
    def __init__(self, cor, iv, threshold=0.8):
        self.cor = cor
        self.iv = iv
        self.threshold = threshold
        self.features_num = len(self.iv)  # features_num = len(self.iv) features_num = len(self.cor)

    def fit(self):
        cor_pair = self.get_cor_pair()
        order = self.iv_sort()
        result = self.filter_by_iv(cor_pair, order)
        left = set(range(self.features_num)) - result
        return left

    def get_cor_pair(self):
        # 获取高度相关的列
        cor_pair = set()
        for i in range(self.features_num):
            for j in range(i + 1, self.features_num):
                if self.cor[i][j] > self.threshold:
                    cor_pair.add((i, j))
        return cor_pair

    def iv_sort(self):
        l = [(i, j) for i, j in enumerate(self.iv)]
        l.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in l]

    def filter_by_iv(self, cor_pair, order):
        remove_col = set()
        for i in order:
            remove_pair = []
            if i in remove_col:
                # 找到所有带i的pair
                for pair in cor_pair:
                    if i in pair:
                        remove_pair.append(pair)
                # 删掉所有带i的pair2
                for pair in remove_pair:
                    cor_pair.remove(pair)
                    if not cor_pair:
                        return remove_col

            else:
                # 找到所有带i的pair
                for pair in cor_pair:
                    if i in pair:
                        remove_pair.append(pair)
                        for col_index in pair:
                            if col_index != i:
                                remove_col.add(col_index)
                # 删掉所有带i的pair
                for pair in remove_pair:
                    cor_pair.remove(pair)
                    if not cor_pair:
                        return remove_col

