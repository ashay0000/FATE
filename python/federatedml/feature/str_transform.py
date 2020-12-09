#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import functools
import math

import numpy as np

from federatedml.model_base import ModelBase
from federatedml.param.str_transform_param import StrTransformParam
from federatedml.protobuf.generated import onehot_param_pb2, onehot_meta_pb2   # todo
from federatedml.statistic.data_overview import get_header
from federatedml.util import LOGGER
from federatedml.util import abnormal_detection
from federatedml.util import consts
from federatedml.util.io_check import assert_io_num_rows_equal

MODEL_PARAM_NAME = 'StrTransformParam'
MODEL_META_NAME = 'StrTransformMeta'
MODEL_NAME = 'StrTransform'


class StrTransformInnerParam(object):
    def __init__(self):
        self.col_name_maps = {}
        self.header = []
        self.transform_indexes = []
        self.transform_names = []

    def set_header(self, header):
        self.header = header
        for idx, col_name in enumerate(self.header):
            self.col_name_maps[col_name] = idx

    def set_transform_all(self):
        self.transform_indexes = [i for i in range(len(self.header))]
        self.transform_names = self.header

    def add_transform_indexes(self, transform_indexes):
        if transform_indexes is None:
            return
        for idx in transform_indexes:
            if idx >= len(self.header):
                LOGGER.warning("Adding a index that out of header's bound")
                continue
            if idx not in self.transform_indexes:
                self.transform_indexes.append(idx)
                self.transform_names.append(self.header[idx])

    def add_transform_names(self, transform_names):
        if transform_names is None:
            return
        for col_name in transform_names:
            idx = self.col_name_maps.get(col_name)
            if idx is None:
                LOGGER.warning("Adding a col_name that is not exist in header")
                continue
            if idx not in self.transform_indexes:
                self.transform_indexes.append(idx)
                self.transform_names.append(self.header[idx])


class TransferPair(object):
    def __init__(self, name):
        self.name = name
        self._values = set()
        # self._transformed_headers = {}
        self._transformed_map = {}

    def add_value(self, value):
        if value in self._values:
            return
        self._values.add(value)
        if len(self._values) > consts.ONE_HOT_LIMIT:
            raise ValueError("Input data should not have more than {} possible value when doing one-hot encode"
                             .format(consts.ONE_HOT_LIMIT))

        self._transformed_map[value] = len(self._transformed_map)

    @property
    def values(self):
        return list(self._values)


class StrTransform(ModelBase):
    def __init__(self):
        super(StrTransform, self).__init__()
        self.col_maps = {}
        self.schema = {}
        self.output_data = None
        self.model_param = StrTransformParam()
        self.inner_param: StrTransformInnerParam = None

    def _init_model(self, model_param):
        self.model_param = model_param
        # self.cols_index = model_param.cols

    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)
        self.check_schema_content(data_instances.schema)

    def fit(self, data_instances):
        self._init_params(data_instances)
        self._abnormal_detection(data_instances)
        f1 = functools.partial(self.record_new_header,
                               inner_param=self.inner_param)

        self.col_maps = data_instances.applyPartitions(f1).reduce(self.merge_col_maps)
        LOGGER.debug("Before set_schema in fit, schema is : {}, header: {}".format(self.schema,
                                                                                   self.inner_param.header))
        data_instances = self.transform(data_instances)
        LOGGER.debug("After transform in fit, schema is : {}, header: {}".format(self.schema,
                                                                                 self.inner_param.header))

        return data_instances

    @assert_io_num_rows_equal
    def transform(self, data_instances):
        self._init_params(data_instances)

        f = functools.partial(self.transfer_one_instance,
                              col_maps=self.col_maps,
                              inner_param=self.inner_param)

        new_data = data_instances.mapValues(f)
        self.set_schema(new_data)

        return new_data

    def _init_params(self, data_instances):
        if len(self.schema) == 0:
            self.schema = data_instances.schema

        if self.inner_param is not None:
            return
        self.inner_param = StrTransformInnerParam()
        # self.schema = data_instances.schema
        LOGGER.debug("In _init_params, schema is : {}".format(self.schema))
        header = get_header(data_instances)
        self.add_summary("original_dimension", len(header))
        self.inner_param.set_header(header)

        if self.model_param.transform_col_indexes == -1:
            self.inner_param.set_transform_all()
        else:
            self.inner_param.add_transform_indexes(self.model_param.transform_col_indexes)
            self.inner_param.add_transform_names(self.model_param.transform_col_names)

    @staticmethod
    def record_new_header(data, inner_param: StrTransformInnerParam):
        """
        Generate a new schema based on data value. Each new value will generate a new header.

        Returns
        -------
        col_maps: a dict in which keys are original header, values are dicts. The dicts in value
        e.g.
        cols_map = {"x1": {1 : "x1_1"},
                    ...}

        """

        col_maps = {}
        for col_name in inner_param.transform_names:
            col_maps[col_name] = TransferPair(col_name)

        for _, instance in data:
            feature = instance.features
            for col_idx, col_name in zip(inner_param.transform_indexes, inner_param.transform_names):
                pair_obj = col_maps.get(col_name)
                feature_value = feature[col_idx]
                if not isinstance(feature_value, str):
                    feature_value = math.ceil(feature_value)
                    if feature_value != feature[col_idx]:
                        raise ValueError("Onehot input data support integer or string only")
                pair_obj.add_value(feature_value)
        return col_maps

    @staticmethod
    def merge_col_maps(col_map1, col_map2):
        if col_map1 is None and col_map2 is None:
            return None

        if col_map1 is None:
            return col_map2

        if col_map2 is None:
            return col_map1

        for col_name, pair_obj in col_map2.items():
            if col_name not in col_map1:
                col_map1[col_name] = pair_obj
                continue
            else:
                col_1_obj = col_map1[col_name]
                for value in pair_obj.values:
                    col_1_obj.add_value(value)
        return col_map1

    @staticmethod
    def transfer_one_instance(instance, col_maps, inner_param):
        feature = instance.features
        result_header = inner_param.result_header
        # new_feature = [0 for _ in result_header]
        _transformed_value = {}

        for idx, col_name in enumerate(inner_param.header):
            value = feature[idx]
            if col_name not in col_maps:
                _transformed_value[col_name] = value
            else:
                pair_obj = col_maps.get(col_name)
                new_col_name = pair_obj.query_name_by_value(value)
                if new_col_name is None:
                    continue
                _transformed_value[new_col_name] = 1

        new_feature = [_transformed_value[x] if x in _transformed_value else 0 for x in result_header]

        feature_array = np.array(new_feature, dtype='float64')
        instance.features = feature_array
        return instance

    def set_schema(self, data_instance):
        data_instance.schema = self.schema

    def _get_meta(self):
        meta_protobuf_obj = onehot_meta_pb2.OneHotMeta(transform_col_names=self.inner_param.transform_names,
                                                       header=self.inner_param.header,
                                                       need_run=self.need_run)
        return meta_protobuf_obj

    def _get_param(self):  # todo
        pb_dict = {}
        for col_name, pair_obj in self.col_maps.items():
            values = [str(x) for x in pair_obj.values]
            value_dict_obj = onehot_param_pb2.ColsMap(values=values,
                                                      transformed_headers=pair_obj.transformed_headers)
            pb_dict[col_name] = value_dict_obj

        result_obj = onehot_param_pb2.OneHotParam(col_map=pb_dict)
        return result_obj

    def export_model(self):
        if self.model_output is not None:
            LOGGER.debug("Model output is : {}".format(self.model_output))
            return self.model_output
        if self.inner_param is None:
            self.inner_param = StrTransformInnerParam()
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            MODEL_META_NAME: meta_obj,
            MODEL_PARAM_NAME: param_obj
        }
        return result

    def load_model(self, model_dict):  # todo
        self._parse_need_run(model_dict, MODEL_META_NAME)
        model_param = list(model_dict.get('model').values())[0].get(MODEL_PARAM_NAME)
        model_meta = list(model_dict.get('model').values())[0].get(MODEL_META_NAME)

        self.model_output = {
            MODEL_META_NAME: model_meta,
            MODEL_PARAM_NAME: model_param
        }

        self.inner_param = StrTransformInnerParam()
        self.inner_param.set_header(list(model_meta.header))
        self.inner_param.add_transform_names(list(model_meta.transform_col_names))

        col_maps = dict(model_param.col_map)
        self.col_maps = {}
        for col_name, cols_map_obj in col_maps.items():
            if col_name not in self.col_maps:
                self.col_maps[col_name] = TransferPair(col_name)
            pair_obj = self.col_maps[col_name]
            for feature_value in list(cols_map_obj.values):
                pair_obj.add_value(eval(feature_value))

