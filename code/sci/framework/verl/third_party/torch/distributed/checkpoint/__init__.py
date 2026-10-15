# official torch 2.6.0 set_model_state_dict API leads to OOM
# this is a copy of torch/distributed/checkpoint from torch 2.7.0

# From PyTorch:

# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# From Caffe2:

# Copyright (c) 2016-present, Facebook Inc. All rights reserved.

# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.

# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.

# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.

# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain

# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.

# All contributions by Tri Dao:
# Copyright (c) 2024 Tri Dao.
# All rights reserved.
