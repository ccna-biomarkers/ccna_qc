# -*- coding: utf-8 -*-

import os
import bids
import pandas as pd
import numpy as np

# import numpy as np

# k=1

# # segmentation
# seg = np.zeros((100,100), dtype='int')
# seg[30:70, 30:70] = k

# # ground truth
# gt = np.zeros((100,100), dtype='int')
# gt[30:70, 40:80] = k

# dice = np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt))

# print 'Dice similarity score is {}'.format(dice)

def get_confounds(bids_dir):
  qc_confounds = []

  if os.path.exists(bids_dir):
    layout = bids.BIDSLayout(bids_dir, validate=False)
    layout.add_derivatives(bids_dir)
    qc_confounds = pd.DataFrame(columns=['subject', 'session'])

    all_confounds = layout.get(scope='derivatives', desc='confounds', suffix='timeseries', extension='tsv')
    for confound in all_confounds:
      all_fds = confound.get_df()['framewise_displacement'].to_numpy()
      entities = { k:[confound.entities[k]] for k in ['subject', 'session', 'task', 'run'] if k in confound.entities }
      entities.update({"fds_mean": [np.nanmean(all_fds)]})
      qc_confounds = qc_confounds.append(pd.DataFrame(entities))
  else:
    raise("Directory {} does not exists!".format(bids_dir))

  return qc_confounds

if __name__ == '__main__':
  input_dir = "/home/ltetrel/Documents/data/ccna_2019"
  confounds = get_confounds(input_dir)