# -*- coding: utf-8 -*-

import os
import glob
import re
import pickle
import pandas
import bids

def get_confounds(bids_dir):
  qc_confounds = []

  if os.path.exists(input_dir):
    layout = bids.BIDSLayout(bids_dir, validate=False)
    layout.add_derivatives(bids_dir)

    all_confounds = layout.get(scope='derivatives', desc='confounds', suffix='timeseries', extension='tsv')
    for confound in all_confounds:
      all_fds = confound.get_df()['framewise_displacement'].to_numpy()
      ents = { k:confound.entities[k] for k in ['subject', 'session', 'task', 'run'] if k in confound.entities }
      ents.update({"all_fds": all_fds})
      qc_confounds += [ents]
  else:
    raise("Directory {} does not exists!".format(input_dir))

  return qc_confounds

if __name__ == '__main__':
  input_dir = "/home/ltetrel/Documents/data/ccna_2019"
  get_confounds(input_dir)