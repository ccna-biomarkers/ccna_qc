# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import pickle
import bids
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import utils.utils as utils

def get_parser():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter, description="", epilog="""
      Documentation at https://github.com/ccna-biomarkers/ccna_qc_summary
      """)

  parser.add_argument(
    "-i", "--input_dir", required=True, default=".", help="Input data directory",
  )

  parser.add_argument(
    "-o", "--output-dir", required=False, help="Output figure data directory (default: \"./reports/figures\")",
  )

  parser.add_argument(
    "-f", "--output-file", required=False, help="Path to the pickled (\".pkl\") generated QC summary metrics (default: \"./data/qc_measures.pkl\")",
  )

  parser.add_argument(
    "--version", action="version", version=utils.get_version()
  )

  return parser.parse_args()

SUBDATASETS = ['hcptrt', 'movie10', 'friends', 'shinobi']
PYBIDS_CACHEDIR = '.pybids_cache'

def main(output_filepath = 'data/qc_measures.pkl'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    layouts = {}
    for ds_name in SUBDATASETS:
        logger.info(f"loading BIDS: {ds_name}")
        ds_path = os.path.join('data', 'cneuromod', ds_name)
        pybids_cachedir = os.path.join(ds_path, PYBIDS_CACHEDIR)
        layouts[ds_name] = bids.BIDSLayout(ds_path, database_path=pybids_cachedir)

        logger.info(f"loading BIDS derivatives: {ds_name}")
        layouts[ds_name].add_derivatives(
            os.path.join(ds_path, 'derivatives', 'fmriprep-20.2lts', 'fmriprep'),
            parent_database_path = pybids_cachedir)
        layouts[ds_name].add_derivatives(
            os.path.join(ds_path, 'derivatives', 'mriqc-0.16'),
            parent_database_path = pybids_cachedir)


    # aggregate fds, scan_date, tsnr
    qc_measures = {}

    for ds_name, layout in layouts.items():
        logger.info(f"loading qc measures: {ds_name}")
        qc_measures[ds_name] = []
        all_confounds = layout.get(scope='fMRIPrep', desc='confounds', suffix='timeseries', extension='tsv')
        for confound in all_confounds:
            all_fds = confound.get_df()['framewise_displacement'].to_numpy()
            ents = { k:confound.entities[k] \
                for k in ['subject','session','task','run'] \
                if k in confound.entities }
            qc_jsons = layout.get(scope='MRIQC', suffix='bold', extension='json', **ents)
            qc = None
            if len(qc_jsons):
                qc = qc_jsons[0].get_dict()
                del qc["bids_meta"]
            ents.update({"qc": qc, "all_fds": all_fds})
            qc_measures[ds_name].append(ents)
    with open(output_filepath, 'wb') as f:
        logger.info(f"saving qc measures in {output_filepath}")
        pickle.dump(qc_measures, f)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()