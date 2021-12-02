# -*- coding: utf-8 -*-

import os
import sys
import bids
import pandas as pd
import numpy as np
import nibabel as nib
import nibabel.processing
import nilearn
import nilearn.input_data
import nilearn.masking
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT_DIR)
import data.make_datasets

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


def dice_coefficient(source_img, target_img):
    '''Compute the Sørensen-dice coefficient between two n-d volumes'''
    source_img = source_img.astype(bool)
    target_img = target_img.astype(bool)
    intersection = np.sum(np.logical_and(source_img, target_img))
    total_elements = np.sum(source_img) + np.sum(target_img)
    dice = 2*intersection/total_elements

    return dice


def compute_qc_metrics(metadata, scrub_threshold=0.5):
    '''Compute mean framewise displacement, mask dice coefficient and mean/std images.'''
    templateflow_list = list(set(metadata['template']))
    metadata['fds_mean_raw'] = None
    metadata['fds_mean_scrubbed'] = None
    metadata['dice'] = None
    # get templateflow
    print("Fetching templateflow for {}".format(templateflow_list))
    data.make_datasets.get_templateflow(templateflow_list)

    # fist compute group fMRI mask for fMRI dice
    
    # strategy to re-use existing fMRIprep masks
    # TODO: check also strategy intersection of masks
    # https://nilearn.github.io/modules/generated/nilearn.masking.intersect_masks.html#nilearn.masking.intersect_masks
    # fmri_mask_paths = []
    # for ii in range(len(metadata)):
    #     if metadata.iloc[ii]['datatype'] == "func":
    #         fmri_mask_paths += [metadata.iloc[ii]['mask']]
    # masker = nilearn.input_data.NiftiMasker(mask_strategy="background", mask_args=)
    # group_fmri_mask = masker.fit(imgs=fmri_mask_paths).mask_img_
    #TODO: check with mask computed from EPI
    images_paths = []
    for ii in range(len(metadata)):
        if metadata.iloc[ii]['datatype'] == "func":
            images_paths += [metadata.iloc[ii]['image']]
    group_fmri_mask = nilearn.masking.compute_multi_epi_mask(images_paths, threshold=0.5)

    # loop through all metadata
    for ii in range(len(metadata)):
        # compute fds score
        if metadata.iloc[ii]['datatype'] == "func":
            all_fds = pd.read_csv(metadata.iloc[ii]['confound'], sep="\t")
            all_fds = all_fds['framewise_displacement'].to_numpy()
            fds_mean_raw = np.nanmean(all_fds)
            fds_mean_scrubbed = np.nanmean(all_fds[all_fds < scrub_threshold])
            metadata.at[ii, 'fds_mean_raw'] = fds_mean_raw
            metadata.at[ii, 'fds_mean_scrubbed'] = fds_mean_scrubbed
        # compute dice
        # loading mask
        mask_path = metadata.iloc[ii]['mask']
        target_img = nib.load(mask_path)
        # for anatomical data use template and for fMRI data use group mask
        if metadata.iloc[ii]['datatype'] == "anat":
            # loading template and resampling (assume isotropic pixel)
            voxel_size = nib.load(mask_path).affine[0, 0]
            res = "{:02d}".format(int(voxel_size))
            if res != "01":
                res = "02"
            template_name = metadata.iloc[ii]['template']
            template_path = f"{os.environ['HOME']}/.cache/templateflow/tpl-{template_name}/tpl-{template_name}_res-{res}_desc-brain_mask.nii.gz"
            source_img = nib.processing.resample_to_output(nib.load(template_path), voxel_sizes=voxel_size)
        else:
            source_img = group_fmri_mask
        # compute dice score
        # plotting.view_img(template_img, cut_coords=[0, 0, 0]).save_as_html('template.html')
        dice = dice_coefficient(source_img.get_fdata(), target_img.get_fdata())
        metadata.at[ii, 'dice'] = dice

    return metadata


def get_metadata(bids_dir):
    '''Return dict metadata with participants and files information.'''
    if not os.path.exists(bids_dir):
        raise ValueError("Directory {} does not exists!".format(bids_dir))

    qc_metadata = pd.DataFrame(
        columns=['datatype', 'image', 'mask', 'confound', 'session', 'subject', 'task', 'template'])
    layout = bids.BIDSLayout(bids_dir, validate=False)
    layout.add_derivatives(bids_dir)

    all_images = layout.get(scope="derivatives", space="MNI152NLin2009cAsym",
                            desc="preproc", suffix=["T1w", "bold"], extension="nii.gz")
    all_confounds = layout.get(
        scope="derivatives", desc="confounds", suffix="timeseries", extension="tsv")
    all_masks = layout.get(scope="derivatives", space="MNI152NLin2009cAsym",
                           desc="brain", suffix="mask", extension="nii.gz")
    # first initialize the dict with images information
    for image in all_images:
        entities = {k: [image.entities[k]] for k in [
            'datatype', 'session', 'subject', 'task'] if k in image.entities}
        entities.update({"image": image.path})
        qc_metadata = qc_metadata.append(pd.DataFrame(entities))
    # update information with confounds and anat/func masks
    for confound in all_confounds:
        row_idx = np.logical_and(qc_metadata['datatype'] == confound.entities['datatype'],
                                 qc_metadata['session'] == confound.entities['session'])
        row_idx = np.logical_and(
            row_idx, qc_metadata['subject'] == confound.entities['subject'])
        row_idx = np.logical_and(
            row_idx, qc_metadata['task'] == confound.entities['task'])
        qc_metadata.at[row_idx, ['confound']] = confound.path

    for mask in all_masks:
        row_idx = np.logical_and(qc_metadata['datatype'] == mask.entities['datatype'],
                                 qc_metadata['session'] == mask.entities['session'])
        row_idx = np.logical_and(
            row_idx, qc_metadata['subject'] == mask.entities['subject'])
        if 'task' in mask.entities.keys():
            row_idx = np.logical_and(
                row_idx, qc_metadata['task'] == mask.entities['task'])
        qc_metadata.at[row_idx, ['mask']] = mask.path
        qc_metadata.at[row_idx, ['template']] = mask.entities['space']

    return qc_metadata.reset_index(drop=True)

# TODO: to remove, was done for compatibility


def get_confounds(bids_dir):
    qc_confounds = []

    if os.path.exists(bids_dir):
        layout = bids.BIDSLayout(bids_dir, validate=False)
        layout.add_derivatives(bids_dir)
        qc_confounds = pd.DataFrame(columns=['subject', 'session'])

        all_confounds = layout.get(
            scope="derivatives", desc="confounds", suffix="timeseries", extension="tsv")
        for confound in all_confounds:
            # all_fds = confound.get_df()['framewise_displacement'].to_numpy()
            entities = {k: [confound.entities[k]] for k in [
                'subject', 'session', 'task', 'run'] if k in confound.entities}
            entities.update({"filepath": confound.path})
            # ,"fds_mean": [np.nanmean(all_fds)]})
            qc_confounds = qc_confounds.append(pd.DataFrame(entities))
    else:
        raise("Directory {} does not exists!".format(bids_dir))

    return qc_confounds.reset_index(drop=True)


if __name__ == '__main__':
    input_dir = "/home/ltetrel/Documents/data/ccna_2019"
    metadata = get_metadata(input_dir)
    metadata = compute_qc_metrics(metadata)
