# -*- coding: utf-8 -*-

import os
import sys
import bids
import pandas as pd
import numpy as np
import nibabel as nib
import nilearn as nil
import nilearn.masking
import nilearn.image
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


def resample_img(img_path, img, affine, shape, interpolation="continuous"):
    '''resample nifti image and save'''
    resampled_img = nil.image.resample_img(
        img, target_affine=affine, target_shape=shape, interpolation=interpolation)
    new_img_filename = os.path.basename(
        img_path.replace(".nii", "_resampled.nii"))
    image_path = os.path.join(ROOT_DIR, "..", "data",
                              "interim", new_img_filename)
    nib.save(resampled_img, image_path)

    return image_path


def fix_voxel_size(metadata):
    '''fix voxel size for func/anat if it does not match the default'''
    # TODO: get most frequent affine and shape from images
    func_affine = np.array(([[3.5,  0.,  0.,  -96.5],
                             [0., 3.5,  0., -132.5],
                             [0.,  0., 3.5,  -78.5],
                             [0.,  0.,  0.,    1.]]))
    func_shape = (56, 67, 56)
    anat_affine = np.array(([[1,  0.,  0.,  -96],
                            [0., 1,  0., -132],
                            [0.,  0., 1,  -78],
                            [0.,  0.,  0.,  1]]))
    anat_shape = (193, 229, 193)
    for ii in range(len(metadata)):
        metadata_row = metadata.iloc[ii]
        img = nib.load(metadata_row['image'])
        mask_img = nib.load(metadata_row['mask'])
        # resampling
        if (metadata_row['datatype'] == "anat") & (not np.isclose(anat_affine, img.affine, atol=1e-6).all()):
            print("\tResampling {} image/mask for {} from affine:\n {} \n to \n {}".format(
                metadata_row['datatype'], metadata_row['subject'], img.affine, anat_affine))
            metadata.at[ii, 'mask'] = resample_img(
                img_path=metadata_row['mask'], img=mask_img, affine=anat_affine, shape=anat_shape, interpolation="nearest")
        elif (metadata_row['datatype'] == "func") & (not np.isclose(func_affine, img.affine, atol=1e-6).all()):
            print("\tResampling {} image/mask for {} from affine:\n {} \n to \n {}".format(
                metadata_row['datatype'], metadata_row['subject'], img.affine, func_affine))
            # functionnal images are needed for group mask computation
            metadata.at[ii, 'image'] = resample_img(
                img_path=metadata_row['image'], img=img, affine=func_affine, shape=func_shape)
            metadata.at[ii, 'mask'] = resample_img(
                img_path=metadata_row['mask'], img=mask_img, affine=func_affine, shape=func_shape, interpolation="nearest")

    return metadata


def compute_group_fmri_mask(metadata):
    '''Compute group fmri mask using func images'''
    # mask computed from EPI images
    images_paths = [metadata_row['image'] for _, metadata_row in metadata.iterrows(
    ) if metadata_row['datatype'] == "func"]
    group_fmri_mask = nil.masking.compute_multi_epi_mask(
        images_paths, threshold=0.5, target_affine=None, target_shape=None)
    # TODO: nilearn masker strategy
    # fmri_mask_paths = [metadata_row['mask'] for _, metadata_row in metadata.iterrows(
    #     ) if metadata_row['datatype'] == "func"]
    # masker = nil.input_data.NiftiMasker(mask_strategy="background")
    # group_fmri_mask = masker.fit(imgs=fmri_mask_paths).mask_img_
    # TODO: check also strategy intersection of masks
    # https://nilearn.github.io/modules/generated/nilearn.masking.intersect_masks.html#nilearn.masking.intersect_masks

    return group_fmri_mask


def compute_qc_metrics(metadata, scrub_threshold=0.5, fix_affine=True):
    '''Compute mean framewise displacement, mask dice coefficient and mean/std images.'''
    templateflow_list = list(set(metadata['template']))
    metadata['fds_mean_raw'] = np.nan
    metadata['fds_mean_scrubbed'] = np.nan
    metadata['dice'] = np.nan
    # get templateflow
    print("\tFetching templateflow for {}".format(templateflow_list))
    data.make_datasets.get_templateflow(templateflow_list)
    # # check if all affine match
    if fix_affine:
        print("\tChecking if affines are different and fixing (voxel size)...")
        metadata = fix_voxel_size(metadata)
    # fist compute group fMRI mask for fMRI dice
    print("\tfMRI group mask...")
    group_fmri_mask = compute_group_fmri_mask(metadata)
    # loop through all metadata
    print("\tmean framewise displacement and dice...")
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
        source_mask = nib.load(metadata.iloc[ii]['mask'])
        # for anatomical data use template and for fMRI data use group mask
        if metadata.iloc[ii]['datatype'] == "anat":
            # loading template and resampling (assume isotropic pixel)
            # voxel_size = source_mask.affine[0, 0]
            # res = "{:02d}".format(int(voxel_size))
            # if res != "01":
            #     res = "02"
            res = "01"
            template_name = metadata.iloc[ii]['template']
            template_path = f"{os.environ['HOME']}/.cache/templateflow/tpl-{template_name}/tpl-{template_name}_res-{res}_desc-brain_mask.nii.gz"
            target_mask = nib.load(template_path)
        elif metadata.iloc[ii]['datatype'] == "func":
            target_mask = group_fmri_mask
        # compute dice score
        # plotting.view_img(template_img, cut_coords=[0, 0, 0]).save_as_html('template.html')
        dice = dice_coefficient(source_mask.get_fdata(),
                                target_mask.get_fdata())
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
    qc_metadata = qc_metadata.reset_index(drop=True)
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
                                 qc_metadata['subject'] == mask.entities['subject'])
        if 'session' in mask.entities.keys():
            row_idx = np.logical_and(
                row_idx, qc_metadata['session'] == mask.entities['session'])
        else:
            row_idx = np.logical_and(row_idx, qc_metadata['session'].isna())
        if 'task' in mask.entities.keys():
            row_idx = np.logical_and(
                row_idx, qc_metadata['task'] == mask.entities['task'])
        else:
            row_idx = np.logical_and(row_idx, qc_metadata['task'].isna())
        qc_metadata.at[row_idx, ['mask']] = mask.path
        qc_metadata.at[row_idx, ['template']] = mask.entities['space']

    return qc_metadata


def auto_quality_control(metrics, anat_dice_pass_threshold=0.99, mean_fds_pass_threshold=0.3):
    metrics['qc'] = ""
    metrics['reason'] = ""
    func_dice_pass_threshold = anat_dice_pass_threshold - 0.1
    # automatic qc based from anat/func threshold and fds_mean
    # qc_failed = metrics['fds_mean_scrubbed'] > mean_fds_pass_threshold
    # qc_failed = qc_failed | (metrics['datatype'] == "anat") & (
    #     metrics['dice'] < anat_dice_pass_threshold)
    # qc_failed = qc_failed | (metrics['datatype'] == "func") & (
    #     metrics['dice'] < func_dice_pass_threshold)
    # metrics['qc'] = ["fail" if qc_state else "pass" for qc_state in qc_failed]
    for ii in range(len(metrics)):
        dice = metrics.at[ii, 'dice']
        metrics.at[ii, 'qc'] = "pass"
        if (metrics.at[ii, 'datatype'] == "anat") & (dice < anat_dice_pass_threshold):
            metrics.at[ii, 'qc'] = "fail"
            metrics.at[ii, 'reason'] = "anat dice {} < {}".format(
                dice, anat_dice_pass_threshold)
        if (metrics.at[ii, 'datatype'] == "func"):
            fds_mean_scrubbed = metrics.at[ii, 'fds_mean_scrubbed']
            if fds_mean_scrubbed > mean_fds_pass_threshold:
                metrics.at[ii, 'qc'] = "fail"
                metrics.at[ii, 'reason'] = "fds_mean_scrubbed {}mm > {}mm; ".format(
                    fds_mean_scrubbed, mean_fds_pass_threshold)
            if (dice < func_dice_pass_threshold):
                metrics.at[ii, 'qc'] = "fail"
                metrics.at[ii, 'reason'] = metrics.at[ii, 'reason'] + \
                    "func dice {} < {}; ".format(
                        dice, func_dice_pass_threshold)

    return metrics


def descriptive_statistics(metrics):
    '''Compute the statistics for dice and fds.'''
    descriptive_stats = pd.DataFrame(
        columns=['name', 'n_samples', 'mean', 'std', 'q_0.01', 'q_0.05', 'q_0.95', 'q_0.99'])
    metric_names = ["anat_dice", "func_dice",
                    "fds_mean_raw", "fds_mean_scrubbed"]

    for metric_name in metric_names:
        split_metric_name = metric_name
        if "dice" in metric_name:
            datatype = metric_name.split("_")[0]
            split_metric_name = metric_name.split("_")[1]
        else:
            datatype = "func"
        samples = metrics[metrics['datatype']
                          == datatype][split_metric_name].to_numpy()
        desc = {'name': metric_name,
                'n_samples': [len(samples)],
                'mean': [np.mean(samples)],
                'std': [np.std(samples)],
                'q_0.01': [np.quantile(samples, q=0.01)],
                'q_0.05': [np.quantile(samples, q=0.05)],
                'q_0.95': [np.quantile(samples, q=0.95)],
                'q_0.99': [np.quantile(samples, q=0.99)]}
        descriptive_stats = descriptive_stats.append(pd.DataFrame(desc))

    return descriptive_stats


if __name__ == '__main__':
    input_dir = "/home/ltetrel/Documents/data/ccna_2019"
    metadata = get_metadata(input_dir)
    metrics = compute_qc_metrics(metadata)
    metrics = auto_quality_control(metrics)
    descriptive_stats = descriptive_statistics(metrics)
