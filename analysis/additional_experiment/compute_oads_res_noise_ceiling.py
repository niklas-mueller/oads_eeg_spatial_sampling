import os
import mne
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # cleaning = '-AutoReject'
    
    # remove_low_acc = False
    manual = True
    mode = '_manual' if manual else ''


    eeg_dir = f"/home/nmuller/projects/fmg_projects/2024_Scholte_FMG-6506_oads_res/oads_resolution_experiment/data_SECOND_BATCH{mode}"

    All_Images_df = pd.read_csv(os.path.join("/home/nmuller/projects/fmg_projects/2024_Scholte_FMG-6506_oads_res/oads_resolution_experiment/analysis", "trigID.csv"), header = None)
    All_Images_dict = dict(zip(All_Images_df[1], All_Images_df[0]))
    nanarray = None

    train_filenames = np.load('/home/nmuller/projects/fmg_projects/2024_Scholte_FMG-6506_oads_res/oads_resolution_experiment/data_clean/base_dir_reordered/data_with_reject/train_filenames.npy')
    test_filenames = np.load('/home/nmuller/projects/fmg_projects/2024_Scholte_FMG-6506_oads_res/oads_resolution_experiment/data_clean/base_dir_reordered/data_with_reject/test_filenames.npy')

    for subject in [15, 16, 17, 18, 19]:
        
        for cond in ['center', 'peri', 'size', 'quality']:

            for cond_index in range(1, 4):
                if cond == 'quality' and cond_index == 3:
                    continue
                
                oads_erp_reps = {}

                for session in range(1, 7):
                    filename = f'sub{str(subject).zfill(3)}_ses{str(session).zfill(3)}{cond}{cond_index}-OC&CSD-epo.fif'
                    try:
                        erps = mne.read_epochs(os.path.join(eeg_dir, 'fif_files', filename), preload=True)
                    except FileNotFoundError:
                        print(f'File {filename} not found')
                        continue

                    # Find repetitions of the same image
                    for index, event_id in enumerate(erps.events[:, 2]):
                        image_name = All_Images_dict[event_id]
                        if image_name not in oads_erp_reps:
                            oads_erp_reps[image_name] = []

                        oads_erp_reps[image_name].append(erps[index].get_data())

                        if nanarray is None:
                            nanarray = np.empty(erps[index].get_data().shape)
                            # print(nanarray.shape)
                            nanarray[:] = np.nan

                # channel_names = erps.ch_names
                train_reps = {f"{image_name.split('/')[-1].split('_')[0]}.png": oads_erp_reps[image_name] for image_name in oads_erp_reps.keys() if f"{image_name.split('/')[-1].split('_')[0]}.png" in train_filenames}
                test_reps = {f"{image_name.split('/')[-1].split('_')[0]}.png": oads_erp_reps[image_name] for image_name in oads_erp_reps.keys() if f"{image_name.split('/')[-1].split('_')[0]}.png" in test_filenames}

                test_data_reps = np.zeros((10, len(test_filenames), 68, 513))

                for i, image_name in enumerate(test_filenames):
                    for rep_index in range(10):
                        if image_name in test_reps.keys() and rep_index < len(test_reps[image_name]):
                            test_data_reps[rep_index, i, :, :] = test_reps[image_name][rep_index]
                        else:
                            test_data_reps[rep_index, i, :, :] = nanarray


                noise_ceiling = {}

                for rep in range(5):
                    split1 = np.random.choice(np.arange(10), 5, replace=False)
                    split2 = np.array([i for i in np.arange(10) if i not in split1])

                    noise_ceiling[rep] = {'splits': (split1, split2)}
                    for channel in range(68):
                        noise_ceiling[rep][channel] = {}
                        for timepoint in range(513):
                            set1 = np.nanmean(test_data_reps[split1, :, channel, timepoint], axis=0)
                            set2 = np.nanmean(test_data_reps[split2, :, channel, timepoint], axis=0)
                            full = np.nanmean(test_data_reps[:, :, channel, timepoint], axis=0)

                            nanmask = np.logical_and(np.logical_and(~np.isnan(set1), ~np.isnan(set2)), ~np.isnan(full))
                            set1 = set1[nanmask]
                            set2 = set2[nanmask]
                            full = full[nanmask]
                            
                            upper = np.corrcoef(set1, full)[0, 1]
                            lower = np.corrcoef(set1, set2)[0, 1]

                            noise_ceiling[rep][channel][timepoint] = {'upper': upper, 'lower': lower}

                upper_noise_ceiling = np.zeros((68, 513))
                lower_noise_ceiling = np.zeros((68, 513))

                for channel in range(68):
                    for timepoint in range(513):
                        upper_noise_ceiling[channel, timepoint] = np.mean([noise_ceiling[rep][channel][timepoint]['upper'] for rep in range(5)])
                        lower_noise_ceiling[channel, timepoint] = np.mean([noise_ceiling[rep][channel][timepoint]['lower'] for rep in range(5)])

                np.save(os.path.join(eeg_dir, 'dataclean', 'preprocessed_for_screening', 'data_with_reject', f'sub{str(subject).zfill(3)}', f'sub{str(subject).zfill(3)}_{cond}{cond_index}_upper_noise_ceiling.npy'), upper_noise_ceiling)
                np.save(os.path.join(eeg_dir, 'dataclean', 'preprocessed_for_screening', 'data_with_reject', f'sub{str(subject).zfill(3)}', f'sub{str(subject).zfill(3)}_{cond}{cond_index}_lower_noise_ceiling.npy'), lower_noise_ceiling)