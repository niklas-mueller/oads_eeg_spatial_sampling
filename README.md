# Spatially sampling convolutional neural network features predicts foveal and peripheral human visual processing

This repository contains the code for the paper: "Spatially sampling convolutional neural network features predicts foveal and peripheral human visual processing".

All data and results associated with this paper can be found on [OSF (https://osf.io/tu34h/?view_only=079ec9bb4734407097cb9dea11f6b71d)](https://osf.io/tu34h/?view_only=079ec9bb4734407097cb9dea11f6b71d)


### To reproduce the figures follow the below steps:

1. download the collective results from the [files tab of the above OSF repository](https://osf.io/tu34h/files/) and place them under ```results``` and ```additional_results```
2. Open ```figure_scripts/make_figures.ipynb```
3. Execute the code for each figure in succession


### To reproduce the results follow the below steps:

Main experiment

1. Download the extracted DNN feature activation map from [OSF](https://osf.io/3jmwr/?view_only=0eba22fbab084fa4808db2c86110ca22) or extract them using ```analysis/main_experiment/feature_extraction.py``` into ```dnn_features```
2. Dowload the EEG data for the main experiment from [OSF](https://osf.io/v5ktp/?view_only=c041ac1b84134b608a1349adccd869d8) into ```eeg_data/main_experiment```
3. To run the encoding models (results for figure 2) use the code in ```analysis/main_experiment/encoding_model.py```
4. To compute partial correlations (results for figure 3) use the code in ```analysis/main_experiment/compute_partial_correlation.py``` 
5. To run the iterative random sampling use the code in ```analysis/main_experiment/random_sampling.py``
6. To analyse the resulting contribution maps (results for figure 6) run the code in ```analysis/main_experiment/analyse_contribution_maps.py```
7. To run the spatially optimized (sampling) encoding model (results for figure 7) run the code in ```analysis/main_experiment/spatially_optimized_encoding_model.py```


Additional experiment

1. Download the extracted DNN feature activation map from [OSF](https://osf.io/3jmwr/?view_only=0eba22fbab084fa4808db2c86110ca22) or extract them using ```analysis/additional_experiment/feature_extraction.py``` into ```dnn_features```
2. Dowload the EEG data for the main experiment from [OSF](https://osf.io/myuj8/?view_only=bc5250407ef9463f83062eff3771d867) into ```eeg_data/additional_experiment```
3. To run the encoding models (results for figure 4+5) use the code in ```analysis/additional_experiment/encoding_model.py```
4. 