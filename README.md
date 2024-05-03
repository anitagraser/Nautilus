# Nautilus
### Official Python implementation of the [Fed]Nautilus model, both centralized and federated learning versions, proposed in the paper "On Vessel Location Forecasting and the Effect of Federated Learning”, MDM Conference, 2024.


# Installation 
In order to use Nautilus in your project, download all necessary modules in your directory of choice via pip or conda, and install their corresponding dependencies, as the following commands suggest:

```Python
conda create -n nautilus python=3.10
pip install -r requirements.txt
```


# Data Preprocessing
In order to perform data preprocessing on your AIS dataset(s), as defined in the paper, run the following script

```bash
python data-preprocessing.py [-h] --data {brest,piraeus,mt} [--min_dt MIN_DT] [--max_dt MAX_DT] [--min_speed MIN_SPEED] [--max_speed MAX_SPEED] [--min_pts MIN_PTS] [--shiptype] [--njobs NJOBS]
```

To follow the preprocessing workflow up to the specifications defined in the paper, adjust the above command as follows:

```bash
python data-preprocessing.py --data {brest,piraeus,mt} --shiptype --min_dt 10 --max_dt {1800,3600}
```

for the 30 min. and 60 min. variant, respectively.

### Reproduction attempt

```bash
python data-preprocessing.py --data brest --shiptype --min_dt 10 --max_dt 1800

FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\GraserA/data/brest-dataset\\nari_dynamic.csv'
(nautilus) PS C:\Users\GraserA\Documents\GitHub\Nautilus>
```

Get nari_dynamic.csv from https://zenodo.org/record/1167595/files/%5BP1%5D%20AIS%20Data.zip?download=1

```bash
(nautilus) PS C:\Users\GraserA\Documents\GitHub\Nautilus> python data-preprocessing.py --data brest --shiptype --min_dt 10 --max_dt 1800
[Raw Dataset] Dataset AIS Positions: 19035630
[Invalid MMSIs] Dataset AIS Positions: 18657808; Time elapsed: 14.489725112915039
Traceback (most recent call last):
  File "C:\Users\GraserA\Documents\GitHub\Nautilus\data-preprocessing.py", line 72, in <module>
    df.loc[:, f'timestamp_sec'] = df.timestamp_datetime.astype(int) // 10**9
  File "C:\Users\GraserA\AppData\Local\miniconda3\envs\nautilus\lib\site-packages\pandas\core\generic.py", line 6324, in astype
    new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
  File "C:\Users\GraserA\AppData\Local\miniconda3\envs\nautilus\lib\site-packages\pandas\core\internals\managers.py", line 451, in astype
    return self.apply(
  File "C:\Users\GraserA\AppData\Local\miniconda3\envs\nautilus\lib\site-packages\pandas\core\internals\managers.py", line 352, in apply
    applied = getattr(b, f)(**kwargs)
  File "C:\Users\GraserA\AppData\Local\miniconda3\envs\nautilus\lib\site-packages\pandas\core\internals\blocks.py", line 511, in astype
    new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
  File "C:\Users\GraserA\AppData\Local\miniconda3\envs\nautilus\lib\site-packages\pandas\core\dtypes\astype.py", line 242, in astype_array_safe
    new_values = astype_array(values, dtype, copy=copy)
  File "C:\Users\GraserA\AppData\Local\miniconda3\envs\nautilus\lib\site-packages\pandas\core\dtypes\astype.py", line 184, in astype_array
    values = values.astype(dtype, copy=copy)
  File "C:\Users\GraserA\AppData\Local\miniconda3\envs\nautilus\lib\site-packages\pandas\core\arrays\datetimes.py", line 701, in astype
    return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy)
  File "C:\Users\GraserA\AppData\Local\miniconda3\envs\nautilus\lib\site-packages\pandas\core\arrays\datetimelike.py", line 472, in astype
    raise TypeError(
TypeError: Converting from datetime64[ns] to int32 is not supported. Do obj.astype('int64').astype(dtype) instead
(nautilus) PS C:\Users\GraserA\Documents\GitHub\Nautilus>
```

## Documentation

```bash
  -h, --help            show this help message and exit
  --data                Select Dataset {brest, piraeus, mt}
  --min_dt MIN_DT       Minimum $\Delta t$ threshold (default:10 sec.)
  --max_dt MAX_DT       Maximum $\Delta t$ threshold (default:1800 sec.)
  --min_speed MIN_SPEED
                        Minimum speed threshold (stationaries; default: 1 knot)
  --max_speed MAX_SPEED
                        Maximum speed threshold (outliers; default: 50 knots)
  --min_pts MIN_PTS     Minimum points threshold for constructing a trajectory (default: 20 points)
  --shiptype            Include shiptype
  --njobs NJOBS         #CPUs (default: 200 cores)
```


# Usage (Centralized)
In order to train ```Nautilus```, run the following script:

```bash
python training-rnn-v2-indie-timesplit.py --data {brest_1800,brest_3600,piraeus_1800,piraeus_3600,mt_1800,mt_3600} [--gpuid GPUID] [--njobs NJOBS] [--crs CRS] [--bi] [--dspeed] [--dcourse] [--shiptype] [--bs BS] [--length LENGTH] [--stride STRIDE] [--patience PATIENCE] [--max_dt MAX_DT] [--skip_train]
```

For training ```Nautilus``` up to specifications defined in the paper, adjust the above command as follows:

```bash
python training-rnn-v2-indie-timesplit.py --data {brest_1800,piraeus_1800,mt_1800} --gpuid 0 --bs 1 --njobs 200 --crs {2154,2100,2100} --length 32 --stride 16 --patience 10 --shiptype --dspeed --dcourse --max_dt {1800,3600}
```


for the 30 min. and 60 min. variant, respectively.

### Replication attempt 


```bash
python training-rnn-v2-indie-timesplit.py --data brest_1800 --gpuid 0 --bs 1 --njobs 50 --crs 2154 --length 32 --stride 16 --patience 10 --shiptype --dspeed --dcourse --max_dt 1800

FileNotFoundError: [Errno 2] No such file or directory: './data/pkl/shiptype_token_lookup_v3.pkl'
```

Copied './data/shiptype_token_lookup_v3.pkl' to './data/pkl/shiptype_token_lookup_v3.pkl' 

```
FileNotFoundError: [Errno 2] No such file or directory: './data/fig/exp_study/delta_series_timeseries_split_lookahead_distribution_brest_1800_window_32_stride_16_crs_2154_dspeed_dcourse_.pdf'
```

Created missing directory data/fig/exp_study

```
(nautilus) PS C:\Users\GraserA\Documents\GitHub\Nautilus> python training-rnn-v2-indie-timesplit.py --data brest_1800 --gpuid 0 --bs 1 --njobs 50 --crs 2154 --length 32 --stride 16 --patience 10 --shiptype --dspeed --dcourse --max_dt 1800
C:\Users\GraserA\AppData\Local\miniconda3\envs\nautilus\lib\site-packages\torch\random.py:42: UserWarning: Failed to initialize NumPy: module compiled against API version 0x10 but this version of numpy is 0xf (Triggered internally at ..\torch\csrc\utils\tensor_numpy.cpp:77.)
  return default_generator.manual_seed(seed)
[Loaded] Dataset AIS Positions: 4408217
[Invalid MIDs] Dataset AIS Positions: 4408217
Train @(min(trajectories_dates[train_dates]), max(trajectories_dates[train_dates]))=(datetime.date(2015, 9, 30), datetime.date(2015, 12, 30));
Dev @(min(trajectories_dates[dev_dates]), max(trajectories_dates[dev_dates]))=(datetime.date(2015, 12, 31), datetime.date(2016, 2, 14));
Test @(min(trajectories_dates[test_dates]), max(trajectories_dates[test_dates]))=(datetime.date(2016, 2, 15), datetime.date(2016, 3, 31))
Sanity Check #1;
        trajectories.groupby([VESSEL_NAME, 'id', 'dataset_tr1_val2_test3'])['timestamp'].is_monotonic_increasing.all()=True
Scaling <function <lambda> at 0x0000022F52A79C60> to 50 CPUs
100%|████████████████████████████████████████████████████████████████████████████| 14421/14421 [02:52<00:00, 83.54it/s]
windowing_params["input_feats"]=['dlon_curr', 'dlat_curr', 'dspeed_curr', 'dcourse_curr', 'dt_curr', 'dt_next']
Scaling <function <lambda> at 0x0000022F52A79CF0> to 50 CPUs
100%|███████████████████████████████████████████████████████████████████████████| 14421/14421 [00:36<00:00, 394.07it/s]
ShipTypeVRF(
  (rnn_cell): LSTM(6, 350, batch_first=True)
  (fc): Sequential(
    (0): Sequential(
      (0): Linear(in_features=356, out_features=150, bias=True)
      (1): ReLU()
    )
    (1): Linear(in_features=150, out_features=2, bias=True)
  )
  (embedding): Embedding(13, 6)
  (dropout): Dropout(p=0.25, inplace=False)
)
device=device(type='cpu')
self.eps=0.0001
.\data\pth\brest_1800\lstm_1_350_fc_150_share_all_window_32_stride_16_crs_2154_dspeed_dcourse_shiptype_batchsize_1_patience_10__brest_1800_dataset__timeseries_split_.dropout_after_cat.sn_cml.epoch{0}.pth
Traceback (most recent call last):
  File "C:\Users\GraserA\Documents\GitHub\Nautilus\training-rnn-v2-indie-timesplit.py", line 313, in <module>
    tr.train_model(
  File "C:\Users\GraserA\Documents\GitHub\Nautilus\train.py", line 153, in train_model
    train_loss = train_step(model, device, criterion, optimizer, train_loader)
  File "C:\Users\GraserA\Documents\GitHub\Nautilus\train.py", line 79, in train_step
    for j, (xb, yb, lb, *args) in (pbar := tqdm.tqdm(enumerate(train_loader), leave=False, total=len(train_loader), dynamic_ncols=True)):
  File "C:\Users\GraserA\AppData\Local\miniconda3\envs\nautilus\lib\site-packages\tqdm\std.py", line 1195, in __iter__
    for obj in iterable:
  File "C:\Users\GraserA\AppData\Local\miniconda3\envs\nautilus\lib\site-packages\torch\utils\data\dataloader.py", line 628, in __next__
    data = self._next_data()
  File "C:\Users\GraserA\AppData\Local\miniconda3\envs\nautilus\lib\site-packages\torch\utils\data\dataloader.py", line 671, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "C:\Users\GraserA\AppData\Local\miniconda3\envs\nautilus\lib\site-packages\torch\utils\data\_utils\fetch.py", line 58, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "C:\Users\GraserA\AppData\Local\miniconda3\envs\nautilus\lib\site-packages\torch\utils\data\_utils\fetch.py", line 58, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "C:\Users\GraserA\Documents\GitHub\Nautilus\dataset.py", line 89, in __getitem__
    return torch.tensor(self.scaler.transform(self.samples[item]).astype(self.dtype)),\
RuntimeError: Could not infer dtype of numpy.float32
```

--> Moved from Windows to WSL/Ubuntu


```
RuntimeError: Parent directory ./data/pth/brest_1800 does not exist.
```

Created output dir 

```bash
(nautilus) grasera@N3DSS2206:/mnt/c/Users/GraserA/Documents/GitHub/Nautilus$ python training-rnn-v2-indie-timesplit.py --data brest_1800 --gpuid 0 --bs 1 --njobs 50 --crs 2154 --length 32 --stride 16 --patience 10 --shiptype --dspeed --dcourse --max_dt 1800
[Loaded] Dataset AIS Positions: 4408217
[Invalid MIDs] Dataset AIS Positions: 4408217
Train @(min(trajectories_dates[train_dates]), max(trajectories_dates[train_dates]))=(datetime.date(2015, 9, 30), datetime.date(2015, 12, 30));
Dev @(min(trajectories_dates[dev_dates]), max(trajectories_dates[dev_dates]))=(datetime.date(2015, 12, 31), datetime.date(2016, 2, 14));
Test @(min(trajectories_dates[test_dates]), max(trajectories_dates[test_dates]))=(datetime.date(2016, 2, 15), datetime.date(2016, 3, 31))
Sanity Check #1;
        trajectories.groupby([VESSEL_NAME, 'id', 'dataset_tr1_val2_test3'])['timestamp'].is_monotonic_increasing.all()=True
Scaling <function <lambda> at 0x7f4e82348d30> to 50 CPUs
100%|█████████████████████████████████████████████████████████████████████████████| 14421/14421 [02:26<00:00, 98.13it/s]
windowing_params["input_feats"]=['dlon_curr', 'dlat_curr', 'dspeed_curr', 'dcourse_curr', 'dt_curr', 'dt_next']
Scaling <function <lambda> at 0x7f4e82348dc0> to 50 CPUs
100%|████████████████████████████████████████████████████████████████████████████| 14421/14421 [00:29<00:00, 495.39it/s]
ShipTypeVRF(
  (rnn_cell): LSTM(6, 350, batch_first=True)
  (fc): Sequential(
    (0): Sequential(
      (0): Linear(in_features=356, out_features=150, bias=True)
      (1): ReLU()
    )
    (1): Linear(in_features=150, out_features=2, bias=True)
  )
  (embedding): Embedding(13, 6)
  (dropout): Dropout(p=0.25, inplace=False)
)
device=device(type='cuda', index=0)
self.eps=0.0001
./data/pth/brest_1800/lstm_1_350_fc_150_share_all_window_32_stride_16_crs_2154_dspeed_dcourse_shiptype_batchsize_1_patience_10__brest_1800_dataset__timeseries_split_.dropout_after_cat.sn_cml.epoch{0}.pth
Loss Decreased (inf -> 19.89650). Saving Model... Done!
Epoch #1/100 | Train Loss: 30.95741 | Validation Loss: 19.89650 | Time Elapsed: 2133.14975
Loss: 19.89650 |  Accuracy: 28.13788 | 16.53355; 562.24835; 1101.75208; 1641.74609; 2646.30444; 3307.54761 m
Loss Decreased (19.89650 -> 18.22664). Saving Model... Done!
Epoch #2/100 | Train Loss: 23.65413 | Validation Loss: 18.22664 | Time Elapsed: 2120.82069
Loss Decreased (18.22664 -> 16.36896). Saving Model... Done!
Epoch #3/100 | Train Loss: 21.82535 | Validation Loss: 16.36896 | Time Elapsed: 2140.66847
Loss Decreased (16.36896 -> 15.79784). Saving Model... Done!
Epoch #4/100 | Train Loss: 20.38762 | Validation Loss: 15.79784 | Time Elapsed: 2104.46122
Loss Increased (15.79784 -> 16.10181).
Epoch #5/100 | Train Loss: 19.44114 | Validation Loss: 16.10181 | Time Elapsed: 2132.84904
Loss Decreased (15.79784 -> 15.18527). Saving Model... Done!
Epoch #6/100 | Train Loss: 18.98068 | Validation Loss: 15.18527 | Time Elapsed: 2109.91644
Loss: 15.18527 |  Accuracy: 21.47519 | 13.47575; 398.90659; 778.09918; 1283.44019; 1679.20813; 1996.84949 m
Loss Decreased (15.18527 -> 14.95022). Saving Model... Done!
Epoch #7/100 | Train Loss: 18.49553 | Validation Loss: 14.95022 | Time Elapsed: 2132.46497
Loss Decreased (14.95022 -> 14.47501). Saving Model... Done!
Epoch #8/100 | Train Loss: 18.11352 | Validation Loss: 14.47501 | Time Elapsed: 2153.99340
Loss Decreased (14.47501 -> 13.96758). Saving Model... Done!
Epoch #9/100 | Train Loss: 17.76096 | Validation Loss: 13.96758 | Time Elapsed: 2120.81118
Loss Decreased (13.96758 -> 13.71689). Saving Model... Done!
Epoch #10/100 | Train Loss: 17.45697 | Validation Loss: 13.71689 | Time Elapsed: 2137.34794
Loss Increased (13.71689 -> 13.96497).
Epoch #11/100 | Train Loss: 17.07539 | Validation Loss: 13.96497 | Time Elapsed: 2134.06785
Loss: 13.96497 |  Accuracy: 19.74942 | 12.35323; 372.93060; 776.15125; 1115.43054; 1670.83691; 1599.05176 m
```


## Documentation
```bash
    -h, --help            Show this help message and exit
    --data                Select Dataset 
                          (Options: brest_1800, brest_3600, piraeus_1800, piraeus_3600, mt_1800, mt_3600)
    --gpuid GPUID         GPU ID
    --njobs NJOBS         #CPUs
    --crs CRS             Dataset CRS (default: 3857)
    --bi                  Use Bidirectional LSTM
    --dspeed              Use first order difference of Speed
    --dcourse             Use first order difference of Course
    --shiptype            Use AIS Shiptype
    --bs BS               Batch Size
    --length LENGTH       Rolling Window Length (default: 32)
    --stride STRIDE       Rolling Window Stride (default: 16)
    --patience PATIENCE   Patience (#Epochs) for Early Stopping (default: 10)
    --max_dt MAX_DT       Maximum $\Delta t$ threshold (default:1800 sec.)
    --skip_train          Skip training; Evaluate best model @ Test Set
```


# Usage (Federated)
In order to train ```FedNautilus```, run the ```server.py``` script in order to instantiate the aggregation script, as well as the  ```client.py``` script for as many available clients (data silos):

```bash
python server.py [-h] [--bi] [--dspeed] [--dcourse] [--shiptype] [--length LENGTH] [--stride STRIDE] [--max_dt MAX_DT] [--silos SILOS] [--fraction_fit FRACTION_FIT] [--fraction_eval FRACTION_EVAL] [--num_rounds NUM_ROUNDS] [--load_check] [--port PORT] [--mu MU]

python client.py [-h] --data {brest_1800,brest_3600,piraeus_1800,piraeus_3600,mt_1800,mt_3600} [--gpuid GPUID] [--crs CRS] [--bi] [--dspeed] [--dcourse] [--shiptype] [--bs BS] [--length LENGTH] [--stride STRIDE] [--aug] [--max_dt MAX_DT] [--load_check] [--port PORT] [--silos SILOS] [--mu MU] [--fraction_fit FRACTION_FIT] [--fraction_eval FRACTION_EVAL] [--personalize] [--global_ver GLOBAL_VER] [--num_rounds NUM_ROUNDS]
```

For training ```Nautilus``` up to specifications defined in the paper, adjust the above command as follows:

```bash
python client.py --data {brest_1800,piraeus_1800,mt_1800} --gpuid 3 --bs 1 --shiptype --crs 2100 --length 32 --stride 16 --dspeed --dcourse --port 8080 --mu 1 --fraction_fit 1 --silos 3 --max_dt {1800,3600}

python server.py --shiptype --dspeed --dcourse --length 32 --stride 16 --num_rounds 70 --silos 3 --port 8080 --mu 1 --fraction_fit 1 --max_dt {1800,3600}
```

for the 30 min. and 60 min. variant, respectively. To run personalization, run the ```client.py``` script with the same parameters and append the ```--personalization``` flag, as illustrated in the following example:

```bash
python client.py --data {brest_1800,piraeus_1800,mt_1800} --gpuid 3 --bs 1 --shiptype --crs 2100 --length 32 --stride 16 --dspeed --dcourse --port 8080 --mu 1 --fraction_fit 1 --silos 3 --max_dt {1800,3600} --personalize
```

### Replication attempt

```bash
python client.py --data brest_1800 --gpuid 3 --bs 1 --shiptype --crs 2100 --length 32 --stride 16 --dspeed --dcourse --port 8080 --mu 1 --fraction_fit 1 --silos 3 --max_dt 1800


```


## Documentation (server)
```bash
  -h, --help            show this help message and exit
  --silos SILOS         #Data Silos (default: 3)
  --fraction_fit FRACTION_FIT
                        #clients to train per round (%)
  --fraction_eval FRACTION_EVAL
                        #clients to evaluate per round (%)
  --num_rounds NUM_ROUNDS
                        #FL Rounds (default: 170)
  --load_check          Continue from Latest FL Round
  --port PORT           Server Port
  --mu MU               Proximal $\mu$
```


## Documentation (client)
```bash
  -h, --help            show this help message and exit
  --personalize         Fine-tune the global model to the local clients data
  --global_ver GLOBAL_VER
                        Version of global model to load
  --num_rounds NUM_ROUNDS
                        Number of epochs for fine-tuning
```



# On Reproducing the Experimental Study

For the sake of convenience the preprocessed versions of the open datasets used in our experimental study can be found in the directory ```./data/{brest,piraeus}-dataset/10_sec__{1800,3600}_sec/dataset_trajectories_preprocessed_with_type.fixed.csv``` (after extracting the corresponding zip files). To extract the files, use an application, such as [7-zip](https://www.7-zip.org) (Windows), [The Unarchiver](https://theunarchiver.com) (Mac), or the following terminal commands (Linux/Mac):

```bash
ls -v 10_sec__{1800,3600}_sec.z* | xargs cat > 10_sec__{1800,3600}_sec.zip.fixed
unzip 10_sec__{1800,3600}_sec.zip.fixed
```

To reproduce the experimental study, i.e., test the performance of the models in the datasets' test set, run the following script (using the same parameters/flags as the aforementioned scripts):

```bash
python model-evaluation.py --data {brest_1800,brest_3600,piraeus_1800,piraeus_3600,mt_1800,mt_3600} [--gpuid GPUID] [--crs CRS] [--bi] [--dspeed] [--dcourse] [--shiptype] [--bs BS] [--length LENGTH] [--stride STRIDE] [--aug] [--max_dt MAX_DT] [--patience PATIENCE] [--silos SILOS] [--fraction_fit FRACTION_FIT] [--fraction_eval FRACTION_EVAL] [--cml] [--fl] [--perfl] [--global_ver GLOBAL_VER] [--mu MU]
```

### Replication attempt

```bash
python model-evaluation.py --data brest_1800 --gpuid 3 --bs 1 --shiptype --crs 2100 --length 32 --stride 16 --dspeed --dcourse --mu 1 --fraction_fit 1 --silos 3 --max_dt 1800

FileNotFoundError: [Errno 2] No such file or directory: './data/pkl/exp_study/brest_1800_dataset_window_32_stride_16_crs_2100_dspeed_dcourse.traj_delta_windows.pickle'
```

## Documentation
```bash
  --cml                 Evaluate Nautilus
  --fl                  Evaluate (global)FedNautilus
  --perfl               Evaluate (per)FedNautilus
```


# Contributors
Andreas Tritsarolis; Department of Informatics, University of Piraeus

Nikos Pelekis; Department of Statistics & Insurance Science, University of Piraeus

Konstantina Bereta; Kpler

Dimitris Zissis; Department of Product & Systems Design Engineering, University of the Aegean

Yannis Theodoridis; Department of Informatics, University of Piraeus


# Citation
If you use [Fed]Nautilus in your project, we would appreciate citations to the following paper:

> Andreas Tritsarolis, Nikos Pelekis, Konstantina Bereta, Dimitris Zissis, and Yannis Theodoridis. 2024. On Vessel Location Forecasting and the Effect of Federated Learning. In Proceedings of the 25th Conference on Mobile Data Management (MDM).


# Acknowledgement
This work was supported in part by the Horizon Framework Programme of the European Union under grant agreement No. 101070279 (MobiSpaces; https://mobispaces.eu). In this work, Kpler provided the Aegean AIS dataset and the requirements of the business case.
