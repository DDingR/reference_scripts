# CM data collector

New feature of CarMaker 12.0. Now, we are available to use python as the simulation controller of CM.

### before use
- The `inf` need to be in the directory
- In `Config.xml`, the project directory is defined

## customization
`simtime` : simulation time [sec]  
`capture_start`: CarMaker dictionaries will be recorded from this time [sec]  
`trg_testrun`: testrun what you want to simulate
`rec_num`: simulation cases (not usable now)
`down_sample_rate`: record rate [Hz]
`var_list`: CM dictionaries will be recorded

## results
- The record log will be saved at `logs`
- The record data will be saved at `results` in `np.array` (size is [var_number, sample number])
- 