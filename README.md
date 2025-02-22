## A SMOLLM2 Model in DeepSeek Model architecture

### Salient Features
- MultiHead Latent Attention used to reduce the parameters count
- Mixture of Experts is used to train the model in with different experts

### Model Parameters

- Vocab Size : 49512
- Sequennce Length : 256
- Batch Size : 4
- Accumulation Steps : 8
- Dimension : 576
- Intermediate Size : 1536
- No Of Layers : 30
- No of Heads : 9
- Compres Ratio : 3
- No Of Experts : 8
- No Of Shared Experts : 1
- Top-K Experts : 2

### [Training Logs](training-logs.out)
```
    wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Tracking run with wandb version 0.19.4
wandb: W&B syncing is set to `offline` in this directory.  
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.

Model Statistics:
Total Parameters: 688,702,098
Model Size: 2627.19 MB
Device: mps
Batch Size: 4
Accumulation Steps: 8
Sequence Length: 256
Learning Rate: 0.0001
--------------------------------------------------

Epoch 1/5
Step 0, Loss: 46.5964, Scaled Loss: 5.8246, LR: 4.00e-06, Accumulation Step: 1/8
Step 10, Loss: 46.2109, Scaled Loss: 5.7764, LR: 4.00e-06, Accumulation Step: 3/8
Step 20, Loss: 45.8798, Scaled Loss: 5.7350, LR: 4.00e-06, Accumulation Step: 5/8
Step 30, Loss: 45.1243, Scaled Loss: 5.6405, LR: 4.00e-06, Accumulation Step: 7/8
Step 40, Loss: 44.5579, Scaled Loss: 5.5697, LR: 4.00e-06, Accumulation Step: 1/8
Step 50, Loss: 44.0808, Scaled Loss: 5.5101, LR: 4.00e-06, Accumulation Step: 3/8
Step 60, Loss: 43.4014, Scaled Loss: 5.4252, LR: 4.00e-06, Accumulation Step: 5/8
Step 70, Loss: 43.2992, Scaled Loss: 5.4124, LR: 4.00e-06, Accumulation Step: 7/8
Step 80, Loss: 42.5501, Scaled Loss: 5.3188, LR: 4.01e-06, Accumulation Step: 1/8
Step 90, Loss: 41.7548, Scaled Loss: 5.2194, LR: 4.01e-06, Accumulation Step: 3/8
Step 100, Loss: 41.6344, Scaled Loss: 5.2043, LR: 4.01e-06, Accumulation Step: 5/8
Step 110, Loss: 41.3968, Scaled Loss: 5.1746, LR: 4.01e-06, Accumulation Step: 7/8
Step 120, Loss: 40.5086, Scaled Loss: 5.0636, LR: 4.01e-06, Accumulation Step: 1/8
Step 130, Loss: 39.8340, Scaled Loss: 4.9792, LR: 4.02e-06, Accumulation Step: 3/8
Step 140, Loss: 39.8251, Scaled Loss: 4.9781, LR: 4.02e-06, Accumulation Step: 5/8
Step 150, Loss: 38.9670, Scaled Loss: 4.8709, LR: 4.02e-06, Accumulation Step: 7/8
Step 160, Loss: 37.3656, Scaled Loss: 4.6707, LR: 4.02e-06, Accumulation Step: 1/8
Step 170, Loss: 36.7511, Scaled Loss: 4.5939, LR: 4.03e-06, Accumulation Step: 3/8
Step 180, Loss: 37.2103, Scaled Loss: 4.6513, LR: 4.03e-06, Accumulation Step: 5/8
Step 190, Loss: 35.7671, Scaled Loss: 4.4709, LR: 4.03e-06, Accumulation Step: 7/8
Step 200, Loss: 37.5680, Scaled Loss: 4.6960, LR: 4.04e-06, Accumulation Step: 1/8
Step 210, Loss: 34.6407, Scaled Loss: 4.3301, LR: 4.04e-06, Accumulation Step: 3/8
Step 220, Loss: 35.0841, Scaled Loss: 4.3855, LR: 4.04e-06, Accumulation Step: 5/8
Step 230, Loss: 34.0282, Scaled Loss: 4.2535, LR: 4.05e-06, Accumulation Step: 7/8
/opt/anaconda3/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Tracking run with wandb version 0.19.4
wandb: W&B syncing is set to `offline` in this directory.  
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.

Model Statistics:
Total Parameters: 688,702,098
Model Size: 2627.19 MB
Device: mps
Batch Size: 4
Accumulation Steps: 8
Sequence Length: 256
Learning Rate: 0.0001
--------------------------------------------------

Epoch 1/5
Step 200, Loss: 36.3152, Scaled Loss: 4.5394, LR: 4.04e-06, Accumulation Step: 1/8
'HTTPSConnectionPool(host='cdn-lfs-us-1.hf.co', port=443): Read timed out.' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00000-of-00104.parquet
Retrying in 1s [Retry 1/5].
'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: 9ba332f0-f751-423d-8b18-064ec9da309f)')' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00000-of-00104.parquet
Retrying in 2s [Retry 2/5].
Step 210, Loss: 35.7971, Scaled Loss: 4.4746, LR: 4.04e-06, Accumulation Step: 3/8
'HTTPSConnectionPool(host='cdn-lfs-us-1.hf.co', port=443): Read timed out.' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00000-of-00104.parquet
Retrying in 1s [Retry 1/5].
'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: 060b426d-1d05-4771-a45c-3c2926ee0af9)')' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00000-of-00104.parquet
Retrying in 2s [Retry 2/5].
Step 220, Loss: 35.4846, Scaled Loss: 4.4356, LR: 4.04e-06, Accumulation Step: 5/8
Step 230, Loss: 33.0212, Scaled Loss: 4.1277, LR: 4.05e-06, Accumulation Step: 7/8
Step 240, Loss: 33.7399, Scaled Loss: 4.2175, LR: 4.05e-06, Accumulation Step: 1/8
Step 250, Loss: 33.2980, Scaled Loss: 4.1622, LR: 4.06e-06, Accumulation Step: 3/8
Step 260, Loss: 31.8024, Scaled Loss: 3.9753, LR: 4.06e-06, Accumulation Step: 5/8
Step 270, Loss: 32.5614, Scaled Loss: 4.0702, LR: 4.06e-06, Accumulation Step: 7/8
Step 280, Loss: 32.0912, Scaled Loss: 4.0114, LR: 4.07e-06, Accumulation Step: 1/8
Step 290, Loss: 30.2997, Scaled Loss: 3.7875, LR: 4.08e-06, Accumulation Step: 3/8
Step 300, Loss: 29.8115, Scaled Loss: 3.7264, LR: 4.08e-06, Accumulation Step: 5/8
Step 310, Loss: 29.5997, Scaled Loss: 3.7000, LR: 4.09e-06, Accumulation Step: 7/8
Step 320, Loss: 28.4398, Scaled Loss: 3.5550, LR: 4.09e-06, Accumulation Step: 1/8
Step 330, Loss: 27.6401, Scaled Loss: 3.4550, LR: 4.10e-06, Accumulation Step: 3/8
Step 340, Loss: 27.8282, Scaled Loss: 3.4785, LR: 4.10e-06, Accumulation Step: 5/8
Step 350, Loss: 26.7542, Scaled Loss: 3.3443, LR: 4.11e-06, Accumulation Step: 7/8
Step 360, Loss: 24.7137, Scaled Loss: 3.0892, LR: 4.12e-06, Accumulation Step: 1/8
Step 370, Loss: 24.1091, Scaled Loss: 3.0136, LR: 4.13e-06, Accumulation Step: 3/8
Step 380, Loss: 24.6299, Scaled Loss: 3.0787, LR: 4.13e-06, Accumulation Step: 5/8
Step 390, Loss: 23.2088, Scaled Loss: 2.9011, LR: 4.14e-06, Accumulation Step: 7/8
Step 400, Loss: 23.8948, Scaled Loss: 2.9869, LR: 4.15e-06, Accumulation Step: 1/8
Step 410, Loss: 22.2113, Scaled Loss: 2.7764, LR: 4.15e-06, Accumulation Step: 3/8
Step 420, Loss: 21.6723, Scaled Loss: 2.7090, LR: 4.16e-06, Accumulation Step: 5/8
Step 430, Loss: 20.4360, Scaled Loss: 2.5545, LR: 4.17e-06, Accumulation Step: 7/8
Step 440, Loss: 19.4344, Scaled Loss: 2.4293, LR: 4.18e-06, Accumulation Step: 1/8
Step 450, Loss: 19.5865, Scaled Loss: 2.4483, LR: 4.19e-06, Accumulation Step: 3/8
Step 460, Loss: 19.5021, Scaled Loss: 2.4378, LR: 4.19e-06, Accumulation Step: 5/8
Step 470, Loss: 18.2499, Scaled Loss: 2.2812, LR: 4.20e-06, Accumulation Step: 7/8
Step 480, Loss: 18.5214, Scaled Loss: 2.3152, LR: 4.21e-06, Accumulation Step: 1/8
Step 490, Loss: 16.7689, Scaled Loss: 2.0961, LR: 4.22e-06, Accumulation Step: 3/8
Step 500, Loss: 17.3181, Scaled Loss: 2.1648, LR: 4.23e-06, Accumulation Step: 5/8

=== Generating Sample Texts ===

Prompt: The future of artificial intelligence
Temperature: 1.0
Generated: The future of artificial intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence

=== End of Samples ===

Step 510, Loss: 16.6552, Scaled Loss: 2.0819, LR: 4.24e-06, Accumulation Step: 7/8
Step 520, Loss: 15.1074, Scaled Loss: 1.8884, LR: 4.25e-06, Accumulation Step: 1/8
Step 530, Loss: 15.1585, Scaled Loss: 1.8948, LR: 4.26e-06, Accumulation Step: 3/8
Step 540, Loss: 14.8117, Scaled Loss: 1.8515, LR: 4.27e-06, Accumulation Step: 5/8
Step 550, Loss: 14.7651, Scaled Loss: 1.8456, LR: 4.27e-06, Accumulation Step: 7/8
Step 560, Loss: 14.0762, Scaled Loss: 1.7595, LR: 4.29e-06, Accumulation Step: 1/8
Step 570, Loss: 13.7881, Scaled Loss: 1.7235, LR: 4.30e-06, Accumulation Step: 3/8
Step 580, Loss: 13.2650, Scaled Loss: 1.6581, LR: 4.31e-06, Accumulation Step: 5/8
Step 590, Loss: 12.9628, Scaled Loss: 1.6203, LR: 4.32e-06, Accumulation Step: 7/8
Step 600, Loss: 12.7956, Scaled Loss: 1.5994, LR: 4.33e-06, Accumulation Step: 1/8
Step 610, Loss: 12.2467, Scaled Loss: 1.5308, LR: 4.34e-06, Accumulation Step: 3/8
Step 620, Loss: 12.3845, Scaled Loss: 1.5481, LR: 4.35e-06, Accumulation Step: 5/8
Step 630, Loss: 12.5469, Scaled Loss: 1.5684, LR: 4.36e-06, Accumulation Step: 7/8
Step 640, Loss: 11.8281, Scaled Loss: 1.4785, LR: 4.38e-06, Accumulation Step: 1/8
Step 650, Loss: 11.6546, Scaled Loss: 1.4568, LR: 4.39e-06, Accumulation Step: 3/8
Step 660, Loss: 11.1913, Scaled Loss: 1.3989, LR: 4.40e-06, Accumulation Step: 5/8
Step 670, Loss: 11.3497, Scaled Loss: 1.4187, LR: 4.41e-06, Accumulation Step: 7/8
Step 680, Loss: 11.4209, Scaled Loss: 1.4276, LR: 4.43e-06, Accumulation Step: 1/8
Step 690, Loss: 10.8761, Scaled Loss: 1.3595, LR: 4.44e-06, Accumulation Step: 3/8
Step 700, Loss: 10.8264, Scaled Loss: 1.3533, LR: 4.45e-06, Accumulation Step: 5/8
Step 710, Loss: 11.0153, Scaled Loss: 1.3769, LR: 4.46e-06, Accumulation Step: 7/8
Step 720, Loss: 10.6460, Scaled Loss: 1.3308, LR: 4.48e-06, Accumulation Step: 1/8
Step 730, Loss: 10.7354, Scaled Loss: 1.3419, LR: 4.49e-06, Accumulation Step: 3/8
Step 740, Loss: 10.6070, Scaled Loss: 1.3259, LR: 4.50e-06, Accumulation Step: 5/8
Step 750, Loss: 10.2873, Scaled Loss: 1.2859, LR: 4.51e-06, Accumulation Step: 7/8
Step 760, Loss: 10.2956, Scaled Loss: 1.2870, LR: 4.53e-06, Accumulation Step: 1/8
Step 770, Loss: 10.2761, Scaled Loss: 1.2845, LR: 4.55e-06, Accumulation Step: 3/8
Step 780, Loss: 10.1994, Scaled Loss: 1.2749, LR: 4.56e-06, Accumulation Step: 5/8
Step 790, Loss: 10.0298, Scaled Loss: 1.2537, LR: 4.57e-06, Accumulation Step: 7/8
Step 800, Loss: 9.9877, Scaled Loss: 1.2485, LR: 4.59e-06, Accumulation Step: 1/8
Step 810, Loss: 9.9383, Scaled Loss: 1.2423, LR: 4.60e-06, Accumulation Step: 3/8
Step 820, Loss: 9.7434, Scaled Loss: 1.2179, LR: 4.62e-06, Accumulation Step: 5/8
Step 830, Loss: 9.9706, Scaled Loss: 1.2463, LR: 4.63e-06, Accumulation Step: 7/8
Step 840, Loss: 9.9884, Scaled Loss: 1.2486, LR: 4.65e-06, Accumulation Step: 1/8
Step 850, Loss: 9.7553, Scaled Loss: 1.2194, LR: 4.66e-06, Accumulation Step: 3/8
Step 860, Loss: 9.7166, Scaled Loss: 1.2146, LR: 4.68e-06, Accumulation Step: 5/8
Step 870, Loss: 9.8442, Scaled Loss: 1.2305, LR: 4.69e-06, Accumulation Step: 7/8
Step 880, Loss: 9.5858, Scaled Loss: 1.1982, LR: 4.72e-06, Accumulation Step: 1/8
Step 890, Loss: 9.5319, Scaled Loss: 1.1915, LR: 4.73e-06, Accumulation Step: 3/8
Step 900, Loss: 9.4747, Scaled Loss: 1.1843, LR: 4.74e-06, Accumulation Step: 5/8
Step 910, Loss: 9.3296, Scaled Loss: 1.1662, LR: 4.75e-06, Accumulation Step: 7/8
Step 920, Loss: 9.3859, Scaled Loss: 1.1732, LR: 4.78e-06, Accumulation Step: 1/8
Step 930, Loss: 9.3954, Scaled Loss: 1.1744, LR: 4.80e-06, Accumulation Step: 3/8
Step 940, Loss: 9.4749, Scaled Loss: 1.1844, LR: 4.81e-06, Accumulation Step: 5/8
Step 950, Loss: 9.1946, Scaled Loss: 1.1493, LR: 4.82e-06, Accumulation Step: 7/8
Step 960, Loss: 9.3090, Scaled Loss: 1.1636, LR: 4.85e-06, Accumulation Step: 1/8
Step 970, Loss: 9.3006, Scaled Loss: 1.1626, LR: 4.87e-06, Accumulation Step: 3/8
Step 980, Loss: 9.3245, Scaled Loss: 1.1656, LR: 4.88e-06, Accumulation Step: 5/8
Step 990, Loss: 9.2699, Scaled Loss: 1.1587, LR: 4.89e-06, Accumulation Step: 7/8
Step 1000, Loss: 9.0943, Scaled Loss: 1.1368, LR: 4.92e-06, Accumulation Step: 1/8

=== Generating Sample Texts ===

Prompt: The future of artificial intelligence
Temperature: 1.0
Generated: The future of artificial intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence

=== End of Samples ===

Step 1010, Loss: 8.8786, Scaled Loss: 1.1098, LR: 4.94e-06, Accumulation Step: 3/8
Step 1020, Loss: 8.9293, Scaled Loss: 1.1162, LR: 4.95e-06, Accumulation Step: 5/8
Step 1030, Loss: 9.0853, Scaled Loss: 1.1357, LR: 4.97e-06, Accumulation Step: 7/8
Step 1040, Loss: 8.8944, Scaled Loss: 1.1118, LR: 5.00e-06, Accumulation Step: 1/8
Step 1050, Loss: 8.9479, Scaled Loss: 1.1185, LR: 5.01e-06, Accumulation Step: 3/8
Step 1060, Loss: 9.0711, Scaled Loss: 1.1339, LR: 5.03e-06, Accumulation Step: 5/8
Step 1070, Loss: 8.9987, Scaled Loss: 1.1248, LR: 5.04e-06, Accumulation Step: 7/8
Step 1080, Loss: 8.8940, Scaled Loss: 1.1117, LR: 5.08e-06, Accumulation Step: 1/8
Step 1090, Loss: 8.7716, Scaled Loss: 1.0965, LR: 5.09e-06, Accumulation Step: 3/8
Step 1100, Loss: 8.7448, Scaled Loss: 1.0931, LR: 5.11e-06, Accumulation Step: 5/8
Step 1110, Loss: 8.4661, Scaled Loss: 1.0583, LR: 5.12e-06, Accumulation Step: 7/8
Step 1120, Loss: 8.5002, Scaled Loss: 1.0625, LR: 5.16e-06, Accumulation Step: 1/8
Step 1130, Loss: 8.5066, Scaled Loss: 1.0633, LR: 5.17e-06, Accumulation Step: 3/8
Step 1140, Loss: 8.5970, Scaled Loss: 1.0746, LR: 5.19e-06, Accumulation Step: 5/8
Step 1150, Loss: 8.6137, Scaled Loss: 1.0767, LR: 5.21e-06, Accumulation Step: 7/8
Step 1160, Loss: 8.6846, Scaled Loss: 1.0856, LR: 5.24e-06, Accumulation Step: 1/8
Step 1170, Loss: 8.5115, Scaled Loss: 1.0639, LR: 5.26e-06, Accumulation Step: 3/8
Step 1180, Loss: 8.5257, Scaled Loss: 1.0657, LR: 5.28e-06, Accumulation Step: 5/8
Step 1190, Loss: 8.6144, Scaled Loss: 1.0768, LR: 5.29e-06, Accumulation Step: 7/8
Step 1200, Loss: 8.6122, Scaled Loss: 1.0765, LR: 5.33e-06, Accumulation Step: 1/8
Step 1210, Loss: 8.3114, Scaled Loss: 1.0389, LR: 5.35e-06, Accumulation Step: 3/8
Step 1220, Loss: 8.4680, Scaled Loss: 1.0585, LR: 5.36e-06, Accumulation Step: 5/8
Step 1230, Loss: 8.5030, Scaled Loss: 1.0629, LR: 5.38e-06, Accumulation Step: 7/8
Step 1240, Loss: 8.5923, Scaled Loss: 1.0740, LR: 5.42e-06, Accumulation Step: 1/8
Step 1250, Loss: 8.1714, Scaled Loss: 1.0214, LR: 5.44e-06, Accumulation Step: 3/8
Step 1260, Loss: 8.2259, Scaled Loss: 1.0282, LR: 5.45e-06, Accumulation Step: 5/8
Step 1270, Loss: 8.5477, Scaled Loss: 1.0685, LR: 5.47e-06, Accumulation Step: 7/8
Step 1280, Loss: 8.4335, Scaled Loss: 1.0542, LR: 5.51e-06, Accumulation Step: 1/8
Step 1290, Loss: 8.2930, Scaled Loss: 1.0366, LR: 5.53e-06, Accumulation Step: 3/8
Step 1300, Loss: 8.5574, Scaled Loss: 1.0697, LR: 5.55e-06, Accumulation Step: 5/8
Step 1310, Loss: 8.3463, Scaled Loss: 1.0433, LR: 5.57e-06, Accumulation Step: 7/8
Step 1320, Loss: 8.3900, Scaled Loss: 1.0487, LR: 5.60e-06, Accumulation Step: 1/8
Step 1330, Loss: 8.3620, Scaled Loss: 1.0453, LR: 5.62e-06, Accumulation Step: 3/8
Step 1340, Loss: 8.2085, Scaled Loss: 1.0261, LR: 5.64e-06, Accumulation Step: 5/8
Step 1350, Loss: 8.2924, Scaled Loss: 1.0366, LR: 5.66e-06, Accumulation Step: 7/8
Step 1360, Loss: 8.1987, Scaled Loss: 1.0248, LR: 5.70e-06, Accumulation Step: 1/8
Step 1370, Loss: 8.2768, Scaled Loss: 1.0346, LR: 5.72e-06, Accumulation Step: 3/8
Step 1380, Loss: 8.1177, Scaled Loss: 1.0147, LR: 5.74e-06, Accumulation Step: 5/8
Step 1390, Loss: 8.2100, Scaled Loss: 1.0262, LR: 5.76e-06, Accumulation Step: 7/8
Step 1400, Loss: 8.1825, Scaled Loss: 1.0228, LR: 5.80e-06, Accumulation Step: 1/8
Step 1410, Loss: 8.1651, Scaled Loss: 1.0206, LR: 5.82e-06, Accumulation Step: 3/8
Step 1420, Loss: 7.9277, Scaled Loss: 0.9910, LR: 5.85e-06, Accumulation Step: 5/8
Step 1430, Loss: 8.1201, Scaled Loss: 1.0150, LR: 5.87e-06, Accumulation Step: 7/8
Step 1440, Loss: 8.0318, Scaled Loss: 1.0040, LR: 5.91e-06, Accumulation Step: 1/8
Step 1450, Loss: 8.1660, Scaled Loss: 1.0208, LR: 5.93e-06, Accumulation Step: 3/8
Step 1460, Loss: 8.1825, Scaled Loss: 1.0228, LR: 5.95e-06, Accumulation Step: 5/8
Step 1470, Loss: 7.8798, Scaled Loss: 0.9850, LR: 5.97e-06, Accumulation Step: 7/8
Step 1480, Loss: 8.0448, Scaled Loss: 1.0056, LR: 6.01e-06, Accumulation Step: 1/8
Step 1490, Loss: 8.0595, Scaled Loss: 1.0074, LR: 6.04e-06, Accumulation Step: 3/8
Step 1500, Loss: 7.9856, Scaled Loss: 0.9982, LR: 6.06e-06, Accumulation Step: 5/8

=== Generating Sample Texts ===

Prompt: The future of artificial intelligence
Temperature: 1.0
Generated: The future of artificial intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence intelligence

=== End of Samples ===

Step 1510, Loss: 8.0138, Scaled Loss: 1.0017, LR: 6.08e-06, Accumulation Step: 7/8
Step 1520, Loss: 7.9129, Scaled Loss: 0.9891, LR: 6.12e-06, Accumulation Step: 1/8
Step 1530, Loss: 7.9691, Scaled Loss: 0.9961, LR: 6.15e-06, Accumulation Step: 3/8
Step 1540, Loss: 8.0894, Scaled Loss: 1.0112, LR: 6.17e-06, Accumulation Step: 5/8
Step 1550, Loss: 7.6740, Scaled Loss: 0.9593, LR: 6.19e-06, Accumulation Step: 7/8
Step 1560, Loss: 7.8765, Scaled Loss: 0.9846, LR: 6.24e-06, Accumulation Step: 1/8
Step 1570, Loss: 7.9838, Scaled Loss: 0.9980, LR: 6.26e-06, Accumulation Step: 3/8
Step 1580, Loss: 7.7818, Scaled Loss: 0.9727, LR: 6.28e-06, Accumulation Step: 5/8
Step 1590, Loss: 7.9278, Scaled Loss: 0.9910, LR: 6.31e-06, Accumulation Step: 7/8
Step 1600, Loss: 7.8628, Scaled Loss: 0.9828, LR: 6.35e-06, Accumulation Step: 1/8
Step 1610, Loss: 7.9802, Scaled Loss: 0.9975, LR: 6.37e-06, Accumulation Step: 3/8
/opt/anaconda3/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Tracking run with wandb version 0.19.4
wandb: W&B syncing is set to `offline` in this directory.  
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.

Model Statistics:
Total Parameters: 688,702,098
Model Size: 2627.19 MB
Device: mps
Batch Size: 4
Accumulation Steps: 8
Sequence Length: 256
Learning Rate: 0.0001
--------------------------------------------------

Epoch 1/5
'HTTPSConnectionPool(host='cdn-lfs-us-1.hf.co', port=443): Read timed out.' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00000-of-00104.parquet
Retrying in 1s [Retry 1/5].
'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: 94472529-bf40-4a6b-a009-67b6222dc1a2)')' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00000-of-00104.parquet
Retrying in 2s [Retry 2/5].
Step 1600, Loss: 7.9203, Scaled Loss: 0.9900, LR: 6.35e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 08:29:41 
Step 1610, Loss: 7.8349, Scaled Loss: 0.9794, LR: 6.37e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 08:42:08 
Step 1620, Loss: 7.7794, Scaled Loss: 0.9724, LR: 6.40e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 08:55:42 
Step 1630, Loss: 7.8132, Scaled Loss: 0.9766, LR: 6.42e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 09:11:28 
Step 1640, Loss: 8.0310, Scaled Loss: 1.0039, LR: 6.47e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 09:27:13 
Step 1650, Loss: 7.7502, Scaled Loss: 0.9688, LR: 6.49e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 09:41:15 
Step 1660, Loss: 7.8374, Scaled Loss: 0.9797, LR: 6.52e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 09:56:34 
Step 1670, Loss: 7.6234, Scaled Loss: 0.9529, LR: 6.54e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 10:11:29 
Step 1680, Loss: 7.7430, Scaled Loss: 0.9679, LR: 6.59e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 10:26:25 
Step 1690, Loss: 7.7852, Scaled Loss: 0.9731, LR: 6.61e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 10:40:08 
Step 1700, Loss: 7.7252, Scaled Loss: 0.9657, LR: 6.64e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 10:54:28 
Step 1710, Loss: 7.8001, Scaled Loss: 0.9750, LR: 6.66e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 11:10:18 
Step 1720, Loss: 7.7555, Scaled Loss: 0.9694, LR: 6.71e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 11:25:23 
Step 1730, Loss: 7.6272, Scaled Loss: 0.9534, LR: 6.74e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 11:39:07 
Step 1740, Loss: 7.6209, Scaled Loss: 0.9526, LR: 6.76e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 11:52:37 
Step 1750, Loss: 7.6990, Scaled Loss: 0.9624, LR: 6.79e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 12:06:51 
Step 1760, Loss: 7.7516, Scaled Loss: 0.9689, LR: 6.84e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 12:21:39 
Step 1770, Loss: 7.6859, Scaled Loss: 0.9607, LR: 6.87e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 12:35:57 
Step 1780, Loss: 7.5924, Scaled Loss: 0.9491, LR: 6.89e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 12:49:14 
Step 1790, Loss: 7.7331, Scaled Loss: 0.9666, LR: 6.92e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 13:04:27 
Step 1800, Loss: 7.6490, Scaled Loss: 0.9561, LR: 6.97e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 13:19:09 
Step 1810, Loss: 7.6889, Scaled Loss: 0.9611, LR: 7.00e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 13:33:57 
Step 1820, Loss: 7.7353, Scaled Loss: 0.9669, LR: 7.02e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 13:49:27 
Step 1830, Loss: 7.4635, Scaled Loss: 0.9329, LR: 7.05e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 14:03:51 
Step 1840, Loss: 7.5079, Scaled Loss: 0.9385, LR: 7.10e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 14:19:31 
Step 1850, Loss: 7.5361, Scaled Loss: 0.9420, LR: 7.13e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 14:35:49 
Step 1860, Loss: 7.5248, Scaled Loss: 0.9406, LR: 7.16e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 14:53:19 
Step 1870, Loss: 7.3755, Scaled Loss: 0.9219, LR: 7.18e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 15:08:52 
Step 1880, Loss: 7.4584, Scaled Loss: 0.9323, LR: 7.24e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 15:25:06 
Step 1890, Loss: 7.6244, Scaled Loss: 0.9530, LR: 7.26e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 15:39:24 
Step 1900, Loss: 7.7253, Scaled Loss: 0.9657, LR: 7.29e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 15:54:38 
Step 1910, Loss: 7.5962, Scaled Loss: 0.9495, LR: 7.32e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 16:10:36 
Step 1920, Loss: 7.4219, Scaled Loss: 0.9277, LR: 7.37e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 16:26:47 
Step 1930, Loss: 7.2995, Scaled Loss: 0.9124, LR: 7.40e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 16:42:21 
Step 1940, Loss: 7.3107, Scaled Loss: 0.9138, LR: 7.43e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 16:56:42 
Step 1950, Loss: 7.3813, Scaled Loss: 0.9227, LR: 7.46e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 17:10:48 
Step 1960, Loss: 7.5344, Scaled Loss: 0.9418, LR: 7.51e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 17:25:47 
Step 1970, Loss: 7.5382, Scaled Loss: 0.9423, LR: 7.54e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 17:40:02 
Step 1980, Loss: 7.2270, Scaled Loss: 0.9034, LR: 7.57e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 17:52:47 
Step 1990, Loss: 7.3467, Scaled Loss: 0.9183, LR: 7.60e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 18:06:08 
Step 2000, Loss: 7.4562, Scaled Loss: 0.9320, LR: 7.66e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 18:19:41 

=== Generating Sample Texts ===

Prompt: The future of artificial intelligence is lies in agentic AI. 
Temperature: 1.0
Generated: The future of artificial intelligence is lies in agentic AI.             judjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjud

Prompt: No matter how athletic a supermarket checkout clerk is, they’re not likely to scan
Temperature: 1.0
Generated: No matter how athletic a supermarket checkout clerk is, they’re not likely to scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan

Prompt: But for many jobs that primarily involve applying knowledge or processing information
Temperature: 1.0
Generated: But for many jobs that primarily involve applying knowledge or processing information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information

=== End of Samples ===

Step 2010, Loss: 7.4070, Scaled Loss: 0.9259, LR: 7.69e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 18:44:01 
Step 2020, Loss: 7.3815, Scaled Loss: 0.9227, LR: 7.72e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 18:58:14 
Step 2030, Loss: 7.5340, Scaled Loss: 0.9417, LR: 7.74e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 19:11:58 
Step 2040, Loss: 7.2198, Scaled Loss: 0.9025, LR: 7.80e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 19:26:12 
Step 2050, Loss: 7.3675, Scaled Loss: 0.9209, LR: 7.83e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 19:38:32 
Step 2060, Loss: 7.1443, Scaled Loss: 0.8930, LR: 7.86e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 19:52:35 
Step 2070, Loss: 7.3825, Scaled Loss: 0.9228, LR: 7.89e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 20:06:29 
Step 2080, Loss: 7.4475, Scaled Loss: 0.9309, LR: 7.95e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 20:21:20 
Step 2090, Loss: 7.3864, Scaled Loss: 0.9233, LR: 7.98e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 20:33:57 
Step 2100, Loss: 7.1531, Scaled Loss: 0.8941, LR: 8.01e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 20:47:41 
Step 2110, Loss: 7.4436, Scaled Loss: 0.9304, LR: 8.04e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 21:04:39 
Step 2120, Loss: 7.3098, Scaled Loss: 0.9137, LR: 8.10e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 21:18:05 
Step 2130, Loss: 7.1971, Scaled Loss: 0.8996, LR: 8.13e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 21:32:44 
Step 2140, Loss: 7.2047, Scaled Loss: 0.9006, LR: 8.16e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 21:47:18 
Step 2150, Loss: 7.2429, Scaled Loss: 0.9054, LR: 8.19e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 22:01:34 
Step 2160, Loss: 7.1442, Scaled Loss: 0.8930, LR: 8.26e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 22:16:21 
Step 2170, Loss: 7.1748, Scaled Loss: 0.8968, LR: 8.29e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 22:31:08 
Step 2180, Loss: 7.2298, Scaled Loss: 0.9037, LR: 8.32e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 22:44:05 
Step 2190, Loss: 7.1344, Scaled Loss: 0.8918, LR: 8.35e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 22:57:32 
Step 2200, Loss: 7.2693, Scaled Loss: 0.9087, LR: 8.41e-06, Accumulation Step: 1/8, Current Time: 2025-02-09 23:12:03 
Step 2210, Loss: 7.1063, Scaled Loss: 0.8883, LR: 8.45e-06, Accumulation Step: 3/8, Current Time: 2025-02-09 23:26:21 
Step 2220, Loss: 7.3419, Scaled Loss: 0.9177, LR: 8.48e-06, Accumulation Step: 5/8, Current Time: 2025-02-09 23:39:59 
Step 2230, Loss: 7.4076, Scaled Loss: 0.9260, LR: 8.51e-06, Accumulation Step: 7/8, Current Time: 2025-02-09 23:53:06 
Step 2240, Loss: 7.2880, Scaled Loss: 0.9110, LR: 8.57e-06, Accumulation Step: 1/8, Current Time: 2025-02-10 00:08:09 
Step 2250, Loss: 7.4294, Scaled Loss: 0.9287, LR: 8.60e-06, Accumulation Step: 3/8, Current Time: 2025-02-10 00:22:57 
Step 2260, Loss: 7.1723, Scaled Loss: 0.8965, LR: 8.64e-06, Accumulation Step: 5/8, Current Time: 2025-02-10 00:35:52 
Step 2270, Loss: 7.2499, Scaled Loss: 0.9062, LR: 8.67e-06, Accumulation Step: 7/8, Current Time: 2025-02-10 00:49:45 
Step 2280, Loss: 7.1593, Scaled Loss: 0.8949, LR: 8.73e-06, Accumulation Step: 1/8, Current Time: 2025-02-10 01:03:08 
Step 2290, Loss: 7.2295, Scaled Loss: 0.9037, LR: 8.77e-06, Accumulation Step: 3/8, Current Time: 2025-02-10 01:15:56 
Step 2300, Loss: 7.1161, Scaled Loss: 0.8895, LR: 8.80e-06, Accumulation Step: 5/8, Current Time: 2025-02-10 01:29:55 
Step 2310, Loss: 7.0538, Scaled Loss: 0.8817, LR: 8.83e-06, Accumulation Step: 7/8, Current Time: 2025-02-10 01:44:00 
Step 2320, Loss: 7.0934, Scaled Loss: 0.8867, LR: 8.90e-06, Accumulation Step: 1/8, Current Time: 2025-02-10 01:57:24 
Step 2330, Loss: 7.1449, Scaled Loss: 0.8931, LR: 8.93e-06, Accumulation Step: 3/8, Current Time: 2025-02-10 02:10:49 
Step 2340, Loss: 7.1101, Scaled Loss: 0.8888, LR: 8.97e-06, Accumulation Step: 5/8, Current Time: 2025-02-10 02:25:10 
Step 2350, Loss: 7.0490, Scaled Loss: 0.8811, LR: 9.00e-06, Accumulation Step: 7/8, Current Time: 2025-02-10 02:38:57 
Step 2360, Loss: 7.1870, Scaled Loss: 0.8984, LR: 9.07e-06, Accumulation Step: 1/8, Current Time: 2025-02-10 02:53:37 
Step 2370, Loss: 7.1526, Scaled Loss: 0.8941, LR: 9.10e-06, Accumulation Step: 3/8, Current Time: 2025-02-10 03:07:56 
Step 2380, Loss: 7.1707, Scaled Loss: 0.8963, LR: 9.13e-06, Accumulation Step: 5/8, Current Time: 2025-02-10 03:22:39 
Step 2390, Loss: 7.0723, Scaled Loss: 0.8840, LR: 9.17e-06, Accumulation Step: 7/8, Current Time: 2025-02-10 03:36:10 
Step 2400, Loss: 7.0188, Scaled Loss: 0.8773, LR: 9.24e-06, Accumulation Step: 1/8, Current Time: 2025-02-10 03:50:54 
Step 2410, Loss: 6.9072, Scaled Loss: 0.8634, LR: 9.27e-06, Accumulation Step: 3/8, Current Time: 2025-02-10 04:06:00 
Step 2420, Loss: 7.1175, Scaled Loss: 0.8897, LR: 9.31e-06, Accumulation Step: 5/8, Current Time: 2025-02-10 04:19:34 
Step 2430, Loss: 7.0469, Scaled Loss: 0.8809, LR: 9.34e-06, Accumulation Step: 7/8, Current Time: 2025-02-10 04:34:00 
Step 2440, Loss: 6.9501, Scaled Loss: 0.8688, LR: 9.41e-06, Accumulation Step: 1/8, Current Time: 2025-02-10 04:49:27 
Step 2450, Loss: 7.1003, Scaled Loss: 0.8875, LR: 9.44e-06, Accumulation Step: 3/8, Current Time: 2025-02-10 05:03:39 
Step 2460, Loss: 7.2675, Scaled Loss: 0.9084, LR: 9.48e-06, Accumulation Step: 5/8, Current Time: 2025-02-10 05:17:14 
Step 2470, Loss: 7.2256, Scaled Loss: 0.9032, LR: 9.51e-06, Accumulation Step: 7/8, Current Time: 2025-02-10 05:31:12 
Step 2480, Loss: 7.0294, Scaled Loss: 0.8787, LR: 9.58e-06, Accumulation Step: 1/8, Current Time: 2025-02-10 05:45:47 
Step 2490, Loss: 6.8715, Scaled Loss: 0.8589, LR: 9.62e-06, Accumulation Step: 3/8, Current Time: 2025-02-10 06:00:04 
Step 2500, Loss: 7.0779, Scaled Loss: 0.8847, LR: 9.66e-06, Accumulation Step: 5/8, Current Time: 2025-02-10 06:15:03 

=== Generating Sample Texts ===

Prompt: The future of artificial intelligence is lies in agentic AI. 
Temperature: 1.0
Generated: The future of artificial intelligence is lies in agentic AI.          describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing describing

Prompt: No matter how athletic a supermarket checkout clerk is, they’re not likely to scan
Temperature: 1.0
Generated: No matter how athletic a supermarket checkout clerk is, they’re not likely to scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan

Prompt: But for many jobs that primarily involve applying knowledge or processing information
Temperature: 1.0
Generated: But for many jobs that primarily involve applying knowledge or processing information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information

=== End of Samples ===

Step 2510, Loss: 6.7392, Scaled Loss: 0.8424, LR: 9.69e-06, Accumulation Step: 7/8, Current Time: 2025-02-10 06:40:43 
Step 2520, Loss: 6.7962, Scaled Loss: 0.8495, LR: 9.76e-06, Accumulation Step: 1/8, Current Time: 2025-02-10 06:58:33 
Step 2530, Loss: 6.9140, Scaled Loss: 0.8643, LR: 9.80e-06, Accumulation Step: 3/8, Current Time: 2025-02-10 07:14:01 
Step 2540, Loss: 6.8856, Scaled Loss: 0.8607, LR: 9.83e-06, Accumulation Step: 5/8, Current Time: 2025-02-10 07:28:59 
Step 2550, Loss: 6.9195, Scaled Loss: 0.8649, LR: 9.87e-06, Accumulation Step: 7/8, Current Time: 2025-02-10 07:44:45 
Step 2560, Loss: 6.9235, Scaled Loss: 0.8654, LR: 9.94e-06, Accumulation Step: 1/8, Current Time: 2025-02-10 08:00:51 
Step 2570, Loss: 6.9014, Scaled Loss: 0.8627, LR: 9.98e-06, Accumulation Step: 3/8, Current Time: 2025-02-10 08:16:32 
Step 2580, Loss: 6.9398, Scaled Loss: 0.8675, LR: 1.00e-05, Accumulation Step: 5/8, Current Time: 2025-02-10 08:33:11 
Step 2590, Loss: 7.0997, Scaled Loss: 0.8875, LR: 1.01e-05, Accumulation Step: 7/8, Current Time: 2025-02-10 08:48:53 
Step 2600, Loss: 6.8823, Scaled Loss: 0.8603, LR: 1.01e-05, Accumulation Step: 1/8, Current Time: 2025-02-10 09:05:49 
Step 2610, Loss: 6.8322, Scaled Loss: 0.8540, LR: 1.02e-05, Accumulation Step: 3/8, Current Time: 2025-02-10 09:21:04 
Step 2620, Loss: 6.8664, Scaled Loss: 0.8583, LR: 1.02e-05, Accumulation Step: 5/8, Current Time: 2025-02-10 09:36:21 
Step 2630, Loss: 6.9030, Scaled Loss: 0.8629, LR: 1.02e-05, Accumulation Step: 7/8, Current Time: 2025-02-10 09:50:17 
Step 2640, Loss: 7.0041, Scaled Loss: 0.8755, LR: 1.03e-05, Accumulation Step: 1/8, Current Time: 2025-02-10 10:05:19 
Step 2650, Loss: 6.6650, Scaled Loss: 0.8331, LR: 1.03e-05, Accumulation Step: 3/8, Current Time: 2025-02-10 10:20:04 
Step 2660, Loss: 6.9025, Scaled Loss: 0.8628, LR: 1.04e-05, Accumulation Step: 5/8, Current Time: 2025-02-10 10:35:39 
Step 2670, Loss: 6.9431, Scaled Loss: 0.8679, LR: 1.04e-05, Accumulation Step: 7/8, Current Time: 2025-02-10 10:51:13 
Step 2680, Loss: 6.9648, Scaled Loss: 0.8706, LR: 1.05e-05, Accumulation Step: 1/8, Current Time: 2025-02-10 11:07:09 
Step 2690, Loss: 6.7518, Scaled Loss: 0.8440, LR: 1.05e-05, Accumulation Step: 3/8, Current Time: 2025-02-10 11:23:53 
Step 2700, Loss: 7.0149, Scaled Loss: 0.8769, LR: 1.06e-05, Accumulation Step: 5/8, Current Time: 2025-02-10 11:39:54 
Step 2710, Loss: 6.8831, Scaled Loss: 0.8604, LR: 1.06e-05, Accumulation Step: 7/8, Current Time: 2025-02-10 11:55:41 
/opt/anaconda3/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Tracking run with wandb version 0.19.4
wandb: W&B syncing is set to `offline` in this directory.  
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.

Model Statistics:
Total Parameters: 688,702,098
Model Size: 2627.19 MB
Device: mps
Batch Size: 4
Accumulation Steps: 8
Sequence Length: 256
Learning Rate: 0.0001
--------------------------------------------------

Epoch 1/5
Step 2700, Loss: 6.9504, Scaled Loss: 0.8688, LR: 1.06e-05, Accumulation Step: 5/8, Current Time: 2025-02-10 13:24:46 
Step 2710, Loss: 6.8707, Scaled Loss: 0.8588, LR: 1.06e-05, Accumulation Step: 7/8, Current Time: 2025-02-10 13:37:27 
Step 2720, Loss: 6.7345, Scaled Loss: 0.8418, LR: 1.07e-05, Accumulation Step: 1/8, Current Time: 2025-02-10 13:50:36 
Step 2730, Loss: 6.8423, Scaled Loss: 0.8553, LR: 1.07e-05, Accumulation Step: 3/8, Current Time: 2025-02-10 14:04:37 
Step 2740, Loss: 7.0060, Scaled Loss: 0.8757, LR: 1.08e-05, Accumulation Step: 5/8, Current Time: 2025-02-10 14:18:16 
Step 2750, Loss: 6.7298, Scaled Loss: 0.8412, LR: 1.08e-05, Accumulation Step: 7/8, Current Time: 2025-02-10 14:31:18 
Step 2760, Loss: 6.8000, Scaled Loss: 0.8500, LR: 1.09e-05, Accumulation Step: 1/8, Current Time: 2025-02-10 14:46:16 
Step 2770, Loss: 6.6729, Scaled Loss: 0.8341, LR: 1.09e-05, Accumulation Step: 3/8, Current Time: 2025-02-10 15:01:23 
Step 2780, Loss: 6.7941, Scaled Loss: 0.8493, LR: 1.10e-05, Accumulation Step: 5/8, Current Time: 2025-02-10 15:15:15 
Step 2790, Loss: 6.8864, Scaled Loss: 0.8608, LR: 1.10e-05, Accumulation Step: 7/8, Current Time: 2025-02-10 15:28:56 
Step 2800, Loss: 6.8039, Scaled Loss: 0.8505, LR: 1.11e-05, Accumulation Step: 1/8, Current Time: 2025-02-10 15:46:12 
Step 2810, Loss: 6.8498, Scaled Loss: 0.8562, LR: 1.11e-05, Accumulation Step: 3/8, Current Time: 2025-02-10 16:01:28 
Step 2820, Loss: 6.8121, Scaled Loss: 0.8515, LR: 1.12e-05, Accumulation Step: 5/8, Current Time: 2025-02-10 16:16:49 
Step 2830, Loss: 6.6542, Scaled Loss: 0.8318, LR: 1.12e-05, Accumulation Step: 7/8, Current Time: 2025-02-10 16:33:02 
Step 2840, Loss: 6.7154, Scaled Loss: 0.8394, LR: 1.13e-05, Accumulation Step: 1/8, Current Time: 2025-02-10 16:49:51 
Step 2850, Loss: 6.7734, Scaled Loss: 0.8467, LR: 1.13e-05, Accumulation Step: 3/8, Current Time: 2025-02-10 17:03:56 
Step 2860, Loss: 6.8854, Scaled Loss: 0.8607, LR: 1.14e-05, Accumulation Step: 5/8, Current Time: 2025-02-10 17:17:26 
Step 2870, Loss: 6.7694, Scaled Loss: 0.8462, LR: 1.14e-05, Accumulation Step: 7/8, Current Time: 2025-02-10 17:31:28 
Step 2880, Loss: 6.6991, Scaled Loss: 0.8374, LR: 1.15e-05, Accumulation Step: 1/8, Current Time: 2025-02-10 17:45:56 
Step 2890, Loss: 6.7697, Scaled Loss: 0.8462, LR: 1.15e-05, Accumulation Step: 3/8, Current Time: 2025-02-10 18:00:03 
Step 2900, Loss: 6.6851, Scaled Loss: 0.8356, LR: 1.16e-05, Accumulation Step: 5/8, Current Time: 2025-02-10 18:13:14 
Step 2910, Loss: 6.8154, Scaled Loss: 0.8519, LR: 1.16e-05, Accumulation Step: 7/8, Current Time: 2025-02-10 18:26:52 
Step 2920, Loss: 6.8151, Scaled Loss: 0.8519, LR: 1.17e-05, Accumulation Step: 1/8, Current Time: 2025-02-10 18:40:36 
Step 2930, Loss: 6.5902, Scaled Loss: 0.8238, LR: 1.17e-05, Accumulation Step: 3/8, Current Time: 2025-02-10 18:54:00 
Step 2940, Loss: 6.6334, Scaled Loss: 0.8292, LR: 1.18e-05, Accumulation Step: 5/8, Current Time: 2025-02-10 19:07:32 
Step 2950, Loss: 6.6641, Scaled Loss: 0.8330, LR: 1.18e-05, Accumulation Step: 7/8, Current Time: 2025-02-10 19:20:36 
Step 2960, Loss: 6.6114, Scaled Loss: 0.8264, LR: 1.19e-05, Accumulation Step: 1/8, Current Time: 2025-02-10 19:34:53 
Step 2970, Loss: 6.5194, Scaled Loss: 0.8149, LR: 1.19e-05, Accumulation Step: 3/8, Current Time: 2025-02-10 19:48:47 
Step 2980, Loss: 6.6147, Scaled Loss: 0.8268, LR: 1.20e-05, Accumulation Step: 5/8, Current Time: 2025-02-10 20:02:56 
Step 2990, Loss: 6.7324, Scaled Loss: 0.8415, LR: 1.20e-05, Accumulation Step: 7/8, Current Time: 2025-02-10 20:19:07 
Step 3000, Loss: 6.6512, Scaled Loss: 0.8314, LR: 1.21e-05, Accumulation Step: 1/8, Current Time: 2025-02-10 20:36:26 

=== Generating Sample Texts ===

Prompt: The future of artificial intelligence is lies in agentic AI. 
Temperature: 1.0
Generated: The future of artificial intelligence is lies in agentic AI.     judjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjudjud

Prompt: No matter how athletic a supermarket checkout clerk is, they’re not likely to scan
Temperature: 1.0
Generated: No matter how athletic a supermarket checkout clerk is, they’re not likely to scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan

Prompt: But for many jobs that primarily involve applying knowledge or processing information
Temperature: 1.0
Generated: But for many jobs that primarily involve applying knowledge or processing information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information information

Prompt: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles or thermogenic effects. 
Temperature: 1.0
Generated: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles or thermogenic effects.         infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt infilt

Prompt: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now.
Temperature: 1.0
Generated: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now.....................................................................................................

=== End of Samples ===

Step 3010, Loss: 6.6565, Scaled Loss: 0.8321, LR: 1.21e-05, Accumulation Step: 3/8, Current Time: 2025-02-10 21:09:46 
/opt/anaconda3/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Tracking run with wandb version 0.19.4
wandb: W&B syncing is set to `offline` in this directory.  
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.

Model Statistics:
Total Parameters: 688,702,098
Model Size: 2627.19 MB
Device: mps
Batch Size: 4
Accumulation Steps: 8
Sequence Length: 256
Learning Rate: 0.0001
--------------------------------------------------

Epoch 1/5
Step 3000, Loss: 6.6259, Scaled Loss: 0.8282, LR: 1.21e-05, Accumulation Step: 1/8, Current Time: 2025-02-10 21:18:05 
'HTTPSConnectionPool(host='cdn-lfs-us-1.hf.co', port=443): Read timed out.' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00000-of-00104.parquet
Retrying in 1s [Retry 1/5].

=== Generating Sample Texts ===
'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: aaa625c9-1711-47ec-b3ac-2f46cc6fc33b)')' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00000-of-00104.parquet
Retrying in 2s [Retry 2/5].

Prompt: The future of artificial intelligence is lies in agentic AI. 
Temperature: 1.0
Generated: The future of artificial intelligence is lies in agentic AI.   precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence precedence

Prompt: No matter how athletic a supermarket checkout clerk is, they’re not likely to scan. 
Temperature: 1.0
Generated: No matter how athletic a supermarket checkout clerk is, they’re not likely to scan.   BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA BPA

Prompt: But for many jobs that primarily involve applying knowledge or processing information. 
Temperature: 1.0
Generated: But for many jobs that primarily involve applying knowledge or processing information.   Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then Then

Prompt: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles or thermogenic effects. 
Temperature: 1.0
Generated: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles or thermogenic effects.            PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART PART

Prompt: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. 
Temperature: 1.0
Generated: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now.    thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost

=== End of Samples ===

Step 3010, Loss: 6.6290, Scaled Loss: 0.8286, LR: 1.21e-05, Accumulation Step: 3/8, Current Time: 2025-02-10 21:52:23 
Step 3020, Loss: 6.5124, Scaled Loss: 0.8141, LR: 1.22e-05, Accumulation Step: 5/8, Current Time: 2025-02-10 22:07:58 
Step 3030, Loss: 6.6033, Scaled Loss: 0.8254, LR: 1.22e-05, Accumulation Step: 7/8, Current Time: 2025-02-10 22:23:25 
Step 3040, Loss: 6.7276, Scaled Loss: 0.8409, LR: 1.23e-05, Accumulation Step: 1/8, Current Time: 2025-02-10 22:39:07 
Step 3050, Loss: 6.5179, Scaled Loss: 0.8147, LR: 1.24e-05, Accumulation Step: 3/8, Current Time: 2025-02-10 22:53:29 
Step 3060, Loss: 6.5055, Scaled Loss: 0.8132, LR: 1.24e-05, Accumulation Step: 5/8, Current Time: 2025-02-10 23:07:50 
Step 3070, Loss: 6.4607, Scaled Loss: 0.8076, LR: 1.24e-05, Accumulation Step: 7/8, Current Time: 2025-02-10 23:22:19 
Step 3080, Loss: 6.5969, Scaled Loss: 0.8246, LR: 1.25e-05, Accumulation Step: 1/8, Current Time: 2025-02-10 23:37:16 
Step 3090, Loss: 6.6337, Scaled Loss: 0.8292, LR: 1.26e-05, Accumulation Step: 3/8, Current Time: 2025-02-10 23:51:58 
Step 3100, Loss: 6.5627, Scaled Loss: 0.8203, LR: 1.26e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 00:06:34 
Step 3110, Loss: 6.6141, Scaled Loss: 0.8268, LR: 1.27e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 00:23:26 
Step 3120, Loss: 6.5240, Scaled Loss: 0.8155, LR: 1.27e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 00:38:58 
Step 3130, Loss: 6.4671, Scaled Loss: 0.8084, LR: 1.28e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 00:53:19 
Step 3140, Loss: 6.4731, Scaled Loss: 0.8091, LR: 1.28e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 01:07:51 
Step 3150, Loss: 6.5463, Scaled Loss: 0.8183, LR: 1.29e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 01:22:36 
Step 3160, Loss: 6.6129, Scaled Loss: 0.8266, LR: 1.30e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 01:38:11 
Step 3170, Loss: 6.4939, Scaled Loss: 0.8117, LR: 1.30e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 01:53:37 
Step 3180, Loss: 6.4686, Scaled Loss: 0.8086, LR: 1.30e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 02:08:34 
Step 3190, Loss: 6.5132, Scaled Loss: 0.8141, LR: 1.31e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 02:24:18 
Step 3200, Loss: 6.4293, Scaled Loss: 0.8037, LR: 1.32e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 02:39:30 
Step 3210, Loss: 6.5807, Scaled Loss: 0.8226, LR: 1.32e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 02:55:07 
Step 3220, Loss: 6.5967, Scaled Loss: 0.8246, LR: 1.33e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 03:10:15 
Step 3230, Loss: 6.3540, Scaled Loss: 0.7942, LR: 1.33e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 03:25:29 
Step 3240, Loss: 6.3831, Scaled Loss: 0.7979, LR: 1.34e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 03:41:28 
Step 3250, Loss: 6.4411, Scaled Loss: 0.8051, LR: 1.34e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 03:55:21 
Step 3260, Loss: 6.3992, Scaled Loss: 0.7999, LR: 1.35e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 04:09:21 
Step 3270, Loss: 6.2955, Scaled Loss: 0.7869, LR: 1.35e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 04:22:32 
Step 3280, Loss: 6.3872, Scaled Loss: 0.7984, LR: 1.36e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 04:36:45 
Step 3290, Loss: 6.4815, Scaled Loss: 0.8102, LR: 1.37e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 04:51:52 
Step 3300, Loss: 6.4697, Scaled Loss: 0.8087, LR: 1.37e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 05:07:03 
Step 3310, Loss: 6.4812, Scaled Loss: 0.8101, LR: 1.38e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 05:23:29 
Step 3320, Loss: 6.4096, Scaled Loss: 0.8012, LR: 1.39e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 05:40:36 
Step 3330, Loss: 6.3657, Scaled Loss: 0.7957, LR: 1.39e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 05:56:43 
Step 3340, Loss: 6.3725, Scaled Loss: 0.7966, LR: 1.39e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 06:12:28 
Step 3350, Loss: 6.4030, Scaled Loss: 0.8004, LR: 1.40e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 06:28:12 
Step 3360, Loss: 6.3934, Scaled Loss: 0.7992, LR: 1.41e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 06:43:21 
Step 3370, Loss: 6.3616, Scaled Loss: 0.7952, LR: 1.41e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 06:58:19 
Step 3380, Loss: 6.3026, Scaled Loss: 0.7878, LR: 1.42e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 07:14:44 
Step 3390, Loss: 6.3567, Scaled Loss: 0.7946, LR: 1.42e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 07:31:19 
Step 3400, Loss: 6.4263, Scaled Loss: 0.8033, LR: 1.43e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 07:51:26 
Step 3410, Loss: 6.3189, Scaled Loss: 0.7899, LR: 1.44e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 08:07:54 
Step 3420, Loss: 6.3846, Scaled Loss: 0.7981, LR: 1.44e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 08:22:44 
Step 3430, Loss: 6.4519, Scaled Loss: 0.8065, LR: 1.45e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 08:48:12 
Step 3440, Loss: 6.2625, Scaled Loss: 0.7828, LR: 1.45e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 09:03:56 
Step 3450, Loss: 6.3590, Scaled Loss: 0.7949, LR: 1.46e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 09:17:40 
Step 3460, Loss: 6.2187, Scaled Loss: 0.7773, LR: 1.46e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 09:32:56 
Step 3470, Loss: 6.3575, Scaled Loss: 0.7947, LR: 1.47e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 09:48:12 
Step 3480, Loss: 6.3551, Scaled Loss: 0.7944, LR: 1.48e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 10:02:11 
Step 3490, Loss: 6.4007, Scaled Loss: 0.8001, LR: 1.48e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 10:15:11 
Step 3500, Loss: 6.2864, Scaled Loss: 0.7858, LR: 1.49e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 10:28:44 

=== Generating Sample Texts ===

Prompt: The future of artificial intelligence is lies in agentic AI. 
Temperature: 1.0
Generated: The future of artificial intelligence is lies in agentic AI.     vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines vines

Prompt: No matter how athletic a supermarket checkout clerk is, they’re not likely to scan. 
Temperature: 1.0
Generated: No matter how athletic a supermarket checkout clerk is, they’re not likely to scan.      Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador

Prompt: But for many jobs that primarily involve applying knowledge or processing information. 
Temperature: 1.0
Generated: But for many jobs that primarily involve applying knowledge or processing information.     Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador Salvador

Prompt: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles or thermogenic effects. 
Temperature: 1.0
Generated: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles or thermogenic effects.          AccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccordingAccording

Prompt: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. 
Temperature: 1.0
Generated: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now.           thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost thermost

=== End of Samples ===

Step 3510, Loss: 6.4427, Scaled Loss: 0.8053, LR: 1.49e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 10:57:06 
Step 3520, Loss: 6.2700, Scaled Loss: 0.7837, LR: 1.50e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 11:12:08 
Step 3530, Loss: 6.3094, Scaled Loss: 0.7887, LR: 1.51e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 11:25:40 
Step 3540, Loss: 6.2830, Scaled Loss: 0.7854, LR: 1.51e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 11:40:19 
Step 3550, Loss: 6.2984, Scaled Loss: 0.7873, LR: 1.52e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 11:53:51 
Step 3560, Loss: 6.2126, Scaled Loss: 0.7766, LR: 1.53e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 12:08:35 
Step 3570, Loss: 6.2312, Scaled Loss: 0.7789, LR: 1.53e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 12:22:10 
Step 3580, Loss: 6.3094, Scaled Loss: 0.7887, LR: 1.54e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 12:34:55 
Step 3590, Loss: 6.2582, Scaled Loss: 0.7823, LR: 1.54e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 12:48:26 
Step 3600, Loss: 6.3092, Scaled Loss: 0.7887, LR: 1.55e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 13:02:54 
Step 3610, Loss: 6.2123, Scaled Loss: 0.7765, LR: 1.56e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 13:19:30 
Step 3620, Loss: 6.2553, Scaled Loss: 0.7819, LR: 1.56e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 13:34:18 
Step 3630, Loss: 6.3716, Scaled Loss: 0.7964, LR: 1.57e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 13:48:48 
Step 3640, Loss: 6.3650, Scaled Loss: 0.7956, LR: 1.58e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 14:03:11 
Step 3650, Loss: 6.4303, Scaled Loss: 0.8038, LR: 1.58e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 14:17:15 
Step 3660, Loss: 6.2864, Scaled Loss: 0.7858, LR: 1.59e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 14:32:25 
Step 3670, Loss: 6.3035, Scaled Loss: 0.7879, LR: 1.59e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 14:49:49 
Step 3680, Loss: 6.2321, Scaled Loss: 0.7790, LR: 1.60e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 15:05:09 
Step 3690, Loss: 6.3408, Scaled Loss: 0.7926, LR: 1.61e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 15:20:24 
Step 3700, Loss: 6.2630, Scaled Loss: 0.7829, LR: 1.61e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 15:36:06 
Step 3710, Loss: 6.1635, Scaled Loss: 0.7704, LR: 1.62e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 15:55:39 
Step 3720, Loss: 6.2130, Scaled Loss: 0.7766, LR: 1.63e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 16:13:37 
Step 3730, Loss: 6.2210, Scaled Loss: 0.7776, LR: 1.63e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 16:45:29 
Step 3740, Loss: 6.2164, Scaled Loss: 0.7771, LR: 1.64e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 17:03:34 
Step 3750, Loss: 6.2018, Scaled Loss: 0.7752, LR: 1.64e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 17:20:33 
Step 3760, Loss: 6.2296, Scaled Loss: 0.7787, LR: 1.65e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 17:37:35 
Step 3770, Loss: 6.3206, Scaled Loss: 0.7901, LR: 1.66e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 17:52:58 
Step 3780, Loss: 6.2469, Scaled Loss: 0.7809, LR: 1.66e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 18:10:23 
Step 3790, Loss: 6.1951, Scaled Loss: 0.7744, LR: 1.67e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 18:25:49 
Step 3800, Loss: 6.1384, Scaled Loss: 0.7673, LR: 1.68e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 18:41:31 
Step 3810, Loss: 6.1482, Scaled Loss: 0.7685, LR: 1.68e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 18:57:29 
Step 3820, Loss: 6.2582, Scaled Loss: 0.7823, LR: 1.69e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 19:12:06 
Step 3830, Loss: 6.1873, Scaled Loss: 0.7734, LR: 1.69e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 19:26:43 
Step 3840, Loss: 6.1506, Scaled Loss: 0.7688, LR: 1.70e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 19:40:52 
Step 3850, Loss: 6.2228, Scaled Loss: 0.7778, LR: 1.71e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 19:54:51 
Step 3860, Loss: 6.2589, Scaled Loss: 0.7824, LR: 1.71e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 20:10:50 
Step 3870, Loss: 6.2080, Scaled Loss: 0.7760, LR: 1.72e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 20:25:58 
Step 3880, Loss: 6.1942, Scaled Loss: 0.7743, LR: 1.73e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 20:46:34 
Step 3890, Loss: 6.1172, Scaled Loss: 0.7647, LR: 1.73e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 21:02:49 
Step 3900, Loss: 6.2117, Scaled Loss: 0.7765, LR: 1.74e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 21:20:21 
Step 3910, Loss: 6.0077, Scaled Loss: 0.7510, LR: 1.74e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 21:35:46 
Step 3920, Loss: 6.0681, Scaled Loss: 0.7585, LR: 1.75e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 21:52:28 
Step 3930, Loss: 6.0995, Scaled Loss: 0.7624, LR: 1.76e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 22:10:01 
Step 3940, Loss: 6.0800, Scaled Loss: 0.7600, LR: 1.76e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 22:24:47 
Step 3950, Loss: 6.1416, Scaled Loss: 0.7677, LR: 1.77e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 22:38:51 
Step 3960, Loss: 6.0958, Scaled Loss: 0.7620, LR: 1.78e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 22:54:31 
Step 3970, Loss: 6.1155, Scaled Loss: 0.7644, LR: 1.79e-05, Accumulation Step: 3/8, Current Time: 2025-02-11 23:08:27 
Step 3980, Loss: 6.1229, Scaled Loss: 0.7654, LR: 1.79e-05, Accumulation Step: 5/8, Current Time: 2025-02-11 23:21:35 
Step 3990, Loss: 6.1959, Scaled Loss: 0.7745, LR: 1.80e-05, Accumulation Step: 7/8, Current Time: 2025-02-11 23:35:28 
Step 4000, Loss: 6.0613, Scaled Loss: 0.7577, LR: 1.81e-05, Accumulation Step: 1/8, Current Time: 2025-02-11 23:50:22 

=== Generating Sample Texts ===

Prompt: The future of artificial intelligence is lies in agentic AI. 
Temperature: 1.0
Generated: The future of artificial intelligence is lies in agentic AI.   cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber cucumber

Prompt: No matter how athletic a supermarket checkout clerk is, they’re not likely to scan. 
Temperature: 1.0
Generated: No matter how athletic a supermarket checkout clerk is, they’re not likely to scan.                   euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph

Prompt: But for many jobs that primarily involve applying knowledge or processing information. 
Temperature: 1.0
Generated: But for many jobs that primarily involve applying knowledge or processing information.         appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately appropriately

Prompt: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles or thermogenic effects. 
Temperature: 1.0
Generated: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles or thermogenic effects.                 Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma Ma

Prompt: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. 
Temperature: 1.0
Generated: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now.     atomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomyatomy

=== End of Samples ===

Step 4010, Loss: 6.1180, Scaled Loss: 0.7648, LR: 1.81e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 00:20:36 
Step 4020, Loss: 6.0593, Scaled Loss: 0.7574, LR: 1.82e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 00:34:56 
Step 4030, Loss: 6.1080, Scaled Loss: 0.7635, LR: 1.82e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 00:48:51 
Step 4040, Loss: 6.0459, Scaled Loss: 0.7557, LR: 1.83e-05, Accumulation Step: 1/8, Current Time: 2025-02-12 01:03:07 
Step 4050, Loss: 5.9840, Scaled Loss: 0.7480, LR: 1.84e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 01:16:52 
Step 4060, Loss: 6.0743, Scaled Loss: 0.7593, LR: 1.84e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 01:30:34 
Step 4070, Loss: 6.0932, Scaled Loss: 0.7616, LR: 1.85e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 01:43:30 
Step 4080, Loss: 6.1274, Scaled Loss: 0.7659, LR: 1.86e-05, Accumulation Step: 1/8, Current Time: 2025-02-12 01:57:32 
Step 4090, Loss: 6.0879, Scaled Loss: 0.7610, LR: 1.87e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 02:11:30 
Step 4100, Loss: 6.1495, Scaled Loss: 0.7687, LR: 1.87e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 02:25:14 
Step 4110, Loss: 6.0843, Scaled Loss: 0.7605, LR: 1.88e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 02:40:45 
Step 4120, Loss: 6.1257, Scaled Loss: 0.7657, LR: 1.89e-05, Accumulation Step: 1/8, Current Time: 2025-02-12 02:55:43 
Step 4130, Loss: 6.1387, Scaled Loss: 0.7673, LR: 1.89e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 03:09:49 
Step 4140, Loss: 6.0464, Scaled Loss: 0.7558, LR: 1.90e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 03:24:30 
Step 4150, Loss: 6.0372, Scaled Loss: 0.7546, LR: 1.90e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 03:38:31 
Step 4160, Loss: 6.0381, Scaled Loss: 0.7548, LR: 1.92e-05, Accumulation Step: 1/8, Current Time: 2025-02-12 03:53:07 
Step 4170, Loss: 6.1749, Scaled Loss: 0.7719, LR: 1.92e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 04:06:29 
Step 4180, Loss: 6.0252, Scaled Loss: 0.7531, LR: 1.93e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 04:19:54 
Step 4190, Loss: 6.1036, Scaled Loss: 0.7630, LR: 1.93e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 04:33:23 
Step 4200, Loss: 6.0061, Scaled Loss: 0.7508, LR: 1.94e-05, Accumulation Step: 1/8, Current Time: 2025-02-12 04:48:07 
Step 4210, Loss: 5.9646, Scaled Loss: 0.7456, LR: 1.95e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 05:03:05 
Step 4220, Loss: 5.9641, Scaled Loss: 0.7455, LR: 1.95e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 05:17:05 
Step 4230, Loss: 6.0740, Scaled Loss: 0.7593, LR: 1.96e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 05:31:24 
Step 4240, Loss: 6.0386, Scaled Loss: 0.7548, LR: 1.97e-05, Accumulation Step: 1/8, Current Time: 2025-02-12 05:47:31 
Step 4250, Loss: 6.0523, Scaled Loss: 0.7565, LR: 1.98e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 06:04:56 
Step 4260, Loss: 6.0452, Scaled Loss: 0.7556, LR: 1.98e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 06:20:48 
Step 4270, Loss: 5.9937, Scaled Loss: 0.7492, LR: 1.99e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 06:38:13 
Step 4280, Loss: 5.9986, Scaled Loss: 0.7498, LR: 2.00e-05, Accumulation Step: 1/8, Current Time: 2025-02-12 06:56:42 
Step 4290, Loss: 6.0012, Scaled Loss: 0.7502, LR: 2.00e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 07:17:58 
Step 4300, Loss: 5.9643, Scaled Loss: 0.7455, LR: 2.01e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 07:33:40 
Step 4310, Loss: 5.9809, Scaled Loss: 0.7476, LR: 2.02e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 07:48:27 
/opt/anaconda3/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Tracking run with wandb version 0.19.4
wandb: W&B syncing is set to `offline` in this directory.  
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.

Model Statistics:
Total Parameters: 688,702,098
Model Size: 2627.19 MB
Device: mps
Batch Size: 4
Accumulation Steps: 8
Sequence Length: 256
Learning Rate: 0.0001
--------------------------------------------------

Epoch 1/5
Step 4300, Loss: 5.9965, Scaled Loss: 0.7496, LR: 2.01e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 16:18:18 
Step 4310, Loss: 6.0118, Scaled Loss: 0.7515, LR: 2.02e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 16:31:52 
Step 4320, Loss: 5.9455, Scaled Loss: 0.7432, LR: 2.03e-05, Accumulation Step: 1/8, Current Time: 2025-02-12 16:45:03 
Step 4330, Loss: 6.0015, Scaled Loss: 0.7502, LR: 2.03e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 17:00:10 
Step 4340, Loss: 6.0354, Scaled Loss: 0.7544, LR: 2.04e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 17:13:53 
Step 4350, Loss: 5.9965, Scaled Loss: 0.7496, LR: 2.04e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 17:28:18 
Step 4360, Loss: 5.9478, Scaled Loss: 0.7435, LR: 2.06e-05, Accumulation Step: 1/8, Current Time: 2025-02-12 17:43:19 
Step 4370, Loss: 5.9291, Scaled Loss: 0.7411, LR: 2.06e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 17:58:28 
Step 4380, Loss: 6.0028, Scaled Loss: 0.7504, LR: 2.07e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 18:16:27 
Step 4390, Loss: 5.9857, Scaled Loss: 0.7482, LR: 2.07e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 18:33:18 
Step 4400, Loss: 5.9830, Scaled Loss: 0.7479, LR: 2.08e-05, Accumulation Step: 1/8, Current Time: 2025-02-12 18:49:39 
Step 4410, Loss: 5.9879, Scaled Loss: 0.7485, LR: 2.09e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 19:08:12 
Step 4420, Loss: 5.9593, Scaled Loss: 0.7449, LR: 2.10e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 19:25:35 
Step 4430, Loss: 5.9677, Scaled Loss: 0.7460, LR: 2.10e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 19:42:13 
Step 4440, Loss: 5.9343, Scaled Loss: 0.7418, LR: 2.11e-05, Accumulation Step: 1/8, Current Time: 2025-02-12 20:00:51 
Step 4450, Loss: 5.9780, Scaled Loss: 0.7472, LR: 2.12e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 20:16:07 
Step 4460, Loss: 5.9801, Scaled Loss: 0.7475, LR: 2.12e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 20:30:30 
Step 4470, Loss: 5.9643, Scaled Loss: 0.7455, LR: 2.13e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 20:45:37 
Step 4480, Loss: 5.9597, Scaled Loss: 0.7450, LR: 2.14e-05, Accumulation Step: 1/8, Current Time: 2025-02-12 21:01:35 
Step 4490, Loss: 5.9113, Scaled Loss: 0.7389, LR: 2.15e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 21:18:44 
Step 4500, Loss: 5.9343, Scaled Loss: 0.7418, LR: 2.15e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 21:38:10 

=== Generating Sample Texts ===

Prompt: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics! 
Temperature: 1.0
Generated: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics!      fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing fixing

Prompt: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them. 
Temperature: 1.0
Generated: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them.  MSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSGMSG

Prompt: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking. 
Temperature: 1.0
Generated: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking.          ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro ro

Prompt: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles. 
Temperature: 1.0
Generated: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles.        Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal Canal

Prompt: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. 
Temperature: 1.0
Generated: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now.      ****************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************

=== End of Samples ===

Step 4510, Loss: 6.0087, Scaled Loss: 0.7511, LR: 2.16e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 22:19:39 
Step 4520, Loss: 6.0178, Scaled Loss: 0.7522, LR: 2.17e-05, Accumulation Step: 1/8, Current Time: 2025-02-12 22:35:58 
Step 4530, Loss: 5.8534, Scaled Loss: 0.7317, LR: 2.18e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 22:51:44 
Step 4540, Loss: 5.8878, Scaled Loss: 0.7360, LR: 2.18e-05, Accumulation Step: 5/8, Current Time: 2025-02-12 23:07:13 
Step 4550, Loss: 5.9480, Scaled Loss: 0.7435, LR: 2.19e-05, Accumulation Step: 7/8, Current Time: 2025-02-12 23:22:49 
Step 4560, Loss: 5.9375, Scaled Loss: 0.7422, LR: 2.20e-05, Accumulation Step: 1/8, Current Time: 2025-02-12 23:38:22 
Step 4570, Loss: 5.8563, Scaled Loss: 0.7320, LR: 2.21e-05, Accumulation Step: 3/8, Current Time: 2025-02-12 23:53:12 
Step 4580, Loss: 5.9211, Scaled Loss: 0.7401, LR: 2.21e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 00:08:27 
Step 4590, Loss: 5.9426, Scaled Loss: 0.7428, LR: 2.22e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 00:23:12 
Step 4600, Loss: 5.9285, Scaled Loss: 0.7411, LR: 2.23e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 00:39:09 
Step 4610, Loss: 5.9261, Scaled Loss: 0.7408, LR: 2.24e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 00:56:40 
Step 4620, Loss: 5.9018, Scaled Loss: 0.7377, LR: 2.24e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 01:11:32 
Step 4630, Loss: 5.9306, Scaled Loss: 0.7413, LR: 2.25e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 01:27:40 
Step 4640, Loss: 5.8721, Scaled Loss: 0.7340, LR: 2.26e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 01:44:09 
Step 4650, Loss: 5.8963, Scaled Loss: 0.7370, LR: 2.27e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 01:59:42 
Step 4660, Loss: 5.8819, Scaled Loss: 0.7352, LR: 2.27e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 02:14:49 
Step 4670, Loss: 5.8636, Scaled Loss: 0.7329, LR: 2.28e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 02:29:14 
Step 4680, Loss: 5.8427, Scaled Loss: 0.7303, LR: 2.29e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 02:44:36 
Step 4690, Loss: 5.8894, Scaled Loss: 0.7362, LR: 2.30e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 02:59:34 
Step 4700, Loss: 5.9092, Scaled Loss: 0.7387, LR: 2.30e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 03:14:22 
Step 4710, Loss: 5.8536, Scaled Loss: 0.7317, LR: 2.31e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 03:31:40 
Step 4720, Loss: 5.8943, Scaled Loss: 0.7368, LR: 2.32e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 03:47:44 
Step 4730, Loss: 5.9378, Scaled Loss: 0.7422, LR: 2.33e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 04:02:24 
Step 4740, Loss: 5.8104, Scaled Loss: 0.7263, LR: 2.33e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 04:18:01 
Step 4750, Loss: 5.8551, Scaled Loss: 0.7319, LR: 2.34e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 04:33:23 
Step 4760, Loss: 5.8183, Scaled Loss: 0.7273, LR: 2.35e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 04:50:14 
Step 4770, Loss: 5.8598, Scaled Loss: 0.7325, LR: 2.36e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 05:06:57 
Step 4780, Loss: 5.8394, Scaled Loss: 0.7299, LR: 2.36e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 05:24:40 
Step 4790, Loss: 5.9270, Scaled Loss: 0.7409, LR: 2.37e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 05:41:43 
Step 4800, Loss: 5.8291, Scaled Loss: 0.7286, LR: 2.38e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 05:59:37 
Step 4810, Loss: 5.9258, Scaled Loss: 0.7407, LR: 2.39e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 06:17:51 
Step 4820, Loss: 5.8453, Scaled Loss: 0.7307, LR: 2.39e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 06:34:56 
Step 4830, Loss: 5.8594, Scaled Loss: 0.7324, LR: 2.40e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 06:50:38 
Step 4840, Loss: 5.8346, Scaled Loss: 0.7293, LR: 2.41e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 07:10:00 
Step 4850, Loss: 5.8350, Scaled Loss: 0.7294, LR: 2.42e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 07:28:25 
Step 4860, Loss: 5.8164, Scaled Loss: 0.7270, LR: 2.42e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 07:43:26 
Step 4870, Loss: 5.8062, Scaled Loss: 0.7258, LR: 2.43e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 07:58:42 
Step 4880, Loss: 5.8628, Scaled Loss: 0.7328, LR: 2.44e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 08:15:07 
Step 4890, Loss: 5.8434, Scaled Loss: 0.7304, LR: 2.45e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 08:31:04 
Step 4900, Loss: 5.8543, Scaled Loss: 0.7318, LR: 2.45e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 08:46:05 
Step 4910, Loss: 5.7930, Scaled Loss: 0.7241, LR: 2.46e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 09:01:08 
Step 4920, Loss: 5.8038, Scaled Loss: 0.7255, LR: 2.47e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 09:16:27 
Step 4930, Loss: 5.8711, Scaled Loss: 0.7339, LR: 2.48e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 09:31:43 
Step 4940, Loss: 5.8668, Scaled Loss: 0.7333, LR: 2.49e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 09:46:57 
Step 4950, Loss: 5.9494, Scaled Loss: 0.7437, LR: 2.49e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 10:01:55 
Step 4960, Loss: 5.8208, Scaled Loss: 0.7276, LR: 2.50e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 10:18:47 
Step 4970, Loss: 5.8312, Scaled Loss: 0.7289, LR: 2.51e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 10:36:26 
Step 4980, Loss: 5.8221, Scaled Loss: 0.7278, LR: 2.52e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 10:53:46 
Step 4990, Loss: 5.9095, Scaled Loss: 0.7387, LR: 2.52e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 11:09:56 
Step 5000, Loss: 5.8227, Scaled Loss: 0.7278, LR: 2.54e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 11:26:35 

=== Generating Sample Texts ===

Prompt: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics! 
Temperature: 1.0
Generated: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics!            FIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFIFI

Prompt: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them. 
Temperature: 1.0
Generated: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them.     
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            

Prompt: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking. 
Temperature: 1.0
Generated: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking.                                Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large Large

Prompt: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles. 
Temperature: 1.0
Generated: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles.        Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler Doppler

Prompt: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. 
Temperature: 1.0
Generated: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now.         atroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatro

=== End of Samples ===

Step 5010, Loss: 5.7905, Scaled Loss: 0.7238, LR: 2.54e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 11:58:21 
Step 5020, Loss: 5.8208, Scaled Loss: 0.7276, LR: 2.55e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 12:13:34 
Step 5030, Loss: 5.7720, Scaled Loss: 0.7215, LR: 2.55e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 12:32:20 
Step 5040, Loss: 5.8069, Scaled Loss: 0.7259, LR: 2.57e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 12:48:59 
Step 5050, Loss: 5.7803, Scaled Loss: 0.7225, LR: 2.57e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 13:04:27 
Step 5060, Loss: 5.7812, Scaled Loss: 0.7227, LR: 2.58e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 13:19:20 
Step 5070, Loss: 5.8349, Scaled Loss: 0.7294, LR: 2.59e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 13:34:36 
Step 5080, Loss: 5.8451, Scaled Loss: 0.7306, LR: 2.60e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 13:50:34 
Step 5090, Loss: 5.7709, Scaled Loss: 0.7214, LR: 2.60e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 14:07:37 
Step 5100, Loss: 5.7432, Scaled Loss: 0.7179, LR: 2.61e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 14:22:57 
Step 5110, Loss: 5.7704, Scaled Loss: 0.7213, LR: 2.62e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 14:38:33 
Step 5120, Loss: 5.8246, Scaled Loss: 0.7281, LR: 2.63e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 14:54:55 
Step 5130, Loss: 5.7812, Scaled Loss: 0.7226, LR: 2.64e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 15:11:25 
Step 5140, Loss: 5.7722, Scaled Loss: 0.7215, LR: 2.64e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 15:29:26 
Step 5150, Loss: 5.7686, Scaled Loss: 0.7211, LR: 2.65e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 15:45:45 
Step 5160, Loss: 5.7992, Scaled Loss: 0.7249, LR: 2.66e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 16:03:11 
Step 5170, Loss: 5.7719, Scaled Loss: 0.7215, LR: 2.67e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 16:18:39 
Step 5180, Loss: 5.7672, Scaled Loss: 0.7209, LR: 2.67e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 16:34:12 
/opt/anaconda3/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Tracking run with wandb version 0.19.4
wandb: W&B syncing is set to `offline` in this directory.  
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.

Model Statistics:
Total Parameters: 688,702,098
Model Size: 2627.19 MB
Device: mps
Batch Size: 4
Accumulation Steps: 8
Sequence Length: 256
Learning Rate: 0.0001
--------------------------------------------------

Epoch 1/5
Step 5100, Loss: 5.7875, Scaled Loss: 0.7234, LR: 2.61e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 21:14:03 
'HTTPSConnectionPool(host='cdn-lfs-us-1.hf.co', port=443): Read timed out.' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00000-of-00104.parquet
Retrying in 1s [Retry 1/5].
'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: 35774065-693d-433b-bd39-062c010f7aec)')' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00000-of-00104.parquet
Retrying in 2s [Retry 2/5].
'HTTPSConnectionPool(host='cdn-lfs-us-1.hf.co', port=443): Read timed out.' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00000-of-00104.parquet
Retrying in 1s [Retry 1/5].
'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: 3f8e7d7a-34ac-48e3-bfac-2e4ec2391984)')' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00000-of-00104.parquet
Retrying in 2s [Retry 2/5].
Step 5110, Loss: 5.8042, Scaled Loss: 0.7255, LR: 2.62e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 21:25:48 
'HTTPSConnectionPool(host='cdn-lfs-us-1.hf.co', port=443): Read timed out.' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00000-of-00104.parquet
Retrying in 1s [Retry 1/5].
'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: 9856b72d-2750-4c74-8f85-ca3f123bc9ec)')' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00000-of-00104.parquet
Retrying in 2s [Retry 2/5].
Step 5120, Loss: 5.7414, Scaled Loss: 0.7177, LR: 2.63e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 21:39:49 
Step 5130, Loss: 5.7917, Scaled Loss: 0.7240, LR: 2.64e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 21:53:40 
Step 5140, Loss: 5.8407, Scaled Loss: 0.7301, LR: 2.64e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 22:09:55 
Step 5150, Loss: 5.8273, Scaled Loss: 0.7284, LR: 2.65e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 22:25:35 
Step 5160, Loss: 5.7317, Scaled Loss: 0.7165, LR: 2.66e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 22:40:20 
Step 5170, Loss: 5.7341, Scaled Loss: 0.7168, LR: 2.67e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 22:54:20 
Step 5180, Loss: 5.7943, Scaled Loss: 0.7243, LR: 2.67e-05, Accumulation Step: 5/8, Current Time: 2025-02-13 23:07:35 
Step 5190, Loss: 5.7575, Scaled Loss: 0.7197, LR: 2.68e-05, Accumulation Step: 7/8, Current Time: 2025-02-13 23:20:19 
Step 5200, Loss: 5.7503, Scaled Loss: 0.7188, LR: 2.69e-05, Accumulation Step: 1/8, Current Time: 2025-02-13 23:34:46 
Step 5210, Loss: 5.7559, Scaled Loss: 0.7195, LR: 2.70e-05, Accumulation Step: 3/8, Current Time: 2025-02-13 23:49:29 
Step 5220, Loss: 5.7633, Scaled Loss: 0.7204, LR: 2.71e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 00:04:17 
Step 5230, Loss: 5.7518, Scaled Loss: 0.7190, LR: 2.71e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 00:18:40 
Step 5240, Loss: 5.7415, Scaled Loss: 0.7177, LR: 2.73e-05, Accumulation Step: 1/8, Current Time: 2025-02-14 00:33:50 
Step 5250, Loss: 5.7507, Scaled Loss: 0.7188, LR: 2.73e-05, Accumulation Step: 3/8, Current Time: 2025-02-14 00:47:20 
Step 5260, Loss: 5.7410, Scaled Loss: 0.7176, LR: 2.74e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 01:01:08 
Step 5270, Loss: 5.8048, Scaled Loss: 0.7256, LR: 2.75e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 01:14:27 
Step 5280, Loss: 5.7442, Scaled Loss: 0.7180, LR: 2.76e-05, Accumulation Step: 1/8, Current Time: 2025-02-14 01:28:25 
Step 5290, Loss: 5.7037, Scaled Loss: 0.7130, LR: 2.77e-05, Accumulation Step: 3/8, Current Time: 2025-02-14 01:41:55 
Step 5300, Loss: 5.7288, Scaled Loss: 0.7161, LR: 2.77e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 01:55:01 
Step 5310, Loss: 5.7724, Scaled Loss: 0.7215, LR: 2.78e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 02:09:22 
Step 5320, Loss: 5.7805, Scaled Loss: 0.7226, LR: 2.79e-05, Accumulation Step: 1/8, Current Time: 2025-02-14 02:23:41 
Step 5330, Loss: 5.6776, Scaled Loss: 0.7097, LR: 2.80e-05, Accumulation Step: 3/8, Current Time: 2025-02-14 02:37:29 
Step 5340, Loss: 5.6946, Scaled Loss: 0.7118, LR: 2.80e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 02:50:40 
Step 5350, Loss: 5.7262, Scaled Loss: 0.7158, LR: 2.81e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 03:04:13 
Step 5360, Loss: 5.7440, Scaled Loss: 0.7180, LR: 2.82e-05, Accumulation Step: 1/8, Current Time: 2025-02-14 03:18:05 
Step 5370, Loss: 5.6477, Scaled Loss: 0.7060, LR: 2.83e-05, Accumulation Step: 3/8, Current Time: 2025-02-14 03:32:36 
Step 5380, Loss: 5.6956, Scaled Loss: 0.7119, LR: 2.84e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 03:46:19 
Step 5390, Loss: 5.7241, Scaled Loss: 0.7155, LR: 2.84e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 03:59:46 
Step 5400, Loss: 5.7353, Scaled Loss: 0.7169, LR: 2.86e-05, Accumulation Step: 1/8, Current Time: 2025-02-14 04:13:49 
Step 5410, Loss: 5.7085, Scaled Loss: 0.7136, LR: 2.86e-05, Accumulation Step: 3/8, Current Time: 2025-02-14 04:29:05 
Step 5420, Loss: 5.6933, Scaled Loss: 0.7117, LR: 2.87e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 04:43:56 
Step 5430, Loss: 5.7190, Scaled Loss: 0.7149, LR: 2.88e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 04:59:55 
Step 5440, Loss: 5.6645, Scaled Loss: 0.7081, LR: 2.89e-05, Accumulation Step: 1/8, Current Time: 2025-02-14 05:15:16 
Step 5450, Loss: 5.6808, Scaled Loss: 0.7101, LR: 2.90e-05, Accumulation Step: 3/8, Current Time: 2025-02-14 05:29:30 
Step 5460, Loss: 5.6778, Scaled Loss: 0.7097, LR: 2.90e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 05:43:07 
Step 5470, Loss: 5.6590, Scaled Loss: 0.7074, LR: 2.91e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 05:55:12 
Step 5480, Loss: 5.6495, Scaled Loss: 0.7062, LR: 2.92e-05, Accumulation Step: 1/8, Current Time: 2025-02-14 06:10:08 
Step 5490, Loss: 5.6749, Scaled Loss: 0.7094, LR: 2.93e-05, Accumulation Step: 3/8, Current Time: 2025-02-14 06:24:53 
Step 5500, Loss: 5.6916, Scaled Loss: 0.7114, LR: 2.94e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 06:39:42 

=== Generating Sample Texts ===

Prompt: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics! 
Temperature: 1.0
Generated: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics! treatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreatedtreated

Prompt: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them. 
Temperature: 1.0
Generated: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them.    decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel decel

Prompt: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking. 
Temperature: 1.0
Generated: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking.   Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote Promote

Prompt: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles. 
Temperature: 1.0
Generated: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles.        transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted transplanted

Prompt: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. 
Temperature: 1.0
Generated: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. configurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfigurationconfiguration

=== End of Samples ===

Step 5510, Loss: 5.6417, Scaled Loss: 0.7052, LR: 2.94e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 07:10:15 
Step 5520, Loss: 5.6886, Scaled Loss: 0.7111, LR: 2.96e-05, Accumulation Step: 1/8, Current Time: 2025-02-14 07:25:34 
Step 5530, Loss: 5.7251, Scaled Loss: 0.7156, LR: 2.96e-05, Accumulation Step: 3/8, Current Time: 2025-02-14 07:40:15 
Step 5540, Loss: 5.6050, Scaled Loss: 0.7006, LR: 2.97e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 07:54:27 
Step 5550, Loss: 5.6300, Scaled Loss: 0.7038, LR: 2.98e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 08:08:53 
Step 5560, Loss: 5.6084, Scaled Loss: 0.7010, LR: 2.99e-05, Accumulation Step: 1/8, Current Time: 2025-02-14 08:23:40 
Step 5570, Loss: 5.6372, Scaled Loss: 0.7046, LR: 3.00e-05, Accumulation Step: 3/8, Current Time: 2025-02-14 08:37:52 
Step 5580, Loss: 5.6288, Scaled Loss: 0.7036, LR: 3.00e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 08:50:59 
Step 5590, Loss: 5.6948, Scaled Loss: 0.7118, LR: 3.01e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 09:04:26 
Step 5600, Loss: 5.6180, Scaled Loss: 0.7023, LR: 3.02e-05, Accumulation Step: 1/8, Current Time: 2025-02-14 09:18:27 
Step 5610, Loss: 5.6890, Scaled Loss: 0.7111, LR: 3.03e-05, Accumulation Step: 3/8, Current Time: 2025-02-14 09:33:50 
Step 5620, Loss: 5.6223, Scaled Loss: 0.7028, LR: 3.04e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 09:47:41 
Step 5630, Loss: 5.6374, Scaled Loss: 0.7047, LR: 3.04e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 10:01:06 
Step 5640, Loss: 5.6125, Scaled Loss: 0.7016, LR: 3.06e-05, Accumulation Step: 1/8, Current Time: 2025-02-14 10:14:54 
Step 5650, Loss: 5.6234, Scaled Loss: 0.7029, LR: 3.06e-05, Accumulation Step: 3/8, Current Time: 2025-02-14 10:27:05 
Step 5660, Loss: 5.6365, Scaled Loss: 0.7046, LR: 3.07e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 10:40:33 
Step 5670, Loss: 5.5803, Scaled Loss: 0.6975, LR: 3.08e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 10:53:27 
Step 5680, Loss: 5.6623, Scaled Loss: 0.7078, LR: 3.09e-05, Accumulation Step: 1/8, Current Time: 2025-02-14 11:06:40 
Step 5690, Loss: 5.6350, Scaled Loss: 0.7044, LR: 3.10e-05, Accumulation Step: 3/8, Current Time: 2025-02-14 11:19:34 
Step 5700, Loss: 5.6057, Scaled Loss: 0.7007, LR: 3.10e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 11:32:37 
Step 5710, Loss: 5.5532, Scaled Loss: 0.6941, LR: 3.11e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 20:01:37 
Step 5720, Loss: 5.6408, Scaled Loss: 0.7051, LR: 3.12e-05, Accumulation Step: 1/8, Current Time: 2025-02-14 20:16:00 
Step 5730, Loss: 5.6463, Scaled Loss: 0.7058, LR: 3.13e-05, Accumulation Step: 3/8, Current Time: 2025-02-14 20:28:58 
Step 5740, Loss: 5.5984, Scaled Loss: 0.6998, LR: 3.14e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 20:44:07 
Step 5750, Loss: 5.6780, Scaled Loss: 0.7098, LR: 3.15e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 20:58:24 
Step 5760, Loss: 5.5862, Scaled Loss: 0.6983, LR: 3.16e-05, Accumulation Step: 1/8, Current Time: 2025-02-14 21:12:19 
Step 5770, Loss: 5.5877, Scaled Loss: 0.6985, LR: 3.17e-05, Accumulation Step: 3/8, Current Time: 2025-02-14 21:25:14 
Step 5780, Loss: 5.6114, Scaled Loss: 0.7014, LR: 3.17e-05, Accumulation Step: 5/8, Current Time: 2025-02-14 21:37:54 
Step 5790, Loss: 5.6717, Scaled Loss: 0.7090, LR: 3.18e-05, Accumulation Step: 7/8, Current Time: 2025-02-14 21:53:12 
Step 5800, Loss: 5.5819, Scaled Loss: 0.6977, LR: 3.19e-05, Accumulation Step: 1/8, Current Time: 2025-02-15 01:30:13 
/opt/anaconda3/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Tracking run with wandb version 0.19.4
wandb: W&B syncing is set to `offline` in this directory.  
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.

Model Statistics:
Total Parameters: 688,702,098
Model Size: 2627.19 MB
Device: mps
Batch Size: 4
Accumulation Steps: 8
Sequence Length: 256
Learning Rate: 0.0001
--------------------------------------------------

Epoch 1/5
Step 5800, Loss: 5.5869, Scaled Loss: 0.6984, LR: 3.19e-05, Accumulation Step: 1/8, Current Time: 2025-02-15 13:37:44 
Step 5810, Loss: 5.5951, Scaled Loss: 0.6994, LR: 3.20e-05, Accumulation Step: 3/8, Current Time: 2025-02-15 13:54:25 
Step 5820, Loss: 5.5381, Scaled Loss: 0.6923, LR: 3.21e-05, Accumulation Step: 5/8, Current Time: 2025-02-15 14:08:41 
Step 5830, Loss: 5.6077, Scaled Loss: 0.7010, LR: 3.21e-05, Accumulation Step: 7/8, Current Time: 2025-02-15 14:24:31 
Step 5840, Loss: 5.5949, Scaled Loss: 0.6994, LR: 3.23e-05, Accumulation Step: 1/8, Current Time: 2025-02-15 14:42:00 
Step 5850, Loss: 5.5807, Scaled Loss: 0.6976, LR: 3.23e-05, Accumulation Step: 3/8, Current Time: 2025-02-15 14:58:13 
Step 5860, Loss: 5.5415, Scaled Loss: 0.6927, LR: 3.24e-05, Accumulation Step: 5/8, Current Time: 2025-02-15 15:14:36 
Step 5870, Loss: 5.5444, Scaled Loss: 0.6930, LR: 3.25e-05, Accumulation Step: 7/8, Current Time: 2025-02-15 15:32:06 
Step 5880, Loss: 5.5700, Scaled Loss: 0.6962, LR: 3.26e-05, Accumulation Step: 1/8, Current Time: 2025-02-15 15:51:19 
Step 5890, Loss: 5.5512, Scaled Loss: 0.6939, LR: 3.27e-05, Accumulation Step: 3/8, Current Time: 2025-02-15 16:08:57 
Step 5900, Loss: 5.5296, Scaled Loss: 0.6912, LR: 3.28e-05, Accumulation Step: 5/8, Current Time: 2025-02-15 16:26:06 
Step 5910, Loss: 5.5585, Scaled Loss: 0.6948, LR: 3.28e-05, Accumulation Step: 7/8, Current Time: 2025-02-15 16:44:35 
Step 5920, Loss: 5.5517, Scaled Loss: 0.6940, LR: 3.30e-05, Accumulation Step: 1/8, Current Time: 2025-02-15 17:02:15 
Step 5930, Loss: 5.5496, Scaled Loss: 0.6937, LR: 3.30e-05, Accumulation Step: 3/8, Current Time: 2025-02-15 17:18:41 
Step 5940, Loss: 5.5502, Scaled Loss: 0.6938, LR: 3.31e-05, Accumulation Step: 5/8, Current Time: 2025-02-15 17:35:02 
Step 5950, Loss: 5.5497, Scaled Loss: 0.6937, LR: 3.32e-05, Accumulation Step: 7/8, Current Time: 2025-02-15 17:51:14 
Step 5960, Loss: 5.5016, Scaled Loss: 0.6877, LR: 3.33e-05, Accumulation Step: 1/8, Current Time: 2025-02-15 18:08:00 
Step 5970, Loss: 5.5552, Scaled Loss: 0.6944, LR: 3.34e-05, Accumulation Step: 3/8, Current Time: 2025-02-15 18:23:13 
Step 5980, Loss: 5.5261, Scaled Loss: 0.6908, LR: 3.34e-05, Accumulation Step: 5/8, Current Time: 2025-02-15 18:38:07 
Step 5990, Loss: 5.4962, Scaled Loss: 0.6870, LR: 3.35e-05, Accumulation Step: 7/8, Current Time: 2025-02-15 18:54:39 
Step 6000, Loss: 5.5135, Scaled Loss: 0.6892, LR: 3.37e-05, Accumulation Step: 1/8, Current Time: 2025-02-15 19:11:42 

=== Generating Sample Texts ===

Prompt: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics! 
Temperature: 1.0
Generated: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics!   euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph

Prompt: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them. 
Temperature: 1.0
Generated: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them.  preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations preparations

Prompt: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking. 
Temperature: 1.0
Generated: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking.  olybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenumolybdenum

Prompt: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles. 
Temperature: 1.0
Generated: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles. ellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellation

Prompt: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. 
Temperature: 1.0
Generated: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now.  reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed

=== End of Samples ===

Step 6010, Loss: 5.5947, Scaled Loss: 0.6993, LR: 3.37e-05, Accumulation Step: 3/8, Current Time: 2025-02-15 19:45:36 
Step 6020, Loss: 5.5338, Scaled Loss: 0.6917, LR: 3.38e-05, Accumulation Step: 5/8, Current Time: 2025-02-15 20:02:31 
Step 6030, Loss: 5.4528, Scaled Loss: 0.6816, LR: 3.39e-05, Accumulation Step: 7/8, Current Time: 2025-02-15 20:19:14 
Step 6040, Loss: 5.4484, Scaled Loss: 0.6810, LR: 3.40e-05, Accumulation Step: 1/8, Current Time: 2025-02-15 20:36:06 
Step 6050, Loss: 5.5073, Scaled Loss: 0.6884, LR: 3.41e-05, Accumulation Step: 3/8, Current Time: 2025-02-15 20:51:12 
Step 6060, Loss: 5.4799, Scaled Loss: 0.6850, LR: 3.41e-05, Accumulation Step: 5/8, Current Time: 2025-02-15 21:07:48 
Step 6070, Loss: 5.4446, Scaled Loss: 0.6806, LR: 3.42e-05, Accumulation Step: 7/8, Current Time: 2025-02-15 21:23:16 
Step 6080, Loss: 5.4694, Scaled Loss: 0.6837, LR: 3.44e-05, Accumulation Step: 1/8, Current Time: 2025-02-15 21:40:57 
Step 6090, Loss: 5.4966, Scaled Loss: 0.6871, LR: 3.44e-05, Accumulation Step: 3/8, Current Time: 2025-02-15 21:58:38 
Step 6100, Loss: 5.5090, Scaled Loss: 0.6886, LR: 3.45e-05, Accumulation Step: 5/8, Current Time: 2025-02-15 22:16:13 
Step 6110, Loss: 5.4893, Scaled Loss: 0.6862, LR: 3.46e-05, Accumulation Step: 7/8, Current Time: 2025-02-15 22:33:24 
Step 6120, Loss: 5.4581, Scaled Loss: 0.6823, LR: 3.47e-05, Accumulation Step: 1/8, Current Time: 2025-02-15 22:50:36 
Step 6130, Loss: 5.4725, Scaled Loss: 0.6841, LR: 3.48e-05, Accumulation Step: 3/8, Current Time: 2025-02-15 23:07:37 
Step 6140, Loss: 5.4484, Scaled Loss: 0.6810, LR: 3.48e-05, Accumulation Step: 5/8, Current Time: 2025-02-15 23:22:43 
Step 6150, Loss: 5.4508, Scaled Loss: 0.6813, LR: 3.49e-05, Accumulation Step: 7/8, Current Time: 2025-02-15 23:39:06 
Step 6160, Loss: 5.4457, Scaled Loss: 0.6807, LR: 3.51e-05, Accumulation Step: 1/8, Current Time: 2025-02-15 23:56:15 
Step 6170, Loss: 5.4287, Scaled Loss: 0.6786, LR: 3.51e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 00:11:38 
Step 6180, Loss: 5.4270, Scaled Loss: 0.6784, LR: 3.52e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 00:27:27 
Step 6190, Loss: 5.4248, Scaled Loss: 0.6781, LR: 3.53e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 00:44:28 
Step 6200, Loss: 5.4679, Scaled Loss: 0.6835, LR: 3.54e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 01:02:41 
Step 6210, Loss: 5.4196, Scaled Loss: 0.6775, LR: 3.55e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 01:18:55 
Step 6220, Loss: 5.4352, Scaled Loss: 0.6794, LR: 3.56e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 01:36:17 
Step 6230, Loss: 5.4513, Scaled Loss: 0.6814, LR: 3.56e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 01:55:00 
Step 6240, Loss: 5.3896, Scaled Loss: 0.6737, LR: 3.58e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 02:12:44 
Step 6250, Loss: 5.4137, Scaled Loss: 0.6767, LR: 3.58e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 02:30:33 
Step 6260, Loss: 5.3778, Scaled Loss: 0.6722, LR: 3.59e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 02:48:28 
Step 6270, Loss: 5.4043, Scaled Loss: 0.6755, LR: 3.60e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 03:04:22 
Step 6280, Loss: 5.4026, Scaled Loss: 0.6753, LR: 3.61e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 03:22:26 
Step 6290, Loss: 5.4445, Scaled Loss: 0.6806, LR: 3.62e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 03:37:28 
Step 6300, Loss: 5.4282, Scaled Loss: 0.6785, LR: 3.63e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 03:53:44 
Step 6310, Loss: 5.4287, Scaled Loss: 0.6786, LR: 3.63e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 04:13:12 
Step 6320, Loss: 5.3915, Scaled Loss: 0.6739, LR: 3.65e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 04:32:17 
Step 6330, Loss: 5.3960, Scaled Loss: 0.6745, LR: 3.66e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 04:49:32 
Step 6340, Loss: 5.3672, Scaled Loss: 0.6709, LR: 3.66e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 05:05:41 
Step 6350, Loss: 5.3777, Scaled Loss: 0.6722, LR: 3.67e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 05:22:17 
Step 6360, Loss: 5.3647, Scaled Loss: 0.6706, LR: 3.68e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 05:39:47 
Step 6370, Loss: 5.3396, Scaled Loss: 0.6675, LR: 3.69e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 05:58:11 
Step 6380, Loss: 5.4115, Scaled Loss: 0.6764, LR: 3.70e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 06:15:17 
Step 6390, Loss: 5.3661, Scaled Loss: 0.6708, LR: 3.71e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 06:33:48 
Step 6400, Loss: 5.3728, Scaled Loss: 0.6716, LR: 3.72e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 06:52:20 
Step 6410, Loss: 5.3188, Scaled Loss: 0.6649, LR: 3.73e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 07:10:07 
Step 6420, Loss: 5.4200, Scaled Loss: 0.6775, LR: 3.73e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 08:05:40 
Step 6430, Loss: 5.3790, Scaled Loss: 0.6724, LR: 3.74e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 09:07:09 
Step 6440, Loss: 5.3429, Scaled Loss: 0.6679, LR: 3.76e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 09:36:05 
Step 6450, Loss: 5.4213, Scaled Loss: 0.6777, LR: 3.76e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 09:58:22 
Step 6460, Loss: 5.3595, Scaled Loss: 0.6699, LR: 3.77e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 10:20:25 
Step 6470, Loss: 5.3853, Scaled Loss: 0.6732, LR: 3.78e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 10:39:30 
Step 6480, Loss: 5.3356, Scaled Loss: 0.6669, LR: 3.79e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 11:00:29 
Step 6490, Loss: 5.3929, Scaled Loss: 0.6741, LR: 3.80e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 11:21:25 
Step 6500, Loss: 5.3369, Scaled Loss: 0.6671, LR: 3.81e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 11:39:17 

=== Generating Sample Texts ===

Prompt: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics! 
Temperature: 1.0
Generated: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics!                                                             atroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatroatro

Prompt: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them. 
Temperature: 1.0
Generated: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them.                                                                                     ellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellation

Prompt: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking. 
Temperature: 1.0
Generated: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking.                                                                                                     

Prompt: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles. 
Temperature: 1.0
Generated: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles.                            euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph euph

Prompt: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. 
Temperature: 1.0
Generated: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now.                                                                                                     

=== End of Samples ===

Step 6510, Loss: 5.3197, Scaled Loss: 0.6650, LR: 3.81e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 12:12:21 
Step 6520, Loss: 5.3761, Scaled Loss: 0.6720, LR: 3.83e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 12:31:44 
Step 6530, Loss: 5.2961, Scaled Loss: 0.6620, LR: 3.83e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 12:49:33 
Step 6540, Loss: 5.3081, Scaled Loss: 0.6635, LR: 3.84e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 13:08:14 
Step 6550, Loss: 5.3246, Scaled Loss: 0.6656, LR: 3.85e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 13:25:43 
Step 6560, Loss: 5.2970, Scaled Loss: 0.6621, LR: 3.86e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 13:44:58 
Step 6570, Loss: 5.3361, Scaled Loss: 0.6670, LR: 3.87e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 14:05:00 
Step 6580, Loss: 5.3301, Scaled Loss: 0.6663, LR: 3.88e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 14:22:10 
Step 6590, Loss: 5.3198, Scaled Loss: 0.6650, LR: 3.89e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 14:39:25 
Step 6600, Loss: 5.2662, Scaled Loss: 0.6583, LR: 3.90e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 14:58:05 
Step 6610, Loss: 5.2920, Scaled Loss: 0.6615, LR: 3.91e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 15:17:43 
Step 6620, Loss: 5.3251, Scaled Loss: 0.6656, LR: 3.91e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 15:35:52 
Step 6630, Loss: 5.2872, Scaled Loss: 0.6609, LR: 3.92e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 15:52:54 
Step 6640, Loss: 5.2759, Scaled Loss: 0.6595, LR: 3.94e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 16:10:47 
Step 6650, Loss: 5.2704, Scaled Loss: 0.6588, LR: 3.94e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 16:28:22 
Step 6660, Loss: 5.2888, Scaled Loss: 0.6611, LR: 3.95e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 16:46:21 
Step 6670, Loss: 5.2652, Scaled Loss: 0.6581, LR: 3.96e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 17:04:32 
Step 6680, Loss: 5.3007, Scaled Loss: 0.6626, LR: 3.97e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 17:21:36 
Step 6690, Loss: 5.2482, Scaled Loss: 0.6560, LR: 3.98e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 17:38:59 
Step 6700, Loss: 5.2770, Scaled Loss: 0.6596, LR: 3.99e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 17:56:27 
Step 6710, Loss: 5.1992, Scaled Loss: 0.6499, LR: 3.99e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 18:15:42 
Step 6720, Loss: 5.1999, Scaled Loss: 0.6500, LR: 4.01e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 18:34:20 
Step 6730, Loss: 5.2253, Scaled Loss: 0.6532, LR: 4.02e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 18:52:35 
Step 6740, Loss: 5.2151, Scaled Loss: 0.6519, LR: 4.02e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 19:09:29 
Step 6750, Loss: 5.2477, Scaled Loss: 0.6560, LR: 4.03e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 19:26:34 
Step 6760, Loss: 5.2302, Scaled Loss: 0.6538, LR: 4.05e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 19:44:38 
Step 6770, Loss: 5.2068, Scaled Loss: 0.6509, LR: 4.05e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 20:00:58 
Step 6780, Loss: 5.2102, Scaled Loss: 0.6513, LR: 4.06e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 20:19:27 
Step 6790, Loss: 5.2356, Scaled Loss: 0.6544, LR: 4.07e-05, Accumulation Step: 7/8, Current Time: 2025-02-16 20:36:57 
Step 6800, Loss: 5.1928, Scaled Loss: 0.6491, LR: 4.08e-05, Accumulation Step: 1/8, Current Time: 2025-02-16 20:55:29 
Step 6810, Loss: 5.2545, Scaled Loss: 0.6568, LR: 4.09e-05, Accumulation Step: 3/8, Current Time: 2025-02-16 21:16:43 
Step 6820, Loss: 5.1907, Scaled Loss: 0.6488, LR: 4.10e-05, Accumulation Step: 5/8, Current Time: 2025-02-16 21:35:37 
Step 6830, Loss: 5.1894, Scaled Loss: 0.6487, LR: 4.10e-05, Accumulation Step: 7/8, Current Time: 2025-02-17 06:57:46 
Step 6840, Loss: 5.1720, Scaled Loss: 0.6465, LR: 4.12e-05, Accumulation Step: 1/8, Current Time: 2025-02-17 08:32:03 
Step 6850, Loss: 5.1429, Scaled Loss: 0.6429, LR: 4.13e-05, Accumulation Step: 3/8, Current Time: 2025-02-17 11:10:32 
Step 6860, Loss: 5.2116, Scaled Loss: 0.6514, LR: 4.13e-05, Accumulation Step: 5/8, Current Time: 2025-02-17 11:37:37 
Step 6870, Loss: 5.1730, Scaled Loss: 0.6466, LR: 4.14e-05, Accumulation Step: 7/8, Current Time: 2025-02-17 13:04:47 
Step 6880, Loss: 5.2296, Scaled Loss: 0.6537, LR: 4.16e-05, Accumulation Step: 1/8, Current Time: 2025-02-17 18:19:09 
Step 6890, Loss: 5.1739, Scaled Loss: 0.6467, LR: 4.16e-05, Accumulation Step: 3/8, Current Time: 2025-02-17 20:58:48 
Step 6900, Loss: 5.1858, Scaled Loss: 0.6482, LR: 4.17e-05, Accumulation Step: 5/8, Current Time: 2025-02-17 22:07:23 
Step 6910, Loss: 5.2028, Scaled Loss: 0.6503, LR: 4.18e-05, Accumulation Step: 7/8, Current Time: 2025-02-17 22:27:27 
Step 6920, Loss: 5.2105, Scaled Loss: 0.6513, LR: 4.19e-05, Accumulation Step: 1/8, Current Time: 2025-02-17 22:45:56 
Step 6930, Loss: 5.2001, Scaled Loss: 0.6500, LR: 4.20e-05, Accumulation Step: 3/8, Current Time: 2025-02-17 23:02:46 
Step 6940, Loss: 5.1644, Scaled Loss: 0.6455, LR: 4.21e-05, Accumulation Step: 5/8, Current Time: 2025-02-17 23:20:53 
Step 6950, Loss: 5.1788, Scaled Loss: 0.6474, LR: 4.22e-05, Accumulation Step: 7/8, Current Time: 2025-02-17 23:38:24 
Step 6960, Loss: 5.1756, Scaled Loss: 0.6469, LR: 4.23e-05, Accumulation Step: 1/8, Current Time: 2025-02-17 23:56:47 
Step 6970, Loss: 5.2314, Scaled Loss: 0.6539, LR: 4.24e-05, Accumulation Step: 3/8, Current Time: 2025-02-18 00:15:02 
Step 6980, Loss: 5.1173, Scaled Loss: 0.6397, LR: 4.24e-05, Accumulation Step: 5/8, Current Time: 2025-02-18 00:32:04 
Step 6990, Loss: 5.1994, Scaled Loss: 0.6499, LR: 4.25e-05, Accumulation Step: 7/8, Current Time: 2025-02-18 00:49:40 
Step 7000, Loss: 5.1361, Scaled Loss: 0.6420, LR: 4.27e-05, Accumulation Step: 1/8, Current Time: 2025-02-18 01:07:40 

=== Generating Sample Texts ===

Prompt: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics! 
Temperature: 1.0
Generated: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics!   carboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarboncarbon

Prompt: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them. 
Temperature: 1.0
Generated: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them.     Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions Instructions

Prompt: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking. 
Temperature: 1.0
Generated: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking.  iciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciaryiciary

Prompt: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles. 
Temperature: 1.0
Generated: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles.   maxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmaxmax

Prompt: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. 
Temperature: 1.0
Generated: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now.    nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides nucleotides

=== End of Samples ===

Step 7010, Loss: 5.0996, Scaled Loss: 0.6375, LR: 4.27e-05, Accumulation Step: 3/8, Current Time: 2025-02-18 01:40:21 
Step 7020, Loss: 5.1128, Scaled Loss: 0.6391, LR: 4.28e-05, Accumulation Step: 5/8, Current Time: 2025-02-18 01:59:20 
Step 7030, Loss: 5.1559, Scaled Loss: 0.6445, LR: 4.29e-05, Accumulation Step: 7/8, Current Time: 2025-02-18 02:17:16 
Step 7040, Loss: 5.1408, Scaled Loss: 0.6426, LR: 4.30e-05, Accumulation Step: 1/8, Current Time: 2025-02-18 02:35:01 
Step 7050, Loss: 5.1270, Scaled Loss: 0.6409, LR: 4.31e-05, Accumulation Step: 3/8, Current Time: 2025-02-18 02:52:04 
Step 7060, Loss: 5.1377, Scaled Loss: 0.6422, LR: 4.32e-05, Accumulation Step: 5/8, Current Time: 2025-02-18 03:09:42 
Step 7070, Loss: 5.0988, Scaled Loss: 0.6374, LR: 4.33e-05, Accumulation Step: 7/8, Current Time: 2025-02-18 03:26:52 
Step 7080, Loss: 5.0711, Scaled Loss: 0.6339, LR: 4.34e-05, Accumulation Step: 1/8, Current Time: 2025-02-18 03:45:30 
Step 7090, Loss: 5.1446, Scaled Loss: 0.6431, LR: 4.35e-05, Accumulation Step: 3/8, Current Time: 2025-02-18 04:03:19 
Step 7100, Loss: 5.0830, Scaled Loss: 0.6354, LR: 4.36e-05, Accumulation Step: 5/8, Current Time: 2025-02-18 04:20:19 
Step 7110, Loss: 5.0918, Scaled Loss: 0.6365, LR: 4.36e-05, Accumulation Step: 7/8, Current Time: 2025-02-18 04:39:17 
Step 7120, Loss: 5.1098, Scaled Loss: 0.6387, LR: 4.38e-05, Accumulation Step: 1/8, Current Time: 2025-02-18 04:58:08 
Step 7130, Loss: 5.0892, Scaled Loss: 0.6362, LR: 4.39e-05, Accumulation Step: 3/8, Current Time: 2025-02-18 05:16:05 
Step 7140, Loss: 5.0938, Scaled Loss: 0.6367, LR: 4.39e-05, Accumulation Step: 5/8, Current Time: 2025-02-18 05:33:07 
Step 7150, Loss: 5.0133, Scaled Loss: 0.6267, LR: 4.40e-05, Accumulation Step: 7/8, Current Time: 2025-02-18 05:50:24 
Step 7160, Loss: 5.0864, Scaled Loss: 0.6358, LR: 4.42e-05, Accumulation Step: 1/8, Current Time: 2025-02-18 06:10:02 
Step 7170, Loss: 5.0537, Scaled Loss: 0.6317, LR: 4.42e-05, Accumulation Step: 3/8, Current Time: 2025-02-18 06:29:59 
Step 7180, Loss: 5.0870, Scaled Loss: 0.6359, LR: 4.43e-05, Accumulation Step: 5/8, Current Time: 2025-02-18 06:54:26 
Step 7190, Loss: 5.0650, Scaled Loss: 0.6331, LR: 4.44e-05, Accumulation Step: 7/8, Current Time: 2025-02-18 07:17:28 
Step 7200, Loss: 5.1010, Scaled Loss: 0.6376, LR: 4.45e-05, Accumulation Step: 1/8, Current Time: 2025-02-18 07:37:09 
Step 7210, Loss: 5.0520, Scaled Loss: 0.6315, LR: 4.46e-05, Accumulation Step: 3/8, Current Time: 2025-02-18 07:58:45 
Step 7220, Loss: 5.0560, Scaled Loss: 0.6320, LR: 4.47e-05, Accumulation Step: 5/8, Current Time: 2025-02-18 08:17:41 
Step 7230, Loss: 5.0849, Scaled Loss: 0.6356, LR: 4.47e-05, Accumulation Step: 7/8, Current Time: 2025-02-18 08:36:30 
Step 7240, Loss: 4.9994, Scaled Loss: 0.6249, LR: 4.49e-05, Accumulation Step: 1/8, Current Time: 2025-02-18 08:54:15 
Step 7250, Loss: 5.0591, Scaled Loss: 0.6324, LR: 4.50e-05, Accumulation Step: 3/8, Current Time: 2025-02-18 09:11:35 
Step 7260, Loss: 5.0583, Scaled Loss: 0.6323, LR: 4.50e-05, Accumulation Step: 5/8, Current Time: 2025-02-18 09:32:58 
Step 7270, Loss: 5.0156, Scaled Loss: 0.6270, LR: 4.51e-05, Accumulation Step: 7/8, Current Time: 2025-02-18 09:54:20 
Step 7280, Loss: 5.0292, Scaled Loss: 0.6286, LR: 4.53e-05, Accumulation Step: 1/8, Current Time: 2025-02-18 11:07:01 
Step 7290, Loss: 4.9821, Scaled Loss: 0.6228, LR: 4.53e-05, Accumulation Step: 3/8, Current Time: 2025-02-18 11:39:53 
Step 7300, Loss: 5.0042, Scaled Loss: 0.6255, LR: 4.54e-05, Accumulation Step: 5/8, Current Time: 2025-02-18 11:58:02 
Step 7310, Loss: 4.9918, Scaled Loss: 0.6240, LR: 4.55e-05, Accumulation Step: 7/8, Current Time: 2025-02-18 12:18:00 
Step 7320, Loss: 4.9841, Scaled Loss: 0.6230, LR: 4.56e-05, Accumulation Step: 1/8, Current Time: 2025-02-18 12:53:36 
Step 7330, Loss: 4.9736, Scaled Loss: 0.6217, LR: 4.57e-05, Accumulation Step: 3/8, Current Time: 2025-02-18 13:11:31 
Step 7340, Loss: 4.9479, Scaled Loss: 0.6185, LR: 4.58e-05, Accumulation Step: 5/8, Current Time: 2025-02-18 13:30:46 
Step 7350, Loss: 4.9698, Scaled Loss: 0.6212, LR: 4.59e-05, Accumulation Step: 7/8, Current Time: 2025-02-18 13:48:16 
Step 7360, Loss: 5.0125, Scaled Loss: 0.6266, LR: 4.60e-05, Accumulation Step: 1/8, Current Time: 2025-02-18 14:06:30 
Step 7370, Loss: 5.0096, Scaled Loss: 0.6262, LR: 4.61e-05, Accumulation Step: 3/8, Current Time: 2025-02-18 14:27:55 
Step 7380, Loss: 5.0229, Scaled Loss: 0.6279, LR: 4.62e-05, Accumulation Step: 5/8, Current Time: 2025-02-18 15:34:02 
Step 7390, Loss: 4.9110, Scaled Loss: 0.6139, LR: 4.62e-05, Accumulation Step: 7/8, Current Time: 2025-02-18 20:52:43 
Step 7400, Loss: 4.8881, Scaled Loss: 0.6110, LR: 4.64e-05, Accumulation Step: 1/8, Current Time: 2025-02-19 06:38:11 
Step 7410, Loss: 4.8966, Scaled Loss: 0.6121, LR: 4.65e-05, Accumulation Step: 3/8, Current Time: 2025-02-19 07:25:29 
Step 7420, Loss: 4.9506, Scaled Loss: 0.6188, LR: 4.65e-05, Accumulation Step: 5/8, Current Time: 2025-02-19 07:42:58 
Step 7430, Loss: 4.8753, Scaled Loss: 0.6094, LR: 4.66e-05, Accumulation Step: 7/8, Current Time: 2025-02-19 08:00:47 
Step 7440, Loss: 4.8479, Scaled Loss: 0.6060, LR: 4.68e-05, Accumulation Step: 1/8, Current Time: 2025-02-19 08:46:11 
Step 7450, Loss: 4.9169, Scaled Loss: 0.6146, LR: 4.68e-05, Accumulation Step: 3/8, Current Time: 2025-02-20 06:15:59 
Step 7460, Loss: 4.8683, Scaled Loss: 0.6085, LR: 4.69e-05, Accumulation Step: 5/8, Current Time: 2025-02-20 06:37:50 
Step 7470, Loss: 4.9249, Scaled Loss: 0.6156, LR: 4.70e-05, Accumulation Step: 7/8, Current Time: 2025-02-20 06:57:24 
Step 7480, Loss: 4.9114, Scaled Loss: 0.6139, LR: 4.71e-05, Accumulation Step: 1/8, Current Time: 2025-02-20 15:56:56 
Step 7490, Loss: 4.9088, Scaled Loss: 0.6136, LR: 4.72e-05, Accumulation Step: 3/8, Current Time: 2025-02-20 16:45:40 
Step 7500, Loss: 4.8226, Scaled Loss: 0.6028, LR: 4.73e-05, Accumulation Step: 5/8, Current Time: 2025-02-20 17:03:45 

=== Generating Sample Texts ===

Prompt: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics! 
Temperature: 1.0
Generated: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics!     authoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthored

Prompt: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them. 
Temperature: 1.0
Generated: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them.  reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed reviewed

Prompt: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking. 
Temperature: 1.0
Generated: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking.    psipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsipsi

Prompt: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles. 
Temperature: 1.0
Generated: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles.   authoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthoredauthored

Prompt: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. 
Temperature: 1.0
Generated: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now.           polarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolarpolar

=== End of Samples ===

Step 7510, Loss: 4.9223, Scaled Loss: 0.6153, LR: 4.74e-05, Accumulation Step: 7/8, Current Time: 2025-02-20 17:39:23 
Step 7520, Loss: 4.8495, Scaled Loss: 0.6062, LR: 4.75e-05, Accumulation Step: 1/8, Current Time: 2025-02-20 17:58:39 
Step 7530, Loss: 4.8465, Scaled Loss: 0.6058, LR: 4.76e-05, Accumulation Step: 3/8, Current Time: 2025-02-20 18:16:04 
Step 7540, Loss: 4.9232, Scaled Loss: 0.6154, LR: 4.77e-05, Accumulation Step: 5/8, Current Time: 2025-02-20 18:35:08 
Step 7550, Loss: 4.9004, Scaled Loss: 0.6126, LR: 4.77e-05, Accumulation Step: 7/8, Current Time: 2025-02-20 19:07:40 
Step 7560, Loss: 4.8349, Scaled Loss: 0.6044, LR: 4.79e-05, Accumulation Step: 1/8, Current Time: 2025-02-20 19:25:08 
Step 7570, Loss: 4.8280, Scaled Loss: 0.6035, LR: 4.80e-05, Accumulation Step: 3/8, Current Time: 2025-02-20 20:07:10 
Step 7580, Loss: 4.8298, Scaled Loss: 0.6037, LR: 4.80e-05, Accumulation Step: 5/8, Current Time: 2025-02-20 20:25:51 
Step 7590, Loss: 4.8385, Scaled Loss: 0.6048, LR: 4.81e-05, Accumulation Step: 7/8, Current Time: 2025-02-20 20:44:59 
Step 7600, Loss: 4.8055, Scaled Loss: 0.6007, LR: 4.83e-05, Accumulation Step: 1/8, Current Time: 2025-02-20 21:06:20 
Step 7610, Loss: 4.8207, Scaled Loss: 0.6026, LR: 4.83e-05, Accumulation Step: 3/8, Current Time: 2025-02-20 21:26:29 
Step 7620, Loss: 4.8073, Scaled Loss: 0.6009, LR: 4.84e-05, Accumulation Step: 5/8, Current Time: 2025-02-20 21:44:29 
Step 7630, Loss: 4.8349, Scaled Loss: 0.6044, LR: 4.85e-05, Accumulation Step: 7/8, Current Time: 2025-02-20 22:01:38 
Step 7640, Loss: 4.7823, Scaled Loss: 0.5978, LR: 4.86e-05, Accumulation Step: 1/8, Current Time: 2025-02-20 22:22:02 
Step 7650, Loss: 4.7887, Scaled Loss: 0.5986, LR: 4.87e-05, Accumulation Step: 3/8, Current Time: 2025-02-20 22:40:44 
Step 7660, Loss: 4.7519, Scaled Loss: 0.5940, LR: 4.88e-05, Accumulation Step: 5/8, Current Time: 2025-02-20 22:59:03 
Step 7670, Loss: 4.8096, Scaled Loss: 0.6012, LR: 4.89e-05, Accumulation Step: 7/8, Current Time: 2025-02-20 23:16:26 
Step 7680, Loss: 4.7381, Scaled Loss: 0.5923, LR: 4.90e-05, Accumulation Step: 1/8, Current Time: 2025-02-20 23:34:53 
Step 7690, Loss: 4.7493, Scaled Loss: 0.5937, LR: 4.91e-05, Accumulation Step: 3/8, Current Time: 2025-02-20 23:52:15 
Step 7700, Loss: 4.7930, Scaled Loss: 0.5991, LR: 4.92e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 00:10:38 
Step 7710, Loss: 4.7393, Scaled Loss: 0.5924, LR: 4.92e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 00:29:48 
Step 7720, Loss: 4.7476, Scaled Loss: 0.5935, LR: 4.94e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 00:47:50 
Step 7730, Loss: 4.7710, Scaled Loss: 0.5964, LR: 4.95e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 01:06:03 
Step 7740, Loss: 4.7548, Scaled Loss: 0.5943, LR: 4.95e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 01:24:10 
Step 7750, Loss: 4.7580, Scaled Loss: 0.5947, LR: 4.96e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 01:42:25 
Step 7760, Loss: 4.7076, Scaled Loss: 0.5885, LR: 4.98e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 02:00:14 
Step 7770, Loss: 4.7806, Scaled Loss: 0.5976, LR: 4.99e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 02:18:39 
Step 7780, Loss: 4.6937, Scaled Loss: 0.5867, LR: 4.99e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 02:36:23 
Step 7790, Loss: 4.7267, Scaled Loss: 0.5908, LR: 5.00e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 02:53:41 
Step 7800, Loss: 4.7206, Scaled Loss: 0.5901, LR: 5.02e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 03:11:11 
Step 7810, Loss: 4.6813, Scaled Loss: 0.5852, LR: 5.02e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 03:29:15 
Step 7820, Loss: 4.7346, Scaled Loss: 0.5918, LR: 5.03e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 03:47:17 
Step 7830, Loss: 4.6957, Scaled Loss: 0.5870, LR: 5.04e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 04:05:35 
Step 7840, Loss: 4.7635, Scaled Loss: 0.5954, LR: 5.05e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 04:24:40 
Step 7850, Loss: 4.6743, Scaled Loss: 0.5843, LR: 5.06e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 04:43:31 
Step 7860, Loss: 4.6881, Scaled Loss: 0.5860, LR: 5.07e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 05:01:27 
Step 7870, Loss: 4.6108, Scaled Loss: 0.5763, LR: 5.08e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 05:19:49 
Step 7880, Loss: 4.6101, Scaled Loss: 0.5763, LR: 5.09e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 05:38:54 
Step 7890, Loss: 4.5957, Scaled Loss: 0.5745, LR: 5.10e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 05:58:00 
Step 7900, Loss: 4.6434, Scaled Loss: 0.5804, LR: 5.11e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 06:19:46 
Step 7910, Loss: 4.6623, Scaled Loss: 0.5828, LR: 5.11e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 06:42:02 
Step 7920, Loss: 4.6413, Scaled Loss: 0.5802, LR: 5.13e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 07:05:07 
Step 7930, Loss: 4.6070, Scaled Loss: 0.5759, LR: 5.14e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 07:26:14 
Step 7940, Loss: 4.6426, Scaled Loss: 0.5803, LR: 5.14e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 07:46:03 
Step 7950, Loss: 4.6601, Scaled Loss: 0.5825, LR: 5.15e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 08:04:09 
Step 7960, Loss: 4.6823, Scaled Loss: 0.5853, LR: 5.17e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 08:22:24 
Step 7970, Loss: 4.6275, Scaled Loss: 0.5784, LR: 5.17e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 08:40:14 
Step 7980, Loss: 4.6256, Scaled Loss: 0.5782, LR: 5.18e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 09:01:03 
Step 7990, Loss: 4.6637, Scaled Loss: 0.5830, LR: 5.19e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 09:19:55 
Step 8000, Loss: 4.5616, Scaled Loss: 0.5702, LR: 5.20e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 09:37:28 

=== Generating Sample Texts ===

Prompt: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics! 
Temperature: 1.0
Generated: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics!       oryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryoryory

Prompt: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them. 
Temperature: 1.0
Generated: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them.      vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel vessel

Prompt: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking. 
Temperature: 1.0
Generated: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking.    "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "# "#

Prompt: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles. 
Temperature: 1.0
Generated: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles.         discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete discrete

Prompt: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. 
Temperature: 1.0
Generated: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now.   नननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननननन

=== End of Samples ===

Step 8010, Loss: 4.5878, Scaled Loss: 0.5735, LR: 5.21e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 10:10:54 
Step 8020, Loss: 4.5945, Scaled Loss: 0.5743, LR: 5.22e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 10:31:32 
Step 8030, Loss: 4.5501, Scaled Loss: 0.5688, LR: 5.23e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 10:49:11 
Step 8040, Loss: 4.5773, Scaled Loss: 0.5722, LR: 5.24e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 11:07:42 
Step 8050, Loss: 4.5524, Scaled Loss: 0.5691, LR: 5.25e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 11:52:29 
Step 8060, Loss: 4.5178, Scaled Loss: 0.5647, LR: 5.26e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 12:11:11 
Step 8070, Loss: 4.5590, Scaled Loss: 0.5699, LR: 5.26e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 12:31:27 
Step 8080, Loss: 4.4695, Scaled Loss: 0.5587, LR: 5.28e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 12:51:01 
Step 8090, Loss: 4.5040, Scaled Loss: 0.5630, LR: 5.29e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 13:08:41 
Step 8100, Loss: 4.4820, Scaled Loss: 0.5603, LR: 5.29e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 14:07:43 
Step 8110, Loss: 4.5090, Scaled Loss: 0.5636, LR: 5.30e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 14:27:27 
Step 8120, Loss: 4.5202, Scaled Loss: 0.5650, LR: 5.32e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 14:46:14 
Step 8130, Loss: 4.5100, Scaled Loss: 0.5637, LR: 5.32e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 15:05:58 
Step 8140, Loss: 4.5292, Scaled Loss: 0.5661, LR: 5.33e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 15:24:02 
Step 8150, Loss: 4.4599, Scaled Loss: 0.5575, LR: 5.34e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 15:43:46 
Step 8160, Loss: 4.4480, Scaled Loss: 0.5560, LR: 5.35e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 16:05:59 
Step 8170, Loss: 4.4361, Scaled Loss: 0.5545, LR: 5.36e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 16:46:43 
Step 8180, Loss: 4.4448, Scaled Loss: 0.5556, LR: 5.37e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 17:05:43 
Step 8190, Loss: 4.4587, Scaled Loss: 0.5573, LR: 5.38e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 17:26:54 
Step 8200, Loss: 4.4191, Scaled Loss: 0.5524, LR: 5.39e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 17:49:27 
Step 8210, Loss: 4.4578, Scaled Loss: 0.5572, LR: 5.40e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 18:09:01 
Step 8220, Loss: 4.4319, Scaled Loss: 0.5540, LR: 5.41e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 18:28:03 
Step 8230, Loss: 4.4623, Scaled Loss: 0.5578, LR: 5.41e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 18:46:49 
Step 8240, Loss: 4.4466, Scaled Loss: 0.5558, LR: 5.43e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 19:06:10 
Step 8250, Loss: 4.4076, Scaled Loss: 0.5509, LR: 5.44e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 19:25:26 
Step 8260, Loss: 4.3725, Scaled Loss: 0.5466, LR: 5.45e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 19:45:25 
Step 8270, Loss: 4.4795, Scaled Loss: 0.5599, LR: 5.45e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 20:32:47 
Step 8280, Loss: 4.3735, Scaled Loss: 0.5467, LR: 5.47e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 20:56:21 
Step 8290, Loss: 4.3559, Scaled Loss: 0.5445, LR: 5.48e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 21:16:43 
Step 8300, Loss: 4.4031, Scaled Loss: 0.5504, LR: 5.48e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 21:35:24 
Step 8310, Loss: 4.3798, Scaled Loss: 0.5475, LR: 5.49e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 21:54:42 
Step 8320, Loss: 4.3765, Scaled Loss: 0.5471, LR: 5.51e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 22:13:26 
Step 8330, Loss: 4.3829, Scaled Loss: 0.5479, LR: 5.51e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 22:32:33 
Step 8340, Loss: 4.3342, Scaled Loss: 0.5418, LR: 5.52e-05, Accumulation Step: 5/8, Current Time: 2025-02-21 22:51:20 
Step 8350, Loss: 4.3664, Scaled Loss: 0.5458, LR: 5.53e-05, Accumulation Step: 7/8, Current Time: 2025-02-21 23:09:43 
Step 8360, Loss: 4.3928, Scaled Loss: 0.5491, LR: 5.54e-05, Accumulation Step: 1/8, Current Time: 2025-02-21 23:29:24 
Step 8370, Loss: 4.3434, Scaled Loss: 0.5429, LR: 5.55e-05, Accumulation Step: 3/8, Current Time: 2025-02-21 23:48:38 
Step 8380, Loss: 4.3231, Scaled Loss: 0.5404, LR: 5.56e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 00:07:35 
Step 8390, Loss: 4.3472, Scaled Loss: 0.5434, LR: 5.57e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 00:26:54 
Step 8400, Loss: 4.3374, Scaled Loss: 0.5422, LR: 5.58e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 00:46:45 
Step 8410, Loss: 4.3148, Scaled Loss: 0.5393, LR: 5.59e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 01:05:17 
Step 8420, Loss: 4.2803, Scaled Loss: 0.5350, LR: 5.60e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 01:22:41 
Step 8430, Loss: 4.3148, Scaled Loss: 0.5394, LR: 5.60e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 01:40:53 
Step 8440, Loss: 4.2576, Scaled Loss: 0.5322, LR: 5.62e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 02:00:03 
Step 8450, Loss: 4.2905, Scaled Loss: 0.5363, LR: 5.63e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 02:18:52 
Step 8460, Loss: 4.3542, Scaled Loss: 0.5443, LR: 5.63e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 02:36:18 
Step 8470, Loss: 4.2682, Scaled Loss: 0.5335, LR: 5.64e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 02:54:21 
Step 8480, Loss: 4.2827, Scaled Loss: 0.5353, LR: 5.66e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 03:13:27 
Step 8490, Loss: 4.2276, Scaled Loss: 0.5285, LR: 5.66e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 03:31:48 
Step 8500, Loss: 4.2220, Scaled Loss: 0.5278, LR: 5.67e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 03:50:12 

=== Generating Sample Texts ===

Prompt: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics! 
Temperature: 1.0
Generated: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics!             streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams streams

Prompt: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them. 
Temperature: 1.0
Generated: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them.    ellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellation

Prompt: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking. 
Temperature: 1.0
Generated: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking.                                         fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy fancy

Prompt: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles. 
Temperature: 1.0
Generated: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles.     statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue statue

Prompt: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. 
Temperature: 1.0
Generated: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now.        efficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacyefficacy

=== End of Samples ===

Step 8510, Loss: 4.2453, Scaled Loss: 0.5307, LR: 5.68e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 04:22:00 
Step 8520, Loss: 4.2004, Scaled Loss: 0.5250, LR: 5.69e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 04:40:39 
Step 8530, Loss: 4.2359, Scaled Loss: 0.5295, LR: 5.70e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 04:58:03 
Step 8540, Loss: 4.1865, Scaled Loss: 0.5233, LR: 5.71e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 05:15:10 
Step 8550, Loss: 4.2510, Scaled Loss: 0.5314, LR: 5.72e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 05:32:19 
Step 8560, Loss: 4.2619, Scaled Loss: 0.5327, LR: 5.73e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 05:50:16 
Step 8570, Loss: 4.1841, Scaled Loss: 0.5230, LR: 5.74e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 06:08:04 
Step 8580, Loss: 4.2169, Scaled Loss: 0.5271, LR: 5.75e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 06:27:13 
Step 8590, Loss: 4.1879, Scaled Loss: 0.5235, LR: 5.75e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 06:48:00 
Step 8600, Loss: 4.2122, Scaled Loss: 0.5265, LR: 5.77e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 07:10:57 
Step 8610, Loss: 4.2121, Scaled Loss: 0.5265, LR: 5.78e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 07:33:06 
Step 8620, Loss: 4.1859, Scaled Loss: 0.5232, LR: 5.78e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 07:55:47 
Step 8630, Loss: 4.1873, Scaled Loss: 0.5234, LR: 5.79e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 08:16:38 
Step 8640, Loss: 4.1585, Scaled Loss: 0.5198, LR: 5.81e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 08:35:04 
Step 8650, Loss: 4.1262, Scaled Loss: 0.5158, LR: 5.81e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 08:51:37 
Step 8660, Loss: 4.1705, Scaled Loss: 0.5213, LR: 5.82e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 09:10:53 
Step 8670, Loss: 4.1191, Scaled Loss: 0.5149, LR: 5.83e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 09:28:20 
Step 8680, Loss: 4.1064, Scaled Loss: 0.5133, LR: 5.84e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 09:47:11 
Step 8690, Loss: 4.1257, Scaled Loss: 0.5157, LR: 5.85e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 10:05:33 
Step 8700, Loss: 4.1113, Scaled Loss: 0.5139, LR: 5.86e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 10:23:06 
Step 8710, Loss: 4.1831, Scaled Loss: 0.5229, LR: 5.87e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 10:46:23 
Step 8720, Loss: 4.1409, Scaled Loss: 0.5176, LR: 5.88e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 11:06:22 
Step 8730, Loss: 4.1216, Scaled Loss: 0.5152, LR: 5.89e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 11:25:50 
Step 8740, Loss: 4.1896, Scaled Loss: 0.5237, LR: 5.90e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 11:44:00 
Step 8750, Loss: 4.0889, Scaled Loss: 0.5111, LR: 5.90e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 12:00:15 
Step 8760, Loss: 4.0540, Scaled Loss: 0.5068, LR: 5.92e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 12:17:14 
Step 8770, Loss: 4.0520, Scaled Loss: 0.5065, LR: 5.93e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 12:32:32 
Step 8780, Loss: 4.0755, Scaled Loss: 0.5094, LR: 5.93e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 12:48:29 
Step 8790, Loss: 4.0819, Scaled Loss: 0.5102, LR: 5.94e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 13:03:09 
Step 8800, Loss: 4.0607, Scaled Loss: 0.5076, LR: 5.95e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 13:21:29 
Step 8810, Loss: 4.0097, Scaled Loss: 0.5012, LR: 5.96e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 13:36:03 
Step 8820, Loss: 4.1337, Scaled Loss: 0.5167, LR: 5.97e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 13:51:58 
Step 8830, Loss: 4.0548, Scaled Loss: 0.5068, LR: 5.98e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 14:06:37 
Step 8840, Loss: 4.0156, Scaled Loss: 0.5020, LR: 5.99e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 14:22:48 
Step 8850, Loss: 4.0199, Scaled Loss: 0.5025, LR: 6.00e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 14:40:09 
Step 8860, Loss: 4.0315, Scaled Loss: 0.5039, LR: 6.01e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 14:55:46 
Step 8870, Loss: 4.0019, Scaled Loss: 0.5002, LR: 6.01e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 15:11:37 
Step 8880, Loss: 3.9996, Scaled Loss: 0.4999, LR: 6.03e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 15:29:06 
Step 8890, Loss: 3.9982, Scaled Loss: 0.4998, LR: 6.04e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 15:44:57 
Step 8900, Loss: 3.9622, Scaled Loss: 0.4953, LR: 6.04e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 16:01:28 
Step 8910, Loss: 3.9989, Scaled Loss: 0.4999, LR: 6.05e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 16:18:38 
Step 8920, Loss: 4.0025, Scaled Loss: 0.5003, LR: 6.07e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 16:35:51 
Step 8930, Loss: 3.9808, Scaled Loss: 0.4976, LR: 6.07e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 16:52:11 
Step 8940, Loss: 3.9893, Scaled Loss: 0.4987, LR: 6.08e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 17:09:07 
Step 8950, Loss: 3.9768, Scaled Loss: 0.4971, LR: 6.09e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 17:27:26 
Step 8960, Loss: 3.9300, Scaled Loss: 0.4913, LR: 6.10e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 17:44:47 
Step 8970, Loss: 3.9899, Scaled Loss: 0.4987, LR: 6.11e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 18:02:29 
Step 8980, Loss: 3.9683, Scaled Loss: 0.4960, LR: 6.12e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 18:20:44 
Step 8990, Loss: 3.9857, Scaled Loss: 0.4982, LR: 6.13e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 18:37:29 
Step 9000, Loss: 3.9642, Scaled Loss: 0.4955, LR: 6.14e-05, Accumulation Step: 1/8, Current Time: 2025-02-22 18:55:16 

=== Generating Sample Texts ===

Prompt: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics! 
Temperature: 1.0
Generated: Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics!                from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from from

Prompt: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them. 
Temperature: 1.0
Generated: Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them.                                      ellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellationellation

Prompt: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking. 
Temperature: 1.0
Generated: All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking.                                        plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex plex

Prompt: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles. 
Temperature: 1.0
Generated: There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles.                         icodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicodeicode

Prompt: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. 
Temperature: 1.0
Generated: Are you looking for vegan sandwich recipes? We’ve rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now.                                   orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard orchard

=== End of Samples ===

Step 9010, Loss: 3.9018, Scaled Loss: 0.4877, LR: 6.15e-05, Accumulation Step: 3/8, Current Time: 2025-02-22 19:31:52 
Step 9020, Loss: 3.8823, Scaled Loss: 0.4853, LR: 6.16e-05, Accumulation Step: 5/8, Current Time: 2025-02-22 19:48:53 
Step 9030, Loss: 3.9272, Scaled Loss: 0.4909, LR: 6.16e-05, Accumulation Step: 7/8, Current Time: 2025-02-22 20:05:23 
/opt/anaconda3/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '

```