All trained model will be saved here: `/output/model_name`  
And every trained model be saved with `events.out.tfevents.***` log file and `log.csv`  
You could observe loss and dice during training through Tensorboard by run:
`tensorboard --logdir ./output/model_name`
