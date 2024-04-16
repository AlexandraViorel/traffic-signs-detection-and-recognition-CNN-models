sweep_configuration = {
    "method": "random",
    "name": "TSR-CNN2",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [32, 64]},
        "epochs": {"values": [50]}, 
        "learning_rate": {"values": [0.01, 0.001, 0.0001]},
        "drop": {"values": [0.3, 0.4, 0.5]},
        "f1": {"values": [32, 48, 64]},
        "f2": {"values": [32, 48, 64]},
        "f3": {"values": [32, 48, 64]},
        "fc_layer_size": {"values": [256, 512, 1024]}
    },
}