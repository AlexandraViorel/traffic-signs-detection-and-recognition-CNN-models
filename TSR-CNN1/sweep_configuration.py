sweep_configuration = {
    "method": "random",
    "name": "TSR-CNN1",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [32, 64]},
        "epochs": {"values": [50, 100]}, 
        "learning_rate": {"values": [0.01, 0.001, 0.0001]},
        "f1": {"values": [24]},
        "f2": {"values": [32]},
        "f3": {"values": [48]},
        "fc_layer_size": {"values": [256, 512, 1024]},
        "drop": {"values": [0.2, 0.3, 0.4, 0.5]}
    },
}