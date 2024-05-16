import wandb
from model_old import TSRNet3Old
from model_new import TSRNet3New
from keras.optimizers import Adam, SGD
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.losses import CategoricalCrossentropy
from data_processing import split_train_val
from sweep_configuration import sweep_configuration
from fsspec import Callback
import os

class CustomWandbCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            wandb.log({'epoch': epoch, 'loss': logs['loss'], 'accuracy': logs['accuracy'], 'val_loss': logs.get('val_loss'), 'val_accuracy': logs.get('val_accuracy')})

def train():

    with wandb.init() as run:
        x_train, x_val, y_train, y_val = split_train_val()

        config = run.config
        wandb.config.architecture_name = "TSR-CNN3New"
        wandb.config.dataset_name = "GermanAndBelgianTS"
        wandb.config.optimizer = "SGD"

        # model = TSRNet3Old.build_model(config.f1, config.f2, config.f3, config.fc_layer_size, config.drop)
        model = TSRNet3New.build_model(config.k, config.f1, config.f2, config.f3, config.fc_layer_size, config.drop)

        optim = SGD(learning_rate=config.learning_rate)
        loss = CategoricalCrossentropy(from_logits=False)
        model.compile(optimizer=optim, loss=loss, metrics=["accuracy"])
        
        model.fit(aug.flow(x_train, y_train, batch_size=config.batch_size), validation_data=(x_val, y_val), epochs=config.epochs, callbacks=[CustomWandbCallback()])

        model_path = os.path.join('TSRNet3New', 'models', f'{run.name}.h5')
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        model.save(model_path)

aug = ImageDataGenerator(rotation_range=10,
                         zoom_range=0.15,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.15,
                         horizontal_flip=False,
                         vertical_flip=False,
                         fill_mode='nearest')

sweep_id = wandb.sweep(sweep=sweep_configuration, project="TSRNet3New")
wandb.agent(sweep_id, function=train, count=10)