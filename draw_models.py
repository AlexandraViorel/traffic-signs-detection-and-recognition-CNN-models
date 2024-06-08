from keras.utils import plot_model
from keras.models import load_model

modelTSRNet1 = load_model(r'D:\Faculty materials\BACHELORS-THESIS\TSR-CNN1\TSRNet1\models\mild-sweep-11.h5')
modelTSRNet2 = load_model(r'D:\Faculty materials\BACHELORS-THESIS\TSR-CNN2\TSRNet2\models\electric-sweep-10.h5')
modelTSRNet3Original = load_model(r'D:\Faculty materials\BACHELORS-THESIS\TSR-CNN3\TSRNet3Original\models\peachy-sweep-9.h5')
modelTSRNet3New = load_model(r'D:\Faculty materials\BACHELORS-THESIS\TSR-CNN3\TSRNet3New\models\skilled-sweep-4.h5')

plot_model(model=modelTSRNet2, to_file='TSRNet2Architecture.png', show_shapes=False, show_layer_names=True, show_layer_activations=True, rankdir='LR')
plot_model(model=modelTSRNet3Original, to_file='TSRNet3OrigArchitecture.png', show_shapes=False, show_layer_names=True, show_layer_activations=True, rankdir='LR')
plot_model(model=modelTSRNet3New, to_file='TSRNet3NewArchitecture.png', show_shapes=False, show_layer_names=True, show_layer_activations=True, rankdir='LR')

