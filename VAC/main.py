from pathlib import Path
import VAC.src.__init__
from VAC.src.visualization.visualize import *
from VAC.src.data.make_dataset import *
from VAC.src.features.build_features import *
from VAC.src.models.train_model import *

from dotenv import find_dotenv, load_dotenv



def main():
    """ Runs data processing scripts to turn raw video data from (../raw) into
        10 frame images of each video of each class ready to be analyzed (saved in ../processed).
    """
    video_capturing_function("/data/raw/train/", "/data/processed/Train_frames")
    video_capturing_function("/data/raw/test/", "/data/processed/Test_frames")

    #Visualize the frames and analyze the object in each frame by plotting the frames of each class per row (6 rows)
    visualize_frames("/data/processed/Train_frames")

    #creating training dataset from frames saved in the directory
    train_dir_path="/data/processed/Train_frames"
    train_dataset_new, train_labels = data_load_function_10frames(train_dir_path)
    
    #creating testing dataset from frames saved in the directory
    test_dir_path="/data/processed/Test_frames"
    test_dataset_new, test_labels = data_load_function_10frames(test_dir_path)

    #One Hot encoding of all training and testing labels
    oh_train_labels, oh_test_labels= label_OHE(train_labels,test_labels)

    #create CNN LSTM model based on alredy done hyperparameter tuning
    #The best hyperparameters are: {'conv_units_1': 320, 'conv_units_2': 128, 'conv_units_3': 128, 'Dropout_rate1': 0.2, 'lstm_units': 96, 'Dropout_rate2': 0.0, 'lr': 0.00010077872683004541}
    model_cnlst=build_convnet(320,128,128,0.2,96,0)
    optimizer_new=optimizers.Nadam(learning_rate=0.00010077872683004541, name='Nadam')
    model_cnlst.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['AUC'])
    history_new_cnlst=model_cnlst.fit(train_dataset_new,oh_train_labels,batch_size=8,epochs=20,
                            validation_data=(test_dataset_new,oh_test_labels))
    AUC(history_new_cnlst)

    #create pre trained VGG16 model
    train_data_rgb = np.repeat(train_dataset_new[..., np.newaxis], 3, -1)
    test_data_rgb = np.repeat(test_dataset_new[..., np.newaxis], 3, -1)

    check_point = keras.callbacks.ModelCheckpoint("VGG_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    model_VGG = create_model_VGG()
    model_VGG.fit(train_data_rgb, oh_train_labels, batch_size=4, epochs=50, validation_data=(test_data_rgb, oh_test_labels), callbacks=[check_point])
    
    predictions_VGG = model_VGG.predict(test_data_rgb, batch_size = 1)
    
    VGG_predicted_classes = np.argmax(predictions_VGG, axis=1)
    true_classes = np.argmax(oh_test_labels, axis=1)
    
    VGG_report = classification_report(true_classes, VGG_predicted_classes, target_names=['boxing', 'running', 'Handclapping', 'jogging', 'Walking', 'handwaving'])
    CF_matrix(true_classes, VGG_predicted_classes)


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()