import numpy as np

def label_OHE(train_labels,test_labels):
    class_names = ['boxing', 'running', 'Handclapping', 'jogging', 'Walking', 'handwaving']
    class_to_int = {class_name: i for i, class_name in enumerate(class_names)}
    train_labels_int = np.array([class_to_int[label] for label in train_labels])
    num_labels = train_labels_int.shape[0]
    num_classes = 6
    oh_train_labels = np.zeros((num_labels, num_classes))
    oh_train_labels[np.arange(num_labels), train_labels_int] = 1

    test_labels_int = np.array([class_to_int[label] for label in test_labels])
    num_labels = test_labels_int.shape[0]
    num_classes = 6
    oh_test_labels = np.zeros((num_labels, num_classes))
    oh_test_labels[np.arange(num_labels), test_labels_int] = 1
    return oh_train_labels, oh_test_labels