import os
import torch
import matplotlib.pyplot as plt

def init_session_history(args):
    """
    Initializes a section in the history file for current training session
    Creates file if it does not exist
    :param base_model_name: the model base name
    :return: None
    """

    with open(args.history_path, 'a+') as hist_fp:
        hist_fp.write(
            '\n============================== Base_model: {} ==============================\n'.format(args.base_model_name)

            + 'arguments: {}\n'.format(args)
        )

def save_weights(model, args, epoch, fold, optimizer):
    """
    Saves a state dictionary given a model, epoch, the epoch its training in, and the optimizer
    :param base_model_name: name of the base model in training session
    :param model: model to save
    :param epoch: epoch model has trained to
    :param optimizer: optimizer used during training
    :param model_path: path of where model checkpoint is saved to
    :return:
    """

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    model_name = '{}_fold{}_{}_{}'.format(args.base_model_name, fold, epoch, args.lr)
    torch.save(state, '{}/{}.pt'.format(args.model_path, model_name))
    
    file_size = os.path.getsize('{}/{}.pt'.format(args.model_path, model_name))
    print('file_size:', file_size)
    print(f"Model file size: {file_size / (1024 ** 2)} MB")

    return model_name

def load_weights(model, args, target_fold):
    """
    Loads previously trained weights into a model given an epoch and the model itself
    :param base_model_name: name of the base model in training session
    :param model: model to load weights into
    :param epoch: what epoch of training to load
    :param model_path: path of where model is loaded from
    :return: the model with weights loaded in
    """

    pretrained_dict = torch.load('{}/{}_{}_{}_{}.pt'.format(args.model_path, args.base_model_name, target_fold, args.start_epoch, args.lr))['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def plot_curves(base_model_name, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, epochs, fold):
    """
    Given progression of train/val loss/acc, plots curves
    :param base_model_name: name of base model in training session
    :param train_loss: the progression of training loss
    :param val_loss: the progression of validation loss
    :param train_acc: the progression of training accuracy
    :param val_acc: the progression of validation accuracy
    :param train_f1: the progression of training f1 score
    :param val_f1: the progression of validation f1 score
    :param epochs: epochs that model ran through
    :return: None
    """

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(epochs, train_loss, label='train loss')
    plt.plot(epochs, val_loss, label='val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Loss curves')
    plt.legend()

    plt.subplot(132)
    plt.plot(epochs, train_acc, label='train accuracy')
    plt.plot(epochs, val_acc, label='val accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Accuracy curves')
    plt.legend()

    plt.subplot(133)
    plt.plot(epochs, train_f1, label='train f1 score')
    plt.plot(epochs, val_f1, label='val f1 score')
    plt.xlabel('epochs')
    plt.ylabel('f1 score')
    plt.title('f1 curves')
    plt.legend()

    plt.suptitle(f'Session: {base_model_name}_{fold}')

    plt.savefig(f'{base_model_name}_{fold}.png')
    plt.show()

def write_history(
        history_path,
        model_name,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
        train_f1,
        val_f1,
        train_precision,
        val_precision,
        train_recall,
        val_recall,
        train_confusion_matrix,
        val_confusion_matrix
):
    """
    Write a history.txt file for each model checkpoint
    :param history_path: path to history file
    :param model_name: name of the current model checkpoint
    :param train_loss: the training loss for current checkpoint
    :param val_loss: the validation loss for current checkpoint
    :param train_acc: the training accuracy for current checkpoint
    :param val_acc: the validation accuracy for current checkpoint
    :param train_f1: the training f1 score for current checkpoint
    :param val_f1: the validation f1 score for current checkpoint
    :param train_precision: the training precision score for current checkpoint
    :param val_precision: the validation precision score for current checkpoint
    :param train_recall: the training recall score for current checkpoint
    :param val_recall: the validation recall score for current checkpoint
    :param train_confusion_matrix: the training conf matrix for current checkpoint
    :param val_confusion_matrix: the validation conf matrix for current checkpoint
    :return: None
    """

    with open(history_path, 'a') as hist_fp:
        hist_fp.write(
            '\ncheckpoint name: {} \n'.format(model_name)

            + 'train loss: {} || train accuracy: {} || train f1: {} || train precision: {} || train recall: {}\n'.format(
                round(train_loss, 5),
                round(train_acc, 5),
                round(train_f1, 5),
                round(train_precision, 5),
                round(train_recall, 5)
            )

            + train_confusion_matrix + '\n'

            + 'val loss: {} || val accuracy: {} || val f1: {} || val precision: {} || val recall: {}\n'.format(
                round(val_loss, 5),
                round(val_acc, 5),
                round(val_f1, 5),
                round(val_precision, 5),
                round(val_recall, 5)
            )

            + val_confusion_matrix + '\n'
        )


def read_history(history_path):
    """
    Reads history file and prints out plots for each training session
    :param history_path: path to history file
    :return: None
    """

    with open(history_path, 'r') as hist:

        # get all lines
        #print('get all lines')
        all_lines = hist.readlines()

        # remove newlines for easier processing
        #print('remove newlines for easier processing')
        rem_newline = []
        for line in all_lines:
            if len(line) == 1 and line == '\n':
                continue
            rem_newline.append(line)

        # get individual training sessions
        #print('get individual training sessions')
        base_names = []
        base_indices = []
        for i in range(len(rem_newline)):
            if rem_newline[i][0] == '=':
                base_names.append(rem_newline[i].replace('=', '').split(' ')[-2])
                base_indices.append(i)

        # create plots for each individual session
        #print('create plots for each individual session')
        fold = 0
        for i in range(len(base_names)):
            name = base_names[i]

            # get last session
            #print('get last session')
            if i == len(base_names) - 1:
                session_data = rem_newline[base_indices[i]:]

            # get session
            else:
                #print('get session')
                session_data = rem_newline[base_indices[i]: base_indices[i + 1]]

            # now generate the plots
            #print('now generate the plots')
            train_plot_loss = []
            val_plot_loss = []
            train_plot_acc = []
            val_plot_acc = []
            train_plot_f1 = []
            val_plot_f1 = []
            plot_epoch = []

            for line in session_data:

                if 'arguments' in line:
                    print("Hyperparameters:")
                    print(line)

                # case for getting checkpoint epoch
                if line.startswith("checkpoint"):
                    #print("case for getting checkpoint epoch")
                    print("fold =",fold," epoch-",line.split('_')[-2]," ",line)
                    plot_epoch.append(int(line.split('_')[-2]))
                    fold = int(line.split('_')[-3])

                # case for getting train data for epoch
                elif line.startswith("train") and 'arguments' not in line:
                    #print("case for getting train data for epoch")
                    print(line)
                    train_plot_loss.append(float(line.split(' ')[2].replace("+0j", "").replace("(", "").replace(")", "")))
                    train_plot_acc.append(float(line.split(' ')[6].replace("+0j", "").replace("(", "").replace(")", "")))
                    train_plot_f1.append(float(line.split(' ')[10].replace("+0j", "").replace("(", "").replace(")", "")))

                # case for getting val data for epoch
                elif 'val' in line and 'arguments' not in line:
                    #print("case for getting val data for epoch")  
                    print(line)
                    val_plot_loss.append(float(line.split(' ')[2].replace("+0j", "").replace("(", "").replace(")", "")))
                    val_plot_acc.append(float(line.split(' ')[6].replace("+0j", "").replace("(", "").replace(")", "")))
                    val_plot_f1.append(float(line.split(' ')[10].replace("+0j", "").replace("(", "").replace(")", "")))

            # plot
            print("plot")  
            plot_curves(
                name,
                train_plot_loss,
                val_plot_loss,
                train_plot_acc,
                val_plot_acc,
                train_plot_f1,
                val_plot_f1,
                plot_epoch,
                fold
            )

if __name__ == "__main__":
    read_history("/content/Basketball-Action-Recognition/histories/history_r2plus1d_augmented_evaluate.txt")
