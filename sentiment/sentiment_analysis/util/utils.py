import os
import torch
from torch.autograd import Variable as Var
from torchtext.legacy import data
import matplotlib.pyplot as plt



# returns all possible combinations of the params contained in each list
def combos(l1, l2, l3):
    return [(p1, p2, p3) for p1 in l1 for p2 in l2 for p3 in l3]




# alternative to the dataloader shown in class
def create_iterator(train_data, val_data, test_data, batch_size, device):
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train_data, val_data, test_data),
            batch_size=batch_size,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            device=device
    )

    return train_iter, val_iter, test_iter




# calculates accuracy for an epoch
def accuracy(probs, target):
    preds = probs.argmax(dim=1)
    # boolean list of correct classifications
    correct = (preds==target)
    # count number of True values and divide by num samples
    return correct.sum().float() / float(target.size(0))




def export_graph(filename, epochs, train_data, val_data, title, y_label):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(y_label)
    ax.xaxis.grid(True, ls='dotted')
    ax.yaxis.grid(True, ls='dotted')
    epochs = range(epochs)
    ax.plot(epochs, train_data, label='train', color='steelblue', marker='d', markersize=8, linestyle='dashdot', linewidth=2)
    ax.plot(epochs, val_data, label='val', color='coral', marker='d', markersize=8, linestyle='dashdot', linewidth=2)
    ax.legend()

    path = '/home/asimov/sentiment_analysis/graphs'
    plt.savefig(os.path.join(path, filename))





def train(model, iterator, optimizer, lfunc):
    epoch_loss = 0
    epoch_acc = 0

    model.train(True)
    with torch.set_grad_enabled(True):
        for batch in iterator:
            optimizer.zero_grad() # zero out accumulated gradients

            text, text_lengths = batch.text
            probs = model(text, text_lengths)

            # calc metrics
            loss = lfunc(probs, batch.labels.squeeze())
            acc = accuracy(probs, batch.labels)

            # backprop
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator) , epoch_acc / len(iterator)




def validate(model, iterator, lfunc):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:

            text, text_lengths = batch.text
            probs = model(text, text_lengths).squeeze(1)

            loss = lfunc(probs, batch.labels)
            acc = accuracy(probs, batch.labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator) , epoch_acc / len(iterator)




def train_val(epochs, model, train_iter, val_iter, optimizer, lfunc, model_type, arch_type, optim_type):
    best_validation_loss = float('inf')
    training_loss, training_acc = (list(), list())
    validation_loss, validation_acc = (list(), list())

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_iter, optimizer, lfunc)
        val_loss, val_acc = validate(model, val_iter, lfunc)

        # appending datapoints for the epoch
        training_loss.append(train_loss)
        training_acc.append(train_acc)
        validation_loss.append(val_loss)
        validation_acc.append(val_acc)


        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            torch.save(model.state_dict(), model_type+'_'+arch_type+'_'+optim_type+'_saved_state.pt')


        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')


    return training_loss, training_acc, validation_loss, validation_acc
