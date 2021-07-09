from torch import optim
from torch.optim import lr_scheduler
from main import *


def learner(model, train_loader, test_loader, epochs, optimiser, scheduler, device):
  '''
  learner => Complete testing and training of the model

  Keyword Arguments:
  model       -> The actual model designed
  train_loader-> The shuffled training dataset
  test_loader -> The shuffled testing dataset
  epochs      -> Number of epochs for training and testing
  optimiser   -> Type optimiser used (ex: Adam/SGD)
  Scheduler   -> LR scheduler
  device      -> Device configuration to which inputs/targets are being sent

  return:
  train_acc     -> training accuracy rate
  train_losses  -> training losses
  test_acc      -> testing accuracy rate
  test_losses   -> testing losses
  model         -> The actual model designed
  '''
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    optimdict = {'Adam': optim.Adam(model.parameters(), lr=0.03), 'SGD': optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9)}
    sched = optim.lr_scheduler.MultiStepLR(
        optimdict[optimiser], milestones=[10, 15], gamma=0.1)
    epoch = epochs

    for epoch in range(1, epoch + 1):
        print(f'Epoch {epoch}:')

        train(model, device, train_loader, optimdict[optimiser],
              train_acc, train_losses)

        test(model, device, test_loader, test_acc, test_losses)
        if scheduler == True:
            sched.step()

    return (train_acc, train_losses, test_acc, test_losses), model
