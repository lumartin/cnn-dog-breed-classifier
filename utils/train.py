import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import datasets

def build_datasets(train_path, validation_path, test_path, 
                    default_transformations, mixed_transformations) :

  train_data = datasets.ImageFolder(train_path, transform=mixed_transformations)
  valid_data = datasets.ImageFolder(validation_path, transform=default_transformations)
  test_data = datasets.ImageFolder(test_path, transform=default_transformations)

  trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
  validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True, num_workers=0)
  testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)


  return {'train': train_data, 'validation': valid_data, 'test': test_data},\
         {'train' : trainloader, 'valid' : validloader, 'test' : testloader}



def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path, verbose=False):
    """returns trained model"""
    
    valid_loss_min = np.Inf 
    initial_time = time.time()
    train_losses, valid_losses = [], []
    
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0
        initial_epoch_time = time.time()

        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            log_ps = model(data)
            loss = criterion(log_ps, target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                if use_cuda:
                    data, target = data.cuda(), target.cuda()

                ps_log = model(data)
                loss = criterion(ps_log, target)
                valid_loss = valid_loss + (1 / (batch_idx + 1)) * (loss.data - valid_loss)

            if verbose:
                final_epoch_time = time.time()
                print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTime spent: {:.6f}'.format(
                   epoch, 
                   train_loss,
                   valid_loss,
                   final_epoch_time - initial_epoch_time
                   ))            
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        if valid_loss <= valid_loss_min:
           torch.save(model.state_dict(), save_path)
           valid_loss_min = valid_loss    
    
    final_time = time.time()
    return model, train_losses, valid_losses, final_time - initial_time


def test(loaders, model, criterion, use_cuda):
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        pred = output.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
    print('Test Loss: {:.6f}'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)\n\n' % (
        100. * correct / total, correct, total))

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def reboot(datasets, loaders, model, criterion, default_model):
    del loaders
    del model
    del criterion
    torch.cuda.empty_cache()
    
    trainloader = torch.utils.data.DataLoader(datasets['train'], batch_size=64, shuffle=True, num_workers=4)
    validloader = torch.utils.data.DataLoader(datasets['validation'], batch_size=32, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(datasets['test'], batch_size=32, shuffle=False, num_workers=4)
    
    new_model = default_model
    new_model.apply(weights_init)
    new_model.cuda()
    
    new_criterion = criterion_scratch = nn.CrossEntropyLoss()
    
    return {'train' : trainloader, 'valid' : validloader, 'test' : testloader}, new_model, new_criterion

def run_experiments(datasets, loaders, hyperparameters, use_cuda=True, model_file='model'):
  current_optimizer = ''
  model_scratch = hyperparameters['model']
  criterion_scratch = ''
  for optimizer in hyperparameters['optimizers']:
      for lr in hyperparameters['learning_rates']:
          loaders, model_scratch, criterion_scratch = reboot(datasets, 
                                                             loaders, 
                                                             model_scratch, 
                                                             criterion_scratch, 
                                                             hyperparameters['model'])
          del current_optimizer
          current_optimizer = optimizer(model_scratch.parameters(), lr)
          
          print(current_optimizer)
          model, train_losses, valid_losses, time_spent = train(hyperparameters['epochs'], 
                                                                loaders, 
                                                                model_scratch, 
                                                                current_optimizer, 
                                                                criterion_scratch, 
                                                                use_cuda, 
                                                                model_file + '_lr_' + str(lr) + '.pt',
                                                                verbose=True)
          
          print("time spent: ", str(datetime.timedelta(seconds=time_spent)))
          
          min_train_loss = float(min(train_losses))
          min_valid_loss = float(min(valid_losses))
          
          print("Minimum trainig loss: ", min_train_loss)
          print("Minimum validation loss: ", min_valid_loss)

          plt.plot(train_losses, label='Training loss')
          plt.plot(valid_losses, label='Validation loss')
          
          plt.legend(frameon=False)
          plt.show()
        
          model_scratch.load_state_dict(torch.load(model_file + '_lr_' + str(lr) + '.pt'))
          test(loaders, model_scratch, criterion_scratch, use_cuda)
     

