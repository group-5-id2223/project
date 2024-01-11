import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')

class nn_factory():
    def __init__(self, model, device, tokenizer):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer
    
    def fit(self, epoch, optimizer, train_loader, val_loader, model_save_path):
        val_loss, val_acc = 0, 0.
        train_loss_hist, train_acc_hist = [],[]
        val_loss_hist, val_acc_hist = [],[]

        for ep in range(1, epoch + 1):
            epoch_begin = time.time()
            cur_train_loss, cur_train_acc = self.train(train_loader, optimizer, ep)
            cur_val_loss, cur_val_acc = self.val(val_loader)

            print('elapse: %.2fs \n' % (time.time() - epoch_begin))

            if cur_val_loss <= val_loss:
                print('improve validataion loss, saving model...\n')
                torch.save(self.model.state_dict(),
                           os.path.join(model_save_path, 
                                        f'best_model_ep_{ep}_loss_{cur_val_loss}_acc_{cur_val_acc}.pt'))

                val_loss = cur_val_loss
                val_acc = cur_val_acc

            train_loss_hist.append(cur_train_loss)
            train_acc_hist.append(cur_train_acc)
            val_loss_hist.append(cur_val_loss)
            val_acc_hist.append(cur_val_acc)

        # save final model
        state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
        torch.save(state, os.path.join(model_save_path, 'last_model.pt'))

        ### graph train hist ###
        fig = plt.figure()
        plt.plot(train_loss_hist)
        plt.plot(val_loss_hist)
        plt.legend(['train loss','val loss'], loc='best')
        plt.savefig(os.path.join(model_save_path, 'loss.jpg'))
        plt.close(fig)
        fig = plt.figure()
        plt.plot(train_acc_hist)
        plt.plot(val_acc_hist)
        plt.legend(['train acc', 'val acc'], loc='best')
        plt.savefig(os.path.join(model_save_path, 'acc.jpg'))
        plt.close(fig)
    
    def train(self, train_loader, optimizer, epoch):
        print('[epoch %d]train on %d data......'%(epoch, len(train_loader.dataset)))
        train_loss, mse_loss = 0, 0

        self.model.train()
        for data, label in tqdm(train_loader):
            device_data = {}
            for k, v in data.items():
                device_data[k] = v.to(self.device)
            device_label = label.to(self.device, dtype=torch.float32)
            
            optimizer.zero_grad()
            output = self.model(device_data)
            criterion = nn.MSELoss()
            mse_loss = criterion(device_label, output)
            loss = self.hybrid_loss(device_label, output)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            mse_loss += mse_loss.detach().cpu().numpy()

        train_loss /= len(train_loader.dataset)
        mse_loss = mse_loss/len(train_loader.dataset)

        print('training set: average loss: %.4f, mse_loss: %d/%d(%.3f%%)' %(train_loss,
              mse_loss, len(train_loader.dataset), 100 * mse_loss))

        return train_loss, mse_loss
    
    def hybrid_loss(self, y_true, y_pred):
        epsilon = 1e-6
        weight_mae = 0.1
        weight_msle = 1.
        weight_poisson = 0.1
        
        y_pred = y_pred.squeeze()
        mae_loss = weight_mae * torch.mean(torch.abs(y_pred - y_true), axis=-1)
        
        first_log = torch.log(torch.clip(y_pred, 1, None) + 1.)
        second_log = torch.log(torch.clip(y_true, epsilon, None) + 1.)
        msle_loss = weight_msle * torch.mean(torch.square(first_log - second_log), axis=-1)
        
        poisson_loss = weight_poisson * torch.mean(y_pred - y_true * torch.log(torch.clip(y_pred, epsilon, None)), axis=-1)
        # print(f"mae_loss: {mae_loss} ; msle_loss: {msle_loss} ; poisson_loss: {poisson_loss}")
        return torch.mean(mae_loss + msle_loss + torch.abs(poisson_loss))


    def val(self, val_loader):
        print('validation on %d data......'%len(val_loader.dataset))
        self.model.eval()
        val_loss, mse_loss = 0, 0.
        with torch.no_grad():
            for data, label in val_loader:
                device_data = {}
                for k, v in data.items():
                    device_data[k] = v.to(self.device)
                device_label = label.to(self.device, dtype=torch.float32)
                
                output = self.model(device_data)
                criterion = nn.MSELoss()
                val_loss += self.hybrid_loss(device_label, output).item() #sum up batch loss

                mse_loss += criterion(device_label, output).detach().cpu().numpy()


            val_loss /= len(val_loader.dataset)  # avg of sum of batch loss
            mse_loss = mse_loss/len(val_loader.dataset)

        print('Val set:Average loss:%.4f, mse_loss:%d/%d(%.3f%%)' %(val_loss,
              mse_loss, len(val_loader.dataset), 100. * mse_loss))

        return val_loss, mse_loss
    
    
    def predict_proba(self, sentence):
        wrapped_input = self.tokenizer(sentence, max_length=30, add_special_tokens=True, 
                                       truncation=True, padding='max_length', return_tensors="pt")

        with torch.no_grad():
            log_prob = self.model(wrapped_input)
            pred_prob = torch.exp(log_prob).data.cpu().numpy()

        return pred_prob


    def predict(self, sentence):
        pred_prob = self.predict_proba(sentence)
        score = np.argmax(pred_prob, axis=0)

        return score