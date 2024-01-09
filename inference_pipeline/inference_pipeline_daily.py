import modal



LOCAL=False
if LOCAL == False:
    stub = modal.Stub('hackernews-score-pred-inference')
    image = modal.Image.debian_slim().pip_install(["hopsworks", "torch", "transformers"])

    @stub.function(image=image,
                schedule=modal.Period(days=1), 
                secret=modal.Secret.from_name("hopsworks.ai"),
                )
    def f():
        g()

def g():
    import torch
    import torch.nn as nn
    import hopsworks
    from transformers import BertTokenizer, BertModel
    from datetime import datetime, timedelta

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
            val_loss, val_acc = np.Inf, 0.
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
            train_loss, correct = np.Inf, 0

            self.model.train()
            for data, label in tqdm(train_loader):
                device_data = {}
                for k, v in data.items():
                    device_data[k] = v.to(self.device)
                device_label = label.to(self.device, dtype=torch.float32)
                
                optimizer.zero_grad()
                output = self.model(device_data)

                criterion = nn.MSELoss()
                loss = criterion(output, device_label)

                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                pred = output.argmax(dim=0)
                correct += pred.eq(device_label).sum().item()

            train_loss /= len(train_loader.dataset)
            acc = correct/len(train_loader.dataset)

            print('training set: average loss: %.4f, acc: %d/%d(%.3f%%)' %(train_loss,
                correct, len(train_loader.dataset), 100 * acc))

            return train_loss, acc


        def val(self, val_loader):
            print('validation on %d data......'%len(val_loader.dataset))
            self.model.eval()
            val_loss, correct = np.Inf, 0.
            with torch.no_grad():
                for data, label in val_loader:
                    device_data = {}
                    for k, v in data.items():
                        device_data[k] = v.to(self.device)
                    device_label = label.to(self.device, dtype=torch.float32)
                    
                    output = self.model(device_data)

                    criterion = nn.MSELoss()
                    val_loss += criterion(output, device_label).item() #sum up batch loss

                    pred = output.argmax(dim=0)
                    correct += pred.eq(device_label).sum().item()
                val_loss /= len(val_loader.dataset)  # avg of sum of batch loss
                acc = correct/len(val_loader.dataset)

            print('Val set:Average loss:%.4f, acc:%d/%d(%.3f%%)' %(val_loss,
                correct, len(val_loader.dataset), 100. * acc))

            return val_loss, acc
        
        
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

    class BERT_classifier(nn.Module):
        def __init__(self, bertmodel, num_score):
            super(BERT_classifier, self).__init__()
            self.bertmodel = bertmodel
            self.dropout = nn.Dropout(p=bertmodel.config.hidden_dropout_prob)
            self.linear = nn.Linear(bertmodel.config.hidden_size, num_score)

        def forward(self, wrapped_input):
            hidden = self.bertmodel(**wrapped_input)
            last_hidden_state, pooler_output = hidden[0], hidden[1]
            output_value = self.linear(pooler_output).squeeze()
            score = torch.sigmoid(output_value) * 1000
            return score

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained("bert-base-uncased")

    model = BERT_classifier(bert, 1)
    model.load_state_dict(torch.load('inference_pipeline/model_1.pt', map_location=torch.device('cpu')))
    model.eval()

    project = hopsworks.login(project='id2223_enric')
    fs = project.get_feature_store()
    
    hackernews_fg = fs.get_feature_group("hackernews_fg", 2)
    query = hackernews_fg.select_all()
    feature_view = fs.get_or_create_feature_view(name="hackernews_fv",
                                    version=2,
                                    description="Hackernews feature view",
                                    labels=["score"],
                                    query=query)

    batch_data = feature_view.get_batch_data(start_time=datetime.now() - timedelta(days=1))
    
    titles = batch_data['title'].tolist()
    
    nn_obj = nn_factory(model, 'cpu', tokenizer)

    for title in titles:
        print(nn_obj.predict(title))
    

if __name__ == "__main__":
    if LOCAL == True:
        # NOTE: Create an .env file in the root directory if you want to run this locally
        g()
    else:
        modal.runner.deploy_stub(stub)
        
        
        