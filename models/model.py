import torch
import  torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AutoConfig, AutoModel
from utils.config import  *

import numpy as np

class transformer_model(nn.Module):
    def __init__(self, model_name, device, drop_prob=0.3):
        super(transformer_model, self).__init__()
        self.device=device
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.output_dim=(self.config.num_hidden_layers+1)*self.config.hidden_size
        self.freeze_layers = self.config.num_hidden_layers - 2
        self.hidden_dim1 = 1000
        self.hidden_dim2 = 40
        self.final_size=1
        if args.to_freeze:
            cnt = 0
            for child in self.model.children():
                cnt = cnt + 1
                if cnt <= self.freeze_layers:
                    for param in child.parameters():
                        param.requires_grad = False
        if args.add_features ==1 :
            self.output_dim += 2
        elif args.add_features == 2:
            self.output_dim += 3
        self.fc1 = nn.Linear(self.output_dim, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, self.final_size)
        self.dropout = nn.Dropout(p=drop_prob)

    def avg(self, a, st, end):
        k = a
        lis = []
        for i in range(st, end):
            lis.append(a[i])
        x = torch.mean(torch.stack(lis), dim=0)
        return x

    def forward(self, token_ids, token_starts, sents_other_feats, lm_lengths=None, labels=None):

        batch_size = token_ids.size()[0]
        pad_size = token_ids.size()[1]
        # print("batch size",batch_size,"\t\tpad_size",pad_size)
        model_mask=(token_ids!=0).int().to(self.device)
        output = self.model(token_ids.long(), attention_mask=model_mask)

        # Concatenating hidden dimensions of all encoder layers
        model_out = output[-1][0]
        for layers in range(1, self.config.num_hidden_layers+1, 1):
            model_out = torch.cat((model_out, output[-1][layers]), dim=2)

        # model_out:[batch_size, seq_len, ]
        # Concatenating other features
        if args.add_features == 1:
            model_out=torch.dstack((model_out,sents_other_feats[:,:,0:2]))
        elif args.add_features == 2:
            model_out=torch.dstack((model_out,sents_other_feats))
        # Fully connected layers with relu and dropouts in between
        pred_logits = torch.relu(self.fc1(self.dropout(model_out).float()))
        pred_logits = torch.relu(self.fc2(self.dropout(pred_logits)))
        pred_logits = torch.sigmoid(self.fc3(self.dropout(pred_logits)))
        pred_logits = torch.squeeze(pred_logits, 2)

        pred_labels = torch.tensor(np.zeros(token_starts.size()), dtype=torch.float64).to(self.device)
        for b in range(batch_size):
            for w in range(pad_size):
                if (token_starts[b][w] != 0):
                    if (token_starts[b][w] >= pad_size):
                        print(token_starts[b])
                    else:
                        st = token_starts[b][w]
                        end = token_starts[b][w + 1]
                        if (end == 0):
                            end = st + 1
                            while (model_mask[b][end] != 0):
                                end = end + 1
                        # For using average or just the first token of a word (in case of word splitting by tokenizer)
                        # pred_labels[b][w] = self.avg(pred_logits[b],st,end)
                        pred_labels[b][w] = pred_logits[b][token_starts[b][w]]

        if (labels != None):
            lm_lengths, lm_sort_ind = lm_lengths.sort(dim=0, descending=True)
            scores = labels[lm_sort_ind]
            targets = pred_labels[lm_sort_ind]
            scores = pack_padded_sequence(scores, lm_lengths.cpu().int(), batch_first=True).data
            targets = pack_padded_sequence(targets, lm_lengths.cpu().int(), batch_first=True).data
            # print(targets,scores)
            loss_fn = nn.BCELoss()
            loss = loss_fn(targets.float(), scores.float())

            return loss, pred_labels
        else:
            return 0.0, pred_labels
