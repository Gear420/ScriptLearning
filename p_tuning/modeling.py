import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join

import re


import torch.nn as nn

from transformers import AutoTokenizer
#
#from models import get_embedding_layer, create_model
from data_utils.vocab import get_vocab_by_strategy, token_wrapper
from data_utils.dataset import load_file
#from prompt_encoder import PromptEncoder


from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM


def create_model(args):
    if '11b' in args.model_name:
        from ..megatron_11b.megatron_wrapper import load_megatron_lm
        print("Warning: loading MegatronLM (11B) in fp16 requires about 28G GPU memory, and may need 3-5 minutes to load.")
        return load_megatron_lm(args)
    MODEL_CLASS, _ = get_model_and_tokenizer_class(args)
    model = MODEL_CLASS.from_pretrained(args.model_name)
    if not args.use_lm_finetune:
        if 'megatron' in args.model_name:
            raise NotImplementedError("MegatronLM 11B is not for fine-tuning.")
        model = model.half()
    return model


def get_model_and_tokenizer_class(args):
    if 'gpt' in args.model_name:
        return GPT2LMHeadModel, AutoTokenizer
    elif 'bert' in args.model_name:
        return AutoModelForMaskedLM, AutoTokenizer
    elif 'megatron' in args.model_name:
        return None, AutoTokenizer
    else:
        raise NotImplementedError("This model type ``{}'' is not implemented.".format(args.model_name))


def get_embedding_layer(args, model):
    if 'roberta' in args.model_name:
        embeddings = model.roberta.get_input_embeddings()
    elif 'bert' in args.model_name:
        embeddings = model.bert.get_input_embeddings()
    elif 'gpt' in args.model_name:
        embeddings = model.base_model.get_input_embeddings()
    elif 'megatron' in args.model_name:
        embeddings = model.decoder.embed_tokens
    else:
        raise NotImplementedError()
    return embeddings



class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device, args):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        
        print(input_embeds.shape)
        print(input_embeds.shape)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        

        return output_embeds
#prompt encoder的作用是什么？:用于生成prompt
#

class PTuneForLAMA(torch.nn.Module):

    def __init__(self, args, device, template):
        super().__init__()
        self.args = args
        self.device = device

        # load relation templates
        self.relation_templates = dict(
            (d['relation'], d['template']) for d in load_file(join(self.args.data_dir, 'relations.jsonl')))
        print(self.relation_templates)

        # load tokenizer
        tokenizer_src = 'roberta-large' if 'megatron' in self.args.model_name else self.args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)
        print(self.tokenizer)

        # load pre-trained model
        if 'megatron' in self.args.model_name and self.args.use_lm_finetune:
            raise RuntimeError("Can not apply args.use_lm_finetune=True on MegatronLM 11B.")
        self.model = create_model(self.args)
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = self.args.use_lm_finetune
        self.embeddings = get_embedding_layer(self.args, self.model)

        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))

        if 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
            template = (template[0], template[1], 0)
        self.template = template

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, self.device, args)
        self.prompt_encoder = self.prompt_encoder.to(self.device)

    def embed_input(self, queries):
        bz = 1
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        # For using handcraft prompts
        if self.args.use_original_template:
            return raw_embeds

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        replace_embeds = self.prompt_encoder()
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def get_query(self, x_h, prompt_tokens, x_t=None):
        # For using handcraft prompts
        if self.args.use_original_template:
            if 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
                query = re.sub(r'\[Y\].*', '', self.relation_templates[self.args.relation_id].replace('[X]', x_h))
                return self.tokenizer(' ' + query)['input_ids']
            else:
                query = self.relation_templates[self.args.relation_id].replace('[X]', x_h).replace('[Y]',
                                                                                                   self.tokenizer.mask_token)
                return self.tokenizer(' ' + query)['input_ids']
        # For P-tuning
        if 'gpt' not in self.args.model_name and 'megatron' not in self.args.model_name:
            # BERT-style model
            return [[self.tokenizer.cls_token_id]  # [CLS]
                    + prompt_tokens * self.template[0]
                    + [self.tokenizer.mask_token_id]  # head entity
                    + prompt_tokens * self.template[1]
                    + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # [MASK] (tail entity)
                    + (prompt_tokens * self.template[2] if self.template[
                                                               2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                    + [self.tokenizer.sep_token_id]
                    ]
        elif 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
            print('in gpt:')
            print(x_h)
            # GPT-style models
            return [prompt_tokens * self.template[0]
                    + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # head entity
                    + prompt_tokens * self.template[1]
                    + (self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(' ' + x_t)) if x_t is not None else [])
                    ]
        else:
            raise NotImplementedError("The query template for {} has not been defined.".format(self.args.model_name))

    def forward(self, x_hs, x_ts, return_candidates=False):
        bz = 1
        print('x_hs:')
        print(x_hs)

        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        print('prompt:')
        print(prompt_tokens)
        x_ts = [token_wrapper(self.args, x_t) for x_t in x_ts]
        #query = self.get_query(x_hs[0],prompt_tokens)
        #PRINT("QUERY:")
        #print(query)
        queries = [torch.LongTensor(self.get_query(x_hs[i], prompt_tokens)).squeeze(0) for i in range(bz)]
        print(queries)
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
        print("queries:")
        print(queries)
        print("x_t:")
        print(x_ts)
        # construct label ids
        label_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(x_ts)).reshape(
            (bz, -1)).to(self.device)
        #label_ids = torch.Tensor(self.tokenizer.convert_tokens_to_ids(x_ts)).to(self.device)
        #tokenized_text2 = self.tokenizer.tokenize(x_ts)
        #label_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_text2)).to(self.device)
        print('label_id:')
        print(label_ids)
        attention_mask = queries != self.pad_token_id

        # get embedded input
        inputs_embeds = self.embed_input(queries)
        print(inputs_embeds.shape)

        def bert_out():
            label_mask = (queries == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(
                1).to(self.device)  # bz * 1
            
            labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
            print(labels)
            labels = labels.scatter_(1, label_mask, label_ids)
            output = self.model(inputs_embeds=inputs_embeds.to(self.device),
                                attention_mask=attention_mask.to(self.device).bool(),
                                labels=labels.to(self.device))
            loss, logits = output.loss, output.logits

            pred_ids = torch.argsort(logits, dim=2, descending=True)
            hit1 = 0
            top10 = []
            for i in range(bz):
                pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
                for pred in pred_seq:
                    if pred in self.allowed_vocab_ids:
                        break
                if pred == label_ids[i, 0]:
                    hit1 += 1

            if return_candidates:
                return loss, hit1, top10
            return loss, hit1

        def gpt_out():
            bz = 1
            labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
            label_mask = (attention_mask.long().sum(dim=1) - 1).unsqueeze(1).to(self.device)
            print("label_mask:")
            print(label_mask)
            print(labels)
            labels = labels.scatter_(1, label_mask, label_ids)
            #labels[0][2] = torch.tensor(20)
            #print(inputs_embeds.shape)
            #inputs_embeds_temp = inputs_embeds[0]
            #print(inputs_embeds_temp.shape)
            output = self.model(inputs_embeds=inputs_embeds.to(self.device).half(),
                                attention_mask=attention_mask.to(self.device).half(),
                                labels=labels.to(self.device))
            loss, logits = output.loss, output.logits
            
            #print(loss)
            #loss.requires_grad = True
            #print(loss)
            pred_ids = torch.argsort(logits, dim=2, descending=True)
            hit1 = 0
            top10 = []
            for i in range(bz):
                top10.append([])
                pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
                for pred in pred_seq:
                    if pred in self.allowed_vocab_ids:
                        top10[-1].append(pred)
                        if len(top10[-1]) >= 10:
                            break
                pred = top10[-1][0]
                if pred == label_ids[i,0]:
                    hit1 += 1

            if return_candidates:
                return loss, hit1, top10
            return loss, hit1

        def megatron_out():
            labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
            label_mask = (attention_mask.long().sum(dim=1) - 1).unsqueeze(1).to(self.device)
            labels = labels.scatter_(1, label_mask, label_ids)
            if not self.args.use_lm_finetune:
                _attention_mask = attention_mask.float().half()
                _input_embeds = inputs_embeds.float().half()
            else:
                _attention_mask = attention_mask.float()
                _input_embeds = inputs_embeds.float()
            output = self.model.decoder.predict(prev_output_tokens=queries,
                                                inputs_embeds=_input_embeds.to(self.device),
                                                attention_mask=_attention_mask.to(self.device).bool(),
                                                labels=labels.to(self.device))
            logits, loss = output

            pred_ids = torch.argsort(logits, dim=2, descending=True)
            hit1 = 0
            top10 = []
            for i in range(bz):
                top10.append([])
                pred_seq = pred_ids[i, label_mask[i, 0]].to(self.device).tolist()
                for pred in pred_seq:
                    if pred in self.allowed_vocab_ids:
                        top10[-1].append(pred)
                        if len(top10[-1]) >= 10:
                            break
                pred = top10[-1][0]
                if pred == label_ids[i, 0]:
                    hit1 += 1
            if return_candidates:
                return loss, hit1, top10
            return loss, hit1

        if 'bert' in self.args.model_name:
            return bert_out()
        elif 'gpt' in self.args.model_name:
            return gpt_out()
        elif 'megatron' in self.args.model_name:
            return megatron_out()
        else:
            raise NotImplementedError()
