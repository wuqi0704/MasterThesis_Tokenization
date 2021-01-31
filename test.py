
#%%
from Users.wuqi.MasterThesis_Tokenization.functions import prepare_cse
from functions import *

#%%
use_CSE = True
embedding_dim = 4096 # because using CSE 
batch_size = 10

train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=shuffle)
dev_loader = DataLoader(dataset=data_dev, batch_size=batch_size, shuffle=shuffle)

# Initialize network
model = LSTMTagger(character_size,embedding_dim,hidden_dim, num_layers,tagset_size,batch_size,use_CSE=use_CSE)
if(torch.cuda.is_available()):
	print(torch.cuda.current_device())
model = model.to(device); model.train()
optimizer = optim.SGD(model.parameters(), learning_rate)
loss_function = nn.NLLLoss()

#%%
item = iter(train_loader)
data,tags = item.next()
#%%
embedprepare_cse(data,batch_size=batch_size)

#%%
sentence = prepare_batch(data,letter_to_ix)
character_embeddings = nn.Embedding(character_size, embedding_dim)
embed = character_embeddings(sentence)

#%%
embed
