# -*- coding: utf-8 -*-
# Proyecto con sentiment analysis
## LSTM Usando la clase de PyTorch
### Set Up
#### Librerias y paquetes

# !pip uninstall -y torch torchvision torchaudio torchtext
# !pip install torch==2.3.1 torchtext==0.18.0 torchvision==0.18.1 torchaudio==2.3.1 numpy datasets tqdm matplotlib
# !python -m spacy download en_core_web_sm

import collections

import datasets # Importar y tratar datos de la base de datos de HuggingFace (https://huggingface.co/datasets)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn # clases y funciones para neural networks (LSTM, RNN...)
import torch.optim as optim
import torchtext # funciones para tratar texto, convertir en tokens, etc.
import tqdm


# Reproducibility: Forzamos a python a usar la misma seed en todos sus métodos, así forzamos reproducibilidad.

seed = 1234
np.random.seed(seed) # Reproducibilidad para numpy
torch.manual_seed(seed) # Reproducibilidad si usamos CPU
torch.cuda.manual_seed(seed) # Reproducibilidad si usamos GPU
torch.backends.cudnn.deterministic = True # Reproducibilidad librería cudnn


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
### Data Loading and Processing

Cargamos los datos de las reviews de la base de datos y los procesamos para el entrenamiento de la red neuronal

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# Data loading:

train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"]) # Cargamos los datos


# Data processing (tokenization)

from torchtext.data.utils import get_tokenizer # Esta función transforma strings de texto en tokens.

def tokenize_example(example, tokenizer, max_length): # Esta función aplica las normas de tokenizer a una review. Devuelve los tokens y la longitud de la review
    tokens = tokenizer(example["text"])[:max_length] # example: diccionario tipo "text": ...
    length = len(tokens) # Número de tokens considerados, <= max_length. Se pierde info si se excede.
    return {"tokens": tokens, "length": length}

tokenizer = torchtext.data.utils.get_tokenizer("basic_english") # Basic english da las normas para partir las str. Separa por espacios/puntuación, transforma mayus en minus, elimina puntuación. También afecta algunos carácteres...
max_length = 200 # Optimización útil. Límite de palabras de una review que se analizarán. Para una review con más de estas palabras, solo se usaran las n = max_length primeras
# .map es una función de la librería datasets que aplica de forma eficiente nuestra función de tokenizar a todos los elementos de la lista train_data sin necesidad de un bucle for.
train_data = train_data.map(
    tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length})
test_data = test_data.map(
    tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length})

# Data split

min_freq = 1 # Frecuencia mínima de palabras. Optimización útil.
# Dividimos la training data en train set (que usaremos para entrenar el modelo) y validation set (que usaremos para validar el modelo)
test_size = 0.25 # Fracción de la training data usada para validar.
train_valid_data = train_data.train_test_split(test_size=test_size) # Esta función de la libreria datasets divide nuestros datos en dos sets para entrenar y validar.
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]

# Unknown characters and padding set up

    # En este bloque trataremos 2 cosas:
    # - Palabras desconocidas, typos o palabras que no aparecen en la libreríe basic_english (unk)
    # - Diferente longitud de las reviews (pad)

special_tokens = ["<unk>", "<pad>"]

from torchtext.vocab import build_vocab_from_iterator
# Esta función (de la librería torchtext) mapea los tokens (palabras) a enteros. Devuelve un objeto Vocab (como un diccionario pero más eficiente)
# vocab es como un traductor palabra->numero/id.

vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["tokens"],
    min_freq=min_freq,
    specials=special_tokens)

unk_index = vocab["<unk>"]
pad_index = vocab["<pad>"]

vocab.set_default_index(unk_index)

# Indexing

    # Esta funcion usa nuestro traductor para transformar una lista de strings en una lista de ids

def numericalize_example(example, vocab):
    ids = vocab.lookup_indices(example["tokens"])
    return {"ids": ids}
    # De nuevo usamos la función map para convertir todas las reviews de nuestro dataset

train_data = train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
valid_data = valid_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
test_data = test_data.map(numericalize_example, fn_kwargs={"vocab": vocab})

# Tensor type

    # Para usar los métodos de PyTorch necesitamos usar estrictamente objetos del tipo PyTorch tensors
    # with_format() es un método de la librería dataset que convierte nuestro dataset en este objeto

train_data = train_data.with_format(type="torch", columns=["ids", "label", "length"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label", "length"])
test_data = test_data.with_format(type="torch", columns=["ids", "label", "length"])

# Padding

# Esta función hace que todas las secuencias en un batch de reviews tengan la misma longitud (necesario para el método de PyTorch)
def get_collate_fn(pad_index): # esta función es un wrapper para pasar el indice asociado al padding.
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch] # Extraemos los ids de las palabras de cada review
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        ) # Esta función coge la review más larga de nuestro batch y añade el indice asociado al padding en las otras reviews hasta que tengan la misma longitud. El output es un tensor de forma (n reviews x longitud de la review mas larga)
        batch_length = [i["length"] for i in batch]
        batch_length = torch.stack(batch_length) # Creamos un vector con la longitud real de cada una (n_reviewsx1)
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label) # Creamos un vector con la label de cada una (n_reviewsx1)
        batch = {"ids": batch_ids, "length": batch_length, "label": batch_label} # Generamos un diccionario con la info
        return batch

    return collate_fn

# Batching

    #Este bloque permite cargar los datos en el modelo por batches de tamaño ```batch_size```. Esta funcion tiene como input el dataset entero, el tamaño de los batches y el indice asociado al padding para ese batch. Shufle es una variable (por defecto en False) que pasaremos a la función DataLoader (La que nos cargará los datos) para indicarle si queremos que coja las reviews aleatoriamente. En entrenamiento queremos que esto este en True y en validation y testing en false


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index) # Llamamos a la función que usará el DataLoader para hacer que todas las reviews tengan la misma longitud
    data_loader = torch.utils.data.DataLoader( #Nuestro dataloader (función de PyTorch).
        dataset=dataset, # El input es el dataset completo
        batch_size=batch_size, #Le indicamos el tamaño de los batches en que ha de separar los datos
        collate_fn=collate_fn, # En el argumento collate_fn le pasamos la función que usará para crear el batch con todas las samples juntas. Si usamos la por defecto simplemente staquearía las reviews. La nuestra hace eso y las hace del mismo tamaño.
        shuffle=shuffle) # Indicamos si ha de coger aleatoriamente

    return data_loader

# En este bloque de código separamos los datos en batches (usaremos minibatch gradient descent, así que necesitamos minibatches):
batch_size = 32

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)

# Data: indexing OK, padding OK, batching OK; packing NO, embedding NO (embedding is also learned!)




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
### LSTM architecture

Definimos la arquitectura de la red neuronal tipo LSTM para el aprendizaje de los datos.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class LSTM(nn.Module): # Creamos nuestra propia clase para el LSTM. Esta clase hereda de PyTorch todas los inbuilt métodos
    def __init__( # En el método init declaramos todas las partes necesarias para que funcione la clase
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index):

    # Estas son las partes de nuestra clase
        super().__init__() # esto inicia las clase de PyTorch de la que heredamos los métodos
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index) # Nuestro traductor: Función de embedding de PyTorch. Coge una palabra y lo transforma en un vector. padding_idx le indica al método que indice se ha usado para rellenar y que lo ignore.

#In a typical neural network for NLP, the embedding layer is just a matrix E of parameters. Each row corresponds to the embedding of a token in the vocabulary(of the datase), and those vectors are learned during training via backpropagation, just like any other weights.
#InitializationThe embedding matrix can be:
    #Randomly initialized (very common)
    #Initialized from pretrained embeddings (like Word2Vec or GloVe) <- OUR CASE
#Connection to linear layers
#If you think of a standard linear layer: y=E^Tx then the embedding layer is a special case where x is constrained to be one-hot and no bias term is used. Since x is one-hot, y gives the embedding of the token x.  



        self.lstm = nn.LSTM( # El método que corre el LSTM. Inbuilt the PyTorch
            embedding_dim, # Tamaño de los vectores de input
            hidden_dim, # dimension de las hidden layers
            n_layers, # numero de hidden layers
            bidirectional=bidirectional, # If bidirectional (forwardpropagation in both left right and right left reading directions)
            dropout=dropout_rate, # Dropout rate for regularization
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate) # Esta función "apaga" un ratio de las "neuronas" durante el entrenamiento para evitar que el modelo aprenda ruido
        # NOTA: Añadir self. antes cuando nuestra clase hereda de pyTorch hace que la librería la tenga en cuenta. Si no lo hacemos pyTorch simplemente ignorará la variable

    def forward(self, ids, length): # Cuando llamamos prediction = model(ids, length) esta función se llama automáticamente
        embedded = self.dropout(self.embedding(ids)) # Aplicamos el "traductor" a nuestras palabras y aplicamos el dropout
        # embedded = [batch size, seq len, embedding dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence( # PACKING: Función de pyTorch que procesa nuestros vectores para que ignore los asociados al padding
            embedded, length, batch_first=True, enforce_sorted=False # usamos length para saber hasta donde llegaba la review
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded) # packed output revela lo que "piensa" el modelo en cada palabra hidden es el último pensamiento
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output) # transforma el packed output en un tensor
        # output = [batch size, seq len, hidden dim * n directions]
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
            # hidden = [batch size, hidden dim * 2]
        else:
            hidden = self.dropout(hidden[-1]) #
            # hidden = [batch size, hidden dim]
        prediction = self.fc(hidden) # Pasa el último pensamient
        # prediction = [batch size, output dim]
        return prediction


vocab_size = len(vocab) # Tamaño de nuestro diccionario de ids-palabra
embedding_dim = 300 # Dimensión de nuestros vectores en el embedding. Esto viene del pretrained model utilizado en el embedding.
hidden_dim = 128
output_dim = len(train_data.unique("label")) # Número final de elecciones. En este caso solo tenemos dos.
n_layers = 2
bidirectional = False
dropout_rate = 0.5

model = LSTM(
    vocab_size,
    embedding_dim,
    hidden_dim,
    output_dim,
    n_layers,
    bidirectional,
    dropout_rate,
    pad_index,
)


# Count how many trainable parameters do we have
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_parameters(model):,} trainable parameters")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
### LSTM training and prediction

Definimos el proceso de aprendizaje y predicción de la red neuronal.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Parameters initialization

    # El siguiente bloque de código se usa para iniciar los pesos del modelo (Parameters = everything the model learns during training Includes: Weights and biases. We do language abuse: weights=params (includes biases)). Cuando usamos ```model.apply(initialize_weights)``` estamos diciendole a PyTorch que aplique esta función a cada una de las partes de nuestra clase


def initialize_weights(m): # NOT REVISED
    if isinstance(m, nn.Linear): # Si la parte es nn.Linear
        nn.init.xavier_normal_(m.weight) # Inicializamos los pesos con una xavier_normal.
        nn.init.zeros_(m.bias) # Inicializamos los bias con zeros para no afectar a los resultados iniciales
    elif isinstance(m, nn.LSTM):# Si la parte es nn.LSTM
        for name, param in m.named_parameters(): # LSTM de PyTorch tiene muchos parametros internos. Hacemos un loop por todos los parametros
            if "bias" in name: # Seleccionamos si es bias e iniciamos con zeros
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.orthogonal_(param) # Seleccionamos si es un peso  iniciamos con una matriz de pesos ortogonal para evitar vanishing/exploding gradient


model.apply(initialize_weights) # iniciamos los pesos del modelo)


# Learning embedding

    #En nuestro modelo usamos embedding para ser más eficientes. Para no tener que entrenar un modelo de 0, usaremos un diccionario pre entrenado conocido como Global Vectors for Word representation. Es un diccionario entrenado con billones de palabras y que mapea las palabras a un espacio 300-dimensional.  Esto da un vector inicial GloVe, que se ajusta con gradientes en el entrenamiento. El embedding también forma parte del proceso del aprendizaje: cada palabra del texto se aprende con un cierto significado, que se refleja en el embedding (así, cada palabra es interpretada con su significado adequado en el contexto del entrenamiento. e.g.: Si entrenamos sobre lectura medieval, "queen" tiene el significado de reina, no del grupo musical. Su embedding refleja este significado, por ejemplo situandolo como el femenino de rey. En un entrenamiento sobre bandas musicales, el significado de "queen" seria totalmente distinto).

#vectors = torchtext.vocab.GloVe() # descarga el diccionario a nuestra ram
vectors = torchtext.vocab.GloVe(name="6B", dim=300) # lighter version


#get_itos() es una función del objeto Vocab. Devuelve una lista de todas las palabras en nuestro datasets ordenadas por id. get_vecs_by_tokens() busca en el diccionario la lista de palabras que se le pasa como argumento y extrae solo los vectores asociados a esas palabras.

pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos()) # Devuelve un tensor con tamaño (n palabras en dataset x 300)

model.embedding.weight.data = pretrained_embedding # Nuestra clase LSTM tiene una parte asociada al embedding. Esto sobrescribe los pesos del embedding (que se ponen aleatoriamente) por los del diccionario


# Optimizer and Loss

# El optimizer que utilizamos es Adam (Adapatative Moment Estimation) que es como un gradient descent pero mucho más optimizado:
    #- Calcula un rolling average de los gradientes previos evitando quedarse atrapado en minimos locales
    #- Calcula dinámicamente un learning rate custom para cada uno de los parámetros

lr = 5e-4
optimizer = optim.Adam(model.parameters(), lr=lr)

# La función de perdida que utilizamos es Cross-Entropy, que aplica softmax a los outputs dados por la red neuronal

criterion = nn.CrossEntropyLoss() # Cross entropy loss.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
model = model.to(device)
criterion = criterion.to(device)


# Training process:

def train(dataloader, model, criterion, optimizer, device):
    model.train() # Le indicamos a la clase que está en modo entreno. Esto le indica a la clase que debe usar Dropout
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(dataloader, desc="training..."):
        ids = batch["ids"].to(device)
        length = batch["length"]
        label = batch["label"].to(device)
        prediction = model(ids, length) # Activamos el forward pass
        loss = criterion(prediction, label) # Loss computation
        accuracy = get_accuracy(prediction, label) # Accuracy prediction (function defined below)
        optimizer.zero_grad() #Hay que setear a 0 el optimizer en cada paso
        loss.backward() # Backpropagation training after each batch
        optimizer.step() # Parameter update
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs) # Returns mean across batches of loss and accuracy

# Prediction process:

def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

# Actual training, epoch loop

training=0
if training==0:
    print("No training")
if training==1:
    n_epochs = 5
    best_valid_loss = float("inf")

    metrics = collections.defaultdict(list)

    for epoch in range(n_epochs):
        train_loss, train_acc = train(
            train_data_loader, model, criterion, optimizer, device
        )
        valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)
        metrics["train_losses"].append(train_loss)
        metrics["train_accs"].append(train_acc)
        metrics["valid_losses"].append(valid_loss)
        metrics["valid_accs"].append(valid_acc)
        if valid_loss < best_valid_loss: # Esto hace que la run ignore los parametros del modelo cuando este empeora. Previene overfitting
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "lstm.pt") # <--HERE: Saving the best trained version of the model during training.
        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")


    # ---- Loss plot ----
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics["train_losses"], label="train loss")
    ax.plot(metrics["valid_losses"], label="valid loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xticks(range(n_epochs))
    ax.legend()
    ax.grid()

    fig.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


    # ---- Accuracy plot ----
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics["train_accs"], label="train accuracy")
    ax.plot(metrics["valid_accs"], label="valid accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_xticks(range(n_epochs))
    ax.legend()
    ax.grid()

    fig.savefig("accuracy_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
### LSTM test
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ---- Load model ----
model.load_state_dict(torch.load("lstm.pt"))

test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)

print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")

# ============================================================
#  GRADIENT × EMBEDDING ANALYSIS ON TEST DATA
# ============================================================
import collections
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from collections import defaultdict
import random

# ============================================================
# SINGLE PHRASE IMPORTANCE (with Prediction in Plot)
# ============================================================

def plot_phrase_importance(model, phrase_ids, phrase_length, vocab, device, phrase_text=None, real_label=None, max_words_total=250):
    model.eval()

    # Optional logic: Truncate if phrase is too long for the LSTM memory
    if phrase_ids.size(0) > max_words_total:
        phrase_ids = phrase_ids[:max_words_total]
        phrase_length = torch.tensor(max_words_total)

    ids = phrase_ids.unsqueeze(0).to(device)
    length = phrase_length.unsqueeze(0)

    # -------------------------
    # embeddings with gradients
    # -------------------------
    embedded = model.embedding(ids) # We call the learned embeddings by the model for the considered words (its ids)
    embedded.requires_grad_(True) # We require to compute the gradients of the prediction wrt the embeddings (sensitivity used for computing importance)
    embedded.retain_grad() # We retain the gradients such that are not erased after forwardpropagation prediction

    packed = rnn_utils.pack_padded_sequence(
        embedded,
        length.cpu(),
        batch_first=True,
        enforce_sorted=False
    ) # We simply pack the considered words the importance of which we want to compute in the considered phrase, to forwardpropagate them in LSTM.

    packed_out, (hidden, cell) = model.lstm(packed) # Forwardpropagation of the words in phrase through LSTM NN

    if model.lstm.bidirectional:
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=-1)
    else:
        hidden = hidden[-1]

    hidden = model.dropout(hidden)
    logits = model.fc(hidden)

    # Prediction logic
    probs = torch.softmax(logits, dim=1) # Explicit softmax for the outputs, not done in the LSTM NN (done automatically in loss, not in prediciton)
    pred_class = torch.argmax(logits, dim=1).item() # Choose class with largest logit (equivalent to choosing that of largest softmax, which is an increasing function)
    pred_text = "POSITIVE" if pred_class == 1 else "NEGATIVE"
    confidence = probs[0][pred_class].item() * 100 # Probability for the class that has been chosen
    # (this logic is a little intrincate for a binary classif., better understood in a more than 2 class classif.)
    

    # Real label logic
    if real_label is not None:
        real_text = "POSITIVE" if real_label == 1 else "NEGATIVE"
        match_status = "CORRECT" if pred_class == real_label else "INCORRECT"
    else:
        real_text = "Unknown"
        match_status = ""

    score = logits[:, pred_class]
    model.zero_grad()
    score.backward()

    importance = (embedded.grad * embedded).sum(dim=-1).squeeze(0)
    tokens = [vocab.get_itos()[i] for i in ids.squeeze(0).tolist()]
    data = [(t, v.item()) for t, v in zip(tokens, importance) if t != "<pad>"]
    words, values = zip(*data)

    # -------------------------
    # PLOT
    # -------------------------
    plt.figure(figsize=(14, 6)) # Slightly wider for long phrases

    # Use plot instead of bar
    plt.plot(range(len(values)), values, marker='o', markersize=4, linestyle='-',
             color='blue' if pred_class == 1 else 'red', linewidth=1.5, alpha=0.7)

    plt.axhline(0, color='black', linestyle='--', alpha=0.3)

    # --- SMART LABELING LOGIC ---
    if len(words) > 16:
        # Calculate threshold for "remarkable" peaks
        # Remarkable = Absolute value is > mean + 1.5 * std_dev <- JUST TAKE IMPORTANCES WHICH ARE OFF THE MEAN BY 1.5 TIMES s.d
        vals_array = np.array(values)
        threshold = np.mean(np.abs(vals_array)) + 1.5 * np.std(np.abs(vals_array))
        
        tick_indices = []
        tick_labels = []
        
        for i, (w, v) in enumerate(zip(words, values)):
            if abs(v) >= threshold:
                tick_indices.append(i)
                tick_labels.append(w)
        
        plt.xticks(tick_indices, tick_labels, rotation=45, fontsize=9)
        plt.xlabel(f"Showing only top {len(tick_labels)} influential words (Length > 16)")
    else:
        # Standard behavior for short phrases
        plt.xticks(range(len(words)), words, rotation=45)

    plt.ylabel("Gradient × Embedding importance")

    title_str = (f"Pred: {pred_text} ({confidence:.1f}%) | Real: {real_text}\n"
                 f"{match_status}\n"
                 f"{phrase_text[:100] + '...' if phrase_text and len(phrase_text) > 100 else phrase_text}")

    plt.title(title_str)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    plt.savefig("word_importance_filtered.png", dpi=300)
    plt.close()

    return data, pred_class

# Example execution for single phrase:
#phrase = "The shot selection was horrid, but in general the movie is the greatest ever."
#phrase = "I loved the movie, but in a technical sense it is a crap."
#phrase = "This is the best action film of the year."
#phrase= "I thought this movie would be a complete disaster but the actors were fantastic."
#phrase= "Terrible terrible terrible terrible."
#phrase = "The worst film ever"
#phrase = "Great acting and not horrible overall"
#phrase = "Great acting and horrible overall"

#tokens = tokenizer(phrase)
#ids = torch.tensor(vocab.lookup_indices(tokens))
#length = torch.tensor(len(ids))
#plot_phrase_importance(
    #model,
    #ids,
    #length,
    #vocab,
    #device,
    #phrase_text=phrase
#)


# ---------------------------------------------------------
# FIND RANDOM EXAMPLE WITH SPECIFIC PREDICTION LOGIC
# ---------------------------------------------------------

def get_random_example_by_prediction(dataset, model, vocab, device, positivity=True, max_words=100):
    # Determine what we are looking for: 1 for Positive, 0 for Negative
    target_class = 1 if positivity else 0
    
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    model.eval()
    print(f"Searching for a review where prediction == {target_class}...")

    with torch.no_grad():
        for idx in indices:
            example = dataset[idx]
            
            # Check length constraint
            if len(example['ids']) < max_words:
                ids = example['ids'].unsqueeze(0).to(device)
                length = example['length'].unsqueeze(0)
                
                # Get model prediction
                logits = model(ids, length)
                pred_class = torch.argmax(logits, dim=1).item()
                
                # Logic: Match the target prediction
                if pred_class == target_class:
                    print(f"Found match at index {idx}! (Length: {len(example['ids'])})")
                    return example
                    
    print("No matching example found within constraints.")
    return dataset[0]

# ==========================================
# EXECUTION
# ==========================================

# SET YOUR LOGIC HERE
positivity = True 

# Get the example based on the logic
example = get_random_example_by_prediction(
    test_data, 
    model, 
    vocab, 
    device, 
    positivity=positivity, # Uses our logic flag
    max_words=150          # Long enough to test "remarkable peaks"
)

# Decode for the plot title
raw_tokens = [vocab.get_itos()[i] for i in example['ids'] if vocab.get_itos()[i] != '<pad>']
readable_text = " ".join(raw_tokens)

# Call the plotting function
plot_phrase_importance(
    model,
    example['ids'],
    example['length'],
    vocab,
    device,
    phrase_text=readable_text,
    real_label=example['label'].item()
)


# ============================================================
# GLOBAL ANALYSIS (Limited to 500 Reviews)
# ============================================================

"""""""""""""""""""""
NOT WORKING PROPERLY
"""""""""""""""""""""

from collections import Counter

def error_analysis_word_importance(model, data_loader, vocab, device, max_reviews=500, top_n=15):
    model.eval()
    
    # The 4 groups for word storage
    groups = {
        "True Positive": [],  # Real: Pos, Pred: Pos
        "False Negative": [], # Real: Pos, Pred: Neg
        "True Negative": [],  # Real: Neg, Pred: Neg
        "False Positive": []  # Real: Neg, Pred: Pos
    }
    
    reviews_count = 0
    print(f"Starting Error Analysis (Limit: {max_reviews} reviews)...")

    for batch in data_loader:
        if reviews_count >= max_reviews: break
        
        ids_batch = batch["ids"].to(device)
        length_batch = batch["length"]
        labels_batch = batch["label"].to(device)

        for i in range(ids_batch.size(0)):
            if reviews_count >= max_reviews: break
            
            ids = ids_batch[i].unsqueeze(0)
            length = length_batch[i].unsqueeze(0)
            real_label = labels_batch[i].item()

            # Forward + Gradient Setup
            embedded = model.embedding(ids)
            embedded.requires_grad_(True)
            embedded.retain_grad()

            packed = rnn_utils.pack_padded_sequence(
                embedded, length.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, (hidden, cell) = model.lstm(packed)
            
            if model.lstm.bidirectional:
                hidden = torch.cat([hidden[-1], hidden[-2]], dim=-1)
            else:
                hidden = hidden[-1]

            logits = model.fc(model.dropout(hidden))
            pred_class = torch.argmax(logits, dim=1).item()

            # Backward relative to prediction
            score = logits[:, pred_class]
            model.zero_grad()
            score.backward()

            # Importance calculation
            importance = (embedded.grad * embedded).sum(dim=-1).squeeze(0)
            tokens = [vocab.get_itos()[idx] for idx in ids.squeeze(0).tolist()]
            values = importance.detach().cpu().tolist()
            
            # Filter out padding
            data = [(t, v) for t, v in zip(tokens, values) if t != "<pad>"]
            if not data: continue
            
            # Logic for choosing the "Most Influential" word
            # Grouping and selecting words based on your criteria
            if real_label == 1 and pred_class == 1:
                # True Positive: Word with highest importance (pushed to Pos)
                influence_word = max(data, key=lambda x: x[1])[0]
                groups["True Positive"].append(influence_word)
            
            elif real_label == 1 and pred_class == 0:
                # False Negative: Word with lowest importance (pushed to Neg)
                influence_word = min(data, key=lambda x: x[1])[0]
                groups["False Negative"].append(influence_word)
                
            elif real_label == 0 and pred_class == 0:
                # True Negative: Word with lowest importance (pushed to Neg)
                influence_word = min(data, key=lambda x: x[1])[0]
                groups["True Negative"].append(influence_word)
                
            elif real_label == 0 and pred_class == 1:
                # False Positive: Word with highest importance (pushed to Pos)
                influence_word = max(data, key=lambda x: x[1])[0]
                groups["False Positive"].append(influence_word)

            reviews_count += 1
            
    # ============================================================
    # PLOTTING THE 4 HISTOGRAMS
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Top Influential Words by Prediction Group (N={reviews_count} reviews)", fontsize=16)
    
    colors = {"True Positive": "green", "False Negative": "orange", 
              "True Negative": "red", "False Positive": "purple"}

    for i, (group_name, word_list) in enumerate(groups.items()):
        ax = axes[i//2, i%2]
        
        if not word_list:
            ax.text(0.5, 0.5, "No examples found", ha='center')
            ax.set_title(group_name)
            continue
            
        # Find the N most frequent words in this group
        word_counts = Counter(word_list).most_common(top_n)
        words, counts = zip(*word_counts)
        
        ax.bar(words, counts, color=colors[group_name])
        ax.set_title(f"{group_name} (n={len(word_list)})")
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel("Frequency of Influence")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("error_analysis_histograms.png", dpi=300)
    plt.close()
    
    return groups

# Execute
analysis_results = error_analysis_word_importance(model, test_data_loader, vocab, device, max_reviews=500)
