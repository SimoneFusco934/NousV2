# NousV2

**Una libreria di Machine Learning in C++ sviluppata interamente da zero, senza dipendenze esterne**

---

## 🔧 Descrizione

**NousV2** è una libreria C++ moderna per costruire, addestrare e salvare reti neurali profonde. È stata realizzata interamente da zero, **senza l'uso di librerie esterne**, per dimostrare un controllo totale sulla logica di training, inferenza e gestione dei modelli neurali.

---

## 🌟 Caratteristiche principali

- ✅ Completamente scritta in C++ (senza dipendenze esterne)
- ✅ Supporta layer:
  - `Dense` (fully-connected)
  - `Conv` (convolutional)
  - `MaxPooling`
- ✅ Funzionalità end‑to‑end:
  - Costruzione del modello
  - Addestramento
  - Salvataggio
  - Inferenza
- ✅ Architettura modulare
- ✅ Esempi di utilizzo già pronti

---

## 📦 Contenuto del repository

| File / Cartella     | Descrizione                                                    |
|---------------------|----------------------------------------------------------------|
| `cppFiles/`         | Codice sorgente della libreria (file cpp)                      |
| `hppFiles/`         | Codice sorgente della libreria (headers)                       |
| `CreateModel.cpp`   | Crea, allena e salva un modello                                |
| `UseModel.cpp`      | Carica e utilizza un modello salvato per fare inferenza        |
| `Makefile`          | Script di compilazione                                         |
| `README.md`         | Documentazione (questo file)                                   |

---

## 🚀 Come testare la libreria

1. **Clona il repository**:
    ```bash
    git clone https://github.com/SimoneFusco934/NousV2.git
    cd NousV2
    ```

2. **Compila il progetto**:
    ```bash
    make
    ```

3. **Esegui**:
    - Addestra e salva un modello:
        ```bash
        ./CreateModel
        ```
    - Carica il modello e usalo:
        ```bash
        ./UseModel
        ```

---

## 🧠 Esempio d’uso

### Creazione del modello (`CreateModel.cpp`)

```cpp
Model model;
model.add(Dense(input_dim, output_dim));
model.add(Conv(...));
model.add(MaxPooling(...));
model.setLossFunction("MSE");
model.setOptimizer("SGD");
model.train(dataset, labels, epochs, batch_size);
model.save("model_path");

