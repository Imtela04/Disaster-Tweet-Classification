# Disaster Tweet Classification using NLP



### 🎯 Project Overview
This project uses three recurrent neural network architectures to classify tweets from Twitter and determine whether they're about real natural disasters. The models analyze tweet text to binary classify them into:
- **Class 1**: Tweet is about a real natural disaster
- **Class 0**: Tweet is not about a real natural disaster

### 📊 Dataset

The dataset is sourced from Kaggle's "Natural Language Processing with Disaster Tweets" competition.

**Dataset Structure:**
- Original columns: `id`, `text`, `location`, `keyword`, `target`
- Used columns: `text` (tweet content) and `target` (classification label)
- Train-test split: 80-20 ratio

**Preprocessing Steps:**
1. **Tokenization**: Split tweets into words, converted to lowercase, removed punctuation
2. **Padding**: Ensured uniform input length by padding shorter sequences
3. **Embedding**: Mapped tokens to dense vectors for model input

### 🤖 Models Implemented

#### 1. Long Short-Term Memory (LSTM)
- Captures sequential dependencies using three gates: Forget, Input, and Output
- Architecture: Embedding Layer → LSTM Layer → Dense Output Layer
- Uses sigmoid activation for binary classification

#### 2. Bi-directional LSTM (Bi-LSTM)
- Enhanced LSTM with bidirectional processing
- Captures context from both past and future sequences
- Outputs only the final layer result

#### 3. Gated Recurrent Unit (GRU)
- Simplified alternative to LSTM with fewer parameters
- Faster training time due to reduced complexity
- Similar performance characteristics to LSTM

### ⚙️ Model Configuration

**Common Hyperparameters:**
- Vocabulary size: 19,416
- Dropout rate: 0.2 (20% layer deactivation)
- Dense layer neurons: 24
- Loss function: Binary Cross-Entropy
- Optimizer: Adam
- Embedding: Custom embedding matrix

### Key Features
- **Word Embeddings**: Pre-trained GloVe vectors (glove.6B.50d.txt)
- **Text Preprocessing**: Tokenization with out-of-vocabulary handling
- **Sequence Padding**: Uniform sequence length processing
- **Early Stopping**: Prevents overfitting during training
- **Comprehensive Evaluation**: Accuracy and F1-score metrics

### Data Preprocessing Pipeline
1. **Dataset Loading**: Automatic search across system drives for train.csv
2. **Text Cleaning**: Column selection and renaming
3. **Visualization**: Word cloud generation for data exploration
4. **Tokenization**: Convert text to sequences with vocabulary limit of 500 words
5. **Padding**: Uniform sequence length using post-padding strategy
6. **Train-Test Split**: 80-20 split with random state for reproducibility

### Model Training Configuration
- **Epochs**: 30 (with early stopping)
- **Optimizer**: Adam
- **Loss Function**: Binary crossentropy
- **Batch Processing**: Automated by Keras
- **Validation**: Real-time validation during training
- **Callbacks**: Early stopping with patience=3 on validation loss

### 📈 Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| LSTM | 55.75% | 61.36% |
| Bi-LSTM | **77.81%** | **74.02%** |
| GRU | 55.75% | 61.36% |

### 🔍 Key Findings

- **Bi-LSTM** significantly outperformed both LSTM and GRU models
- LSTM and GRU showed signs of training issues, possibly:
  - Incorrect hyperparameter tuning
  - Overfitting or underfitting
  - Limited learning beyond the first epoch
- Bi-LSTM's bidirectional nature allowed it to capture more contextual information

### Analysis

The superior performance of Bi-LSTM (77.81% accuracy) can be attributed to its ability to process sequences in both directions, capturing more comprehensive context. The identical poor performance of LSTM and GRU (55.75%) suggests these models may have benefited from:
- Customized hyperparameters instead of shared configuration
- Additional regularization techniques
- Different architecture depth or complexity

### Evaluation Metrics
- **Accuracy**: Training and validation accuracy tracking
- **F1-Score**: Precision-recall harmonic mean
- **Loss Tracking**: Training and validation loss monitoring
- **Visualization**: Performance comparison charts

### Results Visualization
The project includes comprehensive visualization:
- Training vs. validation accuracy plots for each model
- Training vs. validation loss curves
- Model performance comparison bar charts (accuracy and F1-score)
- Word cloud visualization of the dataset


### Dependencies
- **Core Libraries**: numpy, pandas, matplotlib
- **Text Processing**: wordcloud, sklearn
- **Deep Learning**: tensorflow, keras
- **Preprocessing**: tensorflow.keras.preprocessing

### 🚀 Getting Started

#### Prerequisites
```bash
python >= 3.7
tensorflow/keras
numpy
pandas
scikit-learn
matplotlib
wordcloud
```

#### Installation
```bash
git clone https://github.com/Imtela04/disaster-tweet-classification.git
cd disaster-tweet-classification
pip install -r requirements.txt
```

### Usage Instructions
1. Ensure all dependencies are installed
2. Download the required dataset (train.csv) and GloVe embeddings (glove.6B.50d.txt)
3. Run the Jupyter notebook cells sequentially
4. The system will automatically search for the dataset across available drives
5. Models will be trained and evaluated automatically
6. Results will be displayed through various visualizations

### Model Comparison
The project provides direct comparison between:
- **LSTM**: Traditional sequential processing
- **Bi-LSTM**: Bidirectional context understanding
- **GRU**: Simplified architecture with fewer parameters

Each model's performance is evaluated and compared using validation accuracy and F1-score metrics.

### Technical Implementation Details
- **Embedding Matrix**: Custom creation using pre-trained GloVe vectors
- **Sequence Processing**: Post-truncation and post-padding strategies
- **Model Architecture**: Consistent embedding and dropout layers across models
- **Performance Tracking**: Comprehensive history logging for analysis

### 📚 References

- [Natural Language Processing with Disaster Tweets - Kaggle Competition](https://www.kaggle.com/competitions/nlp-getting-started/overview)

### Research Context
This project is developed as part of CSE440 coursework at BRAC University and follows academic standards for deep learning research in natural language processing, specifically disaster tweet classification tasks.

### 📄 License

This project is part of academic coursework at BRAC University.

### 🤝 Contributing

This is an academic project, but suggestions and feedback are welcome!

### 📧 Contact

For questions or collaboration: imtela@gmail.com


*Developed as part of Natural Language Processing coursework at BRAC University, Dhaka, Bangladesh*

*Note: Ensure that the required dataset files (train.csv and glove.6B.50d.txt) are available on your system before running the notebook.*
