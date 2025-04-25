# Elon Musk Tweets Analysis

![Project Banner](https://img.shields.io/badge/Python-3.10-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## 📖 Overview

This project analyzes Elon Musk's tweets from June 4, 2010, to March 23, 2025, to uncover trends, topics, and insights using advanced natural language processing (NLP) techniques. The dataset, sourced from Kaggle, is processed with Python, leveraging libraries like BERTopic, Transformers, and NLTK to perform topic modeling, text cleaning, and visualization.

The core objective is to classify tweets into meaningful topics using the Llama 3.2 model, visualize topic distributions, and explore temporal trends in Musk's Twitter activity. The project is executed in a Jupyter Notebook environment with CUDA support for GPU-accelerated computations.

## 🚀 Features

- **Data Cleaning**: Removes URLs, mentions (except preserved accounts like @Tesla, @SpaceX), emojis, and stopwords for cleaner text analysis.
- **Topic Modeling**: Uses BERTopic with Llama 3.2 to identify 912 distinct topics from 44,900 cleaned tweets.
- **Visualization**: Generates hierarchical clustering, topic distribution scatter plots (e.g., "Documents and Topics"), and temporal trend charts using Plotly and Matplotlib.
- **Trends Analysis**: Examines topic evolution over time with dynamic visualizations, focusing on key topics like "grok_image_xai" and "tesla_tunes_wipers".
- **GPU Support**: Leverages CUDA and NVIDIA Tesla T4 for efficient processing of large datasets.

## 📂 Project Structure

```
elon-musk-tweets-analysis/
│
├── elon-musk-llm.ipynb        # Main Jupyter Notebook with analysis
├── README.md                  # Project documentation
├── /data/                     # Dataset (not included, sourced from Kaggle)
├── /output/                   # Generated visualizations and model outputs
│   ├── topic_model_Llama/     # Saved BERTopic model
│   └── *.pdf                  # Visualizations (e.g., topic distributions)
└── requirements.txt           # Python dependencies
```

## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/elon-musk-tweets-analysis.git
   cd elon-musk-tweets-analysis
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset**:
   - Obtain the dataset from [Kaggle](https://www.kaggle.com/datasets/7006744/elon-musks-tweet).
   - Place it in the `/data/` directory or update the notebook's file path accordingly.

5. **Hugging Face Authentication**:
   - Create a Hugging Face account and generate an API token.
   - Replace the token in the notebook:
     ```python
     hf_token = "your_hugging_face_token"
     ```

6. **GPU Setup (Optional)**:
   - Ensure CUDA is installed (`nvcc --version`).
   - Install PyTorch with CUDA support:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
     ```

## 📊 Usage

1. **Run the Notebook**:
   - Open `elon-musk-llm.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute cells sequentially to:
     - Install dependencies.
     - Load and clean the dataset (53,923 tweets, reduced to 44,900 after cleaning).
     - Perform topic modeling with BERTopic and Llama 3.2, identifying 912 topics.
     - Generate visualizations (e.g., scatter plots of topic distributions, hierarchical clustering, and trends over time).
     - Save outputs (model, plots) to the `/output/` directory.

2. **Explore Outputs**:
   - **Topic Model**: Saved in `/output/topic_model_Llama/`.
   - **Visualizations**: Includes a scatter plot ("Documents and Topics") showing topic clusters, with topics like "grok_image_xai" and "tesla_tunes_wipers", saved as PDFs and interactive HTML plots in `/output/`.
   - **Zipped Model**: `topic_model_Llama_3.2.zip` for easy sharing.

3. **Customize**:
   - Adjust `min_topic_size` in BERTopic for different topic granularities.
   - Modify `preserved_accounts` in the cleaning function to retain specific mentions.
   - Experiment with visualization parameters (e.g., `n_words`, `topics`) to focus on specific topics like "video_edgey_swipe" or "raptor_thrust_merlin".

## 🔍 Key Insights

- **Topic Diversity**: Identified 912 topics, including "grok_image_xai" (related to AI and xAI), "spacex_package_51" (SpaceX missions), and "tesla_tunes_wipers" (Tesla features).
- **Sentiment and Tone**: Frequent use of positive descriptors ("great_nods_oh", "thanks_luck_yud") and casual expressions ("yup," "haha").
- **Temporal Trends**: Topics like "starfactory_installer_adopt" and "astronauts_returning_mission" show increased activity over time, reflecting SpaceX milestones.
- **Data Quality**: Handled missing values and mixed data types effectively, ensuring robust analysis.

## 📈 Results and Visualizations

Below are examples of visualizations generated by the project:

- **Topic Distribution Scatter Plot**: "Documents and Topics" visualizes 912 topics in 2D space, with clusters like "grok_image_xai" and "tesla_tunes_wipers".  
  ![Documents and Topics Scatter Plot](https://via.placeholder.com/1200x750.png?text=Documents+and+Topics+Scatter+Plot)  
  *Note*: Replace the placeholder link with the actual path to the scatter plot image in `/output/` (e.g., `/output/documents_and_topics.png`).

- **Extracted Topics**: A screenshot of the extracted topics from the topic modeling process.  
  ![Extracted Topics Screenshot](https://i.postimg.cc/W4tHgtSZ/newplot.png)
  *Note*: Replace the placeholder link with the actual path to your screenshot (e.g., `/output/extracted_topics_screenshot.png`).

- **Hierarchical Clustering**: Shows relationships between topics.
- **Topics Over Time**: Tracks topic prevalence from 2010 to 2025, focusing on trends in topics like "video_edgey_swipe" and "raptor_thrust_merlin".

To generate these, run the visualization cells in the notebook.

## 🧑‍💻 Dependencies

Key libraries used in the project:

```text
pandas
numpy
matplotlib
seaborn
plotly
nltk
bertopic
transformers
sentence-transformers
torch
umap-learn
tqdm
missingno
```

Install them using:
```bash
pip install -r requirements.txt
```

## ⚠️ Notes

- **Hugging Face Token**: Required for Llama 3.2 access. Ensure your token is valid.
- **GPU Requirement**: The notebook is optimized for CUDA-enabled GPUs (e.g., NVIDIA Tesla T4). CPU execution is slower.
- **Dataset Size**: The dataset contains 53,923 tweets, with 44,900 cleaned tweets used for topic modeling.
- **Model Saving**: The BERTopic model is saved using `safetensors` for efficient storage.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 🙌 Acknowledgments

- **Dataset**: Sourced from [Kaggle](https://www.kaggle.com/datasets/7006744/elon-musks-tweet).
- **Libraries**: Thanks to the developers of BERTopic, Transformers, and Plotly.
- **Hugging Face**: For providing access to Llama 3.2 and other models.

## 📬 Contact

For questions or suggestions, feel free to open an issue or contact me at [tonmoycse98@gmail.com](mailto:your.email@example.com).

⭐ **Star this repository if you find it useful!**
