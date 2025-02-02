# webpage-chat-ollama

A web-based question-answering application built with Streamlit, LangChain, and Ollama. This project enables anyone to provide a webpage URL, crawl its contents, and interactively query the text with natural language questions. For further experimentation, you can choose between the Llama 3.3 and DeepSeek r1 models.

## Features
- **Webpage Crawling**: Automatically fetch and parse content from a web URL.
- **Text Splitting**: Uses RecursiveCharacterTextSplitter to split large webpage content into smaller chunks for better indexing and retrieval performance.
- **Vector Store**: Leverages an in-memory vector store to quickly retrieve the most relevant segments of text.
- **Question-Answering**: Uses Ollama’s language model to provide concise answers to user queries based on the retrieved content.
- **Interactive Chat**: Offers a chat-style interface built on Streamlit for seamless interactions.

## Getting Started

### Prerequisites

1. **Python 3.8+**
Make sure you have a recent version of Python installed on your machine.
2. **Ollama**
This app depends on Ollama’s language models. Refer to Ollama’s documentation for instructions on how to install and configure it.
3. **DeepSeek-R1-Distill-Qwen-14B Model**
The default model used is deepseek-r1:14b since it balances good results with reasonable performance on most laptops. You can find more details about this model on the Ollama deepseek-r1 Website. Make sure you have it set up and accessible to Ollama. You can download the model by running:

	`ollama pull DeepSeek-R1-Distill-Qwen-14B`

### Installation

1. Clone this repository:

	`git clone https://github.com/jkanalakis/webpage-chat-ollama.git`

2. Navigate to the project directory:

	`cd webpage-chat-ollama`

3. Install Python dependencies (example shown with pip):

	`pip install -r requirements.txt`

If you use a different package manager or a virtual environment, adapt as needed.

## Running the Application

1. Start the Streamlit app:

	`streamlit run web_chat.py`

Replace web_chat.py with the actual filename you used in your code if it differs.

2. Open the interface:

After running the command, Streamlit will provide a local URL (e.g., http://localhost:8501). Open it in your browser.

3. Enter a Web URL:
Provide the URL of the webpage you want to crawl. The application will process and index the webpage in the background.

4. Ask Questions:
- Use the text input box at the bottom of the chat interface to type your question.
- The app will display both the user query and the generated answer.
- If relevant content cannot be found, the system may respond that it does not know the answer.

## Customization

- Prompt Template: Edit the PROMPT_TEMPLATE in web_chat.py to change how the model answers questions.
- Chunk Size: Adjust chunk_size and chunk_overlap in the text splitter for different webpage sizes or more fine-grained search.
- Embeddings and Model: Swap out the Ollama embeddings or model for other language models, depending on your needs.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with suggested improvements or bug fixes.

## License

This project is open-sourced under the MIT License. Feel free to use and modify this code.

Happy exploring, and enjoy discovering insights from the web with Ollama!
