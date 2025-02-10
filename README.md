# TinySwallow Llama.cpp

TinySwallow-1.5B-Instruct, a Japanese instruction-tuned language model, leverages TAID knowledge distillation from Qwen2.5-32B-Instruct, improving instruction following and conversational abilities.

## Project Structure

The project is structured as follows:

- `app.py`: The file containing the main gradio application.
- `logger.py`: The file containing the code for logging the application.
- `exception.py`: The file containing the code for custom exceptions used in the project.
- `requirements.txt`: The file containing the list of dependencies for the project.
- `LICENSE`: The license file for the project.
- `README.md`: The README file that contains information about the project.
- `assets`: The folder containing screenshots for working on the application.
- `.gitignore`: The file containing the list of files and directories to be ignored by Git.

## Tech Stack

- Python (for the programming language)
- Llama.cpp (llama-cpp-python as Python binding for llama.cpp)
- Hugging Face Hub (for the GGUF model)
- Gradio (for the main application)

## Getting Started

To get started with this project, follow the steps below:

1. Clone the repository: `git clone https://github.com/sitamgithub-MSIT/TinySwallow-llamacpp.git`
2. Change the directory: `cd TinySwallow-llamacpp`
3. Create a virtual environment: `python -m venv tutorial-env`
4. Activate the virtual environment: `tutorial-env\Scripts\activate`
5. Install the required dependencies: `pip install -r requirements.txt`
6. Run the Gradio application: `python app.py`

Now, open up your local host and see the web application running. For more information, please refer to the Gradio documentation [here](https://www.gradio.app/docs/interface). Also, a live version of the application can be found [here](https://huggingface.co/spaces/sitammeur/TinySwallow-llamacpp).

## Contributing

Contributions are welcome! If you would like to contribute to this project, please raise an issue to discuss the changes you want to make. Once the changes are approved, you can create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions about the project, please contact me on my GitHub profile.

Happy coding! 🚀
