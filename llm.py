from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from huggingface_hub import hf_hub_download

# Callbacks support token-wise streaming

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# creation of a floder for the model to be dowloaded from https://huggingface.co/ 

MODELS_PATH = "./models"
model_path = hf_hub_download(   
    repo_id= "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    resume_download=True,
    cache_dir=MODELS_PATH,)


# setting up the parameters for the model 

kwargs = {
    "model_path": model_path,
    "temperature": 0.7,
    "top_p" : 1,
    "n_ctx": 2048,
    "callback_manager" : callback_manager,
    "max_tokens": 100,
    "verbose" : True, 
    "n_batch": 512,  # set this based on your GPU & CPU RAM
}
# setting up promt template

prompt_template = "<s>[INST] "+"""You are a helpful robot assistant,
you will answer user questions by thinking step by step. 
Give out short answers. 
Human: {question}
Assistant:"""+" [/INST]"

prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

# setting up the llm as llama

llm = LlamaCpp(**kwargs)

# calling the chain

llm_chain = LLMChain(prompt=prompt, llm=llm)
while True:
    question = input("\nEnter a query: ")
    ans = llm_chain.run(question)
    print(ans)
