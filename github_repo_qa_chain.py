from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from github_loader_util import get_github_repo_retriever

NAME = "GithubRepoQAChain"

DEFAULT_MODEL = 'gpt-3.5-turbo'


class GithubRepoQAChain:

    def __init__(self, url):
        self.message_history = []
        self.chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(
            model=DEFAULT_MODEL,
        ), retriever=get_github_repo_retriever(url))

    def run(self, user_input: str, **kwargs) -> (str, str):
        result = self.chain({
            "question": user_input,
            "chat_history": self.message_history
        })

        self.message_history.append((user_input, result['answer']))

        return result['answer']
