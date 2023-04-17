import glob
import os
from pathlib import Path

import git
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake

STORAGE_DIR =  Path(os.path.dirname(os.path.abspath(__file__))) / "storage"
GITHUB_CLONE_DIR = STORAGE_DIR / "github_clones"

DEFAULT_GLOBS = (
    '*.py', '*.js', '*.java', '*.cpp', '*.c', '*.rb', '*.go', '*.php', '*.rs', '*.sh', '*.swift', '*.ts', '*.cs')


def get_github_repo_retriever(url, globs=DEFAULT_GLOBS):
    """Clone a repo from github and index it"""
    repo_path = clone_repo(url)

    return load_repo(repo_path, globs)


def create_github_clone_dir():
    """Create the github clone directory"""
    if not os.path.exists(GITHUB_CLONE_DIR):
        os.makedirs(GITHUB_CLONE_DIR)


def clone_repo(url):
    """Clone a repo from github"""
    create_github_clone_dir()
    repo_name = url.split("/")[-1]

    local_path = os.path.join(GITHUB_CLONE_DIR, repo_name)

    if os.path.exists(local_path):
        return local_path

    git.Repo.clone_from(url, local_path)

    print(f"Repo cloned to {local_path}.")

    return local_path


def load_repo(name, globs):
    print('Loading repo...')
    path = Path(os.path.join(GITHUB_CLONE_DIR, name))

    if not path.exists():
        return 'Repo directory does not exist!'

    print('Loading docs...')
    docs = load_docs(path, globs)

    print('Splitting texts...')
    texts = split_texts(docs)

    print('Indexing documents...')
    index = index_documents(texts)

    return index.as_retriever()


def index_documents(texts):
    db = DeepLake.from_documents(texts, OpenAIEmbeddings())
    return db


def split_texts(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)


def load_docs(path, globs):
    docs = []
    for extension in globs:
        for file_path in glob.glob(f"{path}/**/{extension}", recursive=True):
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass

    return docs
