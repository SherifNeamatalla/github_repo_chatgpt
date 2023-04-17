from dotenv import load_dotenv

from github_repo_qa_chain import GithubRepoQAChain

load_dotenv()

url = input("Enter the github repo url: ")
chain = GithubRepoQAChain(url)

while True:
    result = chain.run(input("Enter your query: "))
    print(result)
