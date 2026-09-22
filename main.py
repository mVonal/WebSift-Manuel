"""

main.py - CLI araştırma ajanı.
Kullanım:
    python main.py --url <URL> --question "soru"
    python main.py --url <URL1> --url <URL2> --question "soru"

İşleyiş: Verilen URL'leri indeksle (Write) -> İlgili chunk'ları çek (Select)
    -> Sadece onlarla cevapla (Tüm sayfa değil sadece ilgili parçalar).

"""

import os
import argparse
from openai import OpenAI
from indexer import index_urls
from search import search

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")


def ask(question):
    chunks = search(question)
    sources = "\n\n".join(chunks)
    prompt = f"""Answer the question using only the sources below. If the answer is not in the sources, say so.search

    SOURCES:
    {sources}

    QUESTION:
    {question}
    """
    resp = client.chat.completions.create(
        model = "openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content, resp.usage.prompt_tokens



def main ():
    parser = argparse.ArgumentParser(description="Research Agent")
    parser.add_argument("--url", action="append", required=True,
                        help="Source URL (use again for more than one resource)")
    parser.add_argument("--question", required=True, help="Question that will be asked")
    parser.add_argument("--k", type=int, default=3, help="How many chunk should be fetched?")
    args = parser.parse_args()

    print("Sources are indexed...")
    n = index_urls(args.url)
    if n == 0:
        print("No source was indexed, exiting...")
        return

    print("\n Generating Answer...\n")
    answer, tokens = ask(args.question)
    print("=" * 55)
    print("Answer:")
    print(answer)
    print("=" * 55)
    print(f"[input tokens: {tokens}]")


if __name__ == "__main__":
    main()