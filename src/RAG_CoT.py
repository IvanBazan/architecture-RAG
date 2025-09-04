import os
from typing import Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

PROMPT_TEMPLATE_CoT= (
    """<|system|>
    Ты — экспертный русскоязычный ассистент,который сначала размышляет, а потом отвечает.
    Отвечай ТОЛЬКО на основе предоставленного контекста. 
    Если информации в контексте недостаточно - честно скажи об этом.
    Всегда пиши свои шаги. Отвечай строго по схеме: анализ → вывод → ответ.
    <|end|>

    <|user|>
    Контекст: Горовосхождений в России до 1786 года, принятого за официальную дату начала мирового альпинизма, практически не было. Первым из известных восхождений русских людей, был подъем Петра I на гору Броккен(1142м), в Южной Германии в 1697 году. 
    Вопрос: Какие горы находятся в Северной Америке<|end|>

    <|assistant|>
    [Анализ] Контекст не имеет отношения к Северной Америке.
    [Вывод] В конексте нет необходимой информации.
    [Ответ] У меня нет знаний о данном вопросе. <|end|>  

    <|user|>
    Контекст: Виталий Абалаков — главный создатель советского альпинистского снаряжения. Он разработал: Веревку Абалакова (первую отечественную альпинистскую веревку).
    Крючья Абалакова (скальные и ледовые). Рюкзак Абалакова (конструкция которого стала классической на десятилетия). Спусковое устройство («лепесток Абалакова»).
    Вопрос: Какие изобретения сделал Абалаков?<|end|>

    <|assistant|>
    [Анализ] В контексте указано, что Виталий Абалаков главный создатель советского альпинистского снаряжения, а также перечислен список.
    [Вывод] В контексте указаны конкретные виды снаряжения..
    [Ответ] Виталий Абалаков разработал разнообразное снаряжение для альпинизма. В том числе: Веревку Абалакова, Крючья Абалакова, Рюкзак Абалакова и Спусковое устройство («лепесток Абалакова») <|end|>

    <|user|>
    Контекст: Братья Абалаковы совершили первовосхождение на пик Ленина в 1934 году.
    Вопрос: Какие восхождения совершили Абалаковы?<|end|>

    <|assistant|>
    [Анализ] Контекст сообщает о первовосхождении братьев Абалаковых на пик Ленина в 1934 году.
    [Вывод] Конкретное восхождение указано в контексте.
    [Ответ] Братья Абалаковы совершили первовосхождение на пик Ленина в 1934 году.<|end|>    


    <|user|>
    Контекст: {context}
    Вопрос: {question}<|end|>
    <|assistant|>
    """
)

class Phi3RAGClient_CoT:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "soviet_mountaineering",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        phi3_model: str = "phi3:mini",
        temperature: float = 0.2,
        k: int = 3,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.phi3_model = phi3_model
        self.temperature = temperature
        self.k = k
        self.chain = None
        
    def _get_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    
    def _get_vectorstore(self):
        embeddings = self._get_embeddings()
        return Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=embeddings,
        )
    
    def _build_retriever(self):
        vectorstore = self._get_vectorstore()
        return vectorstore.as_retriever(search_kwargs={"k": self.k})
    
    def _get_phi3_ollama(self):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        return ChatOllama(
            model=self.phi3_model,
            base_url=base_url,
            temperature=self.temperature,
            num_ctx=4096,
        )
    
    def _build_prompt(self):
        return PromptTemplate(
            template=PROMPT_TEMPLATE_CoT,
            input_variables=["context", "question"],
        )
    
    def initialize(self):
        retriever = self._build_retriever()
        llm = self._get_phi3_ollama()
        prompt = self._build_prompt()

        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
    
    def ask(self, question: str) -> Dict[str, Any]:
        if self.chain is None:
            self.initialize()
            
        result = self.chain.invoke({"query": question})
        
        sources = []
        for i, doc in enumerate(result.get("source_documents", []) or []):
            meta = doc.metadata or {}
            sources.append(
                {
                    "rank": i + 1,
                    "source": meta.get("source", "unknown"),
                    "content_preview": (doc.page_content[:200] + "...") if len(doc.page_content) > 200 else doc.page_content,
                }
            )
        
        return {
            "answer": result.get("result"),
            "sources": sources,
        }

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phi-3 Local RAG runner")
    parser.add_argument("question", type=str, help="Текст запроса пользователя")
    
    args = parser.parse_args()

    try:
        client = Phi3RAGClient_CoT()
        result = client.ask(args.question)

        print(f"\nОтвет Phi-3:\n{result['answer'] or ''}")
        if result["sources"]:
            print("\nИсточники:")
            for s in result["sources"]:
                print(f"- {s['source']}: {s['content_preview']}")

    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()