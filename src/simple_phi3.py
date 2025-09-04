import os
from typing import Optional
from langchain_ollama import ChatOllama

class Phi3Client:    
    def __init__(
        self,
        model: str = "phi3:mini",
        temperature: float = 0.2,
        base_url: Optional[str] = None
    ):
        self.model = model
        self.temperature = temperature
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm = None
        
    def _initialize(self):
        self.llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            num_ctx=4096,
        )
    
    def ask(self, question: str) -> str:
        if self.llm is None:
            self._initialize()
            
        prompt = f"""Ты русскоязычный ассистент. Отвечай кратко и точно на русском языке.
        Вопрос: {question}
        Ответ:"""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("'Ваш вопрос'")
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    
    try:
        client = Phi3Client()
        response = client.ask(question)
        print(f"\nОтвет Phi-3:\n{response}")
        
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()