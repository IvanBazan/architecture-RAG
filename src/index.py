import os
import glob
import chromadb
import uuid
import argparse
import json

from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

class ChromaDocumentProcessor:
    def __init__(self, persist_directory: str = "./chroma_db", chunk_size: int = 1000, chunk_overlap: int = 150):
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Инициализация текстового сплиттера
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            keep_separator=False,
            separators=[" ","\n","\n\n"]
        )

        # Инициализация модели эмбеддингов
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Клиент ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)


    def load_documents(self, folder_path: str) -> List[Document]:
        documents = []
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        
        print(f"Найдено txt файлов: {len(txt_files)}")
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                if not content.strip():
                    print(f"Предупреждение: файл {file_path} пуст")
                    continue
                    
                # Создаем документ с метаданными
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": os.path.basename(file_path),
                        "file_path": file_path,
                        "document_type": "txt",
                        "file_size": len(content)
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                print(f"Ошибка при чтении файла {file_path}: {e}")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        all_chunks = []
        
        for doc in documents:
            try:
                # Разбиваем документ на чанки
                doc_chunks = self.text_splitter.split_documents([doc])
                
                for i, chunk in enumerate(doc_chunks):
                    chunk.metadata.update({
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_index": i,
                        "total_chunks": len(doc_chunks),
                        "original_source": chunk.metadata.get("source", "unknown")
                    })
                
                all_chunks.extend(doc_chunks)
                
            except Exception as e:
                print(f"Ошибка при разбиении документа {doc.metadata.get('source', 'unknown')}: {e}")
        
        return all_chunks

    def verify_overlap(self, chunks: List[Document]):
        #Проверка overlap между чанками
        print("\n Проверка overlap: ")
        
        if len(chunks) <= 1:
            print("Недостаточно чанков для проверки overlap")
            return
        
        total_overlap = 0
        valid_pairs = 0
        
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i].page_content
            chunk2 = chunks[i + 1].page_content
            
            max_possible_overlap = min(150, len(chunk1), len(chunk2))
            actual_overlap = 0
            
            for overlap_size in range(1, max_possible_overlap + 1):
                if chunk1.endswith(chunk2[:overlap_size]):
                    actual_overlap = overlap_size
                    break
            
            print(f"Чанки {i}-{i+1}: overlap = {actual_overlap} символов")
            
            if actual_overlap > 0:
                total_overlap += actual_overlap
                valid_pairs += 1
        
        if valid_pairs > 0:
            avg_overlap = total_overlap / valid_pairs
            print(f"Средний overlap: {avg_overlap:.1f} символов")

    
    def create_vectorstore(self, chunks: List[Document], collection_name: str = "documents") -> Chroma:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
        return vectorstore

    def save_chunks_metadata(self, chunks: List[Document], output_path: str):
        chunks_info = []
        
        for chunk in chunks:
            chunks_info.append({
                "chunk_id": chunk.metadata.get("chunk_id"),
                "metadata": chunk.metadata,
                "content_length": len(chunk.page_content),
                "word_count": len(chunk.page_content.split())
            })
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_info, f, ensure_ascii=False, indent=2)

    def process_documents(self, input_folder: str, collection_name: str = "documents"):
        print("Начало обработки документов")
        
        # Загрузка документов
        print("1. Загрузка документов: ")
        documents = self.load_documents(input_folder)
        if not documents:
            raise ValueError("Нет документов")
        
        # Разбитие на чанки
        print("2. Разбитие на чанки...")
        chunks = self.split_documents(documents)
        print(f"   Создано чанков: {len(chunks)}")

        self.verify_overlap(chunks)
        
        # Создание векторного хранилища
        print("3. Создание векторного индекса в ChromaDB...")
        vectorstore = self.create_vectorstore(chunks, collection_name)
        
        # Сохранение информации о чанках
        print("4. Сохранение метаданных...")
        self.save_chunks_metadata(chunks, os.path.join(self.persist_directory, "chunks_metadata.json"))
        
        print("Обработка завершена")
        print(f"Векторное хранилище сохранено в: {self.persist_directory}")
        print(f"Коллекция: {collection_name}")
        print(f"Общее количество чанков: {len(chunks)}")
        
        return vectorstore, chunks    

class ChromaSearch:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def load_vectorstore(self, collection_name: str = "documents") -> Chroma:
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        return vectorstore
    
    def search_documents(self, query: str, collection_name: str = "documents", k: int = 5) -> List[Dict]:
        vectorstore = self.load_vectorstore(collection_name)

        results = vectorstore.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for i, (doc, score) in enumerate(results):
            formatted_results.append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "file_path": doc.metadata.get("file_path", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", "unknown"),
                "total_chunks": doc.metadata.get("total_chunks", "unknown"),
                "similarity_score": float(score),
                "chunk_id": doc.metadata.get("chunk_id", "unknown")
            })
        return formatted_results
    
    def get_collection_info(self, collection_name: str = "documents") -> Dict:
        client = chromadb.PersistentClient(path=self.persist_directory)
        collection = client.get_collection(collection_name)
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata
        }
    
    def print_results(self, results):
        sorted_results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
        for i, res in enumerate(sorted_results):
            print(f"\n")
            print(f"схожесть: {res['similarity_score']:.4f})")
            print(f"источник: {res['source']}")
            print(f"чанк: {res['chunk_index'] + 1} из {res['total_chunks']}")
            print(res['content'][:500] + "..." if len(res['content']) > 500 else res['content'])
            print(f"\n")

def main():
    
    parser = argparse.ArgumentParser(description="Построение векторного индекса")
    parser.add_argument("--input", "-i", default="knowledge_base/soviet_mountaineering", 
                       help="Папка с исходными документами")
    parser.add_argument("--output", "-o", default="./chroma_db", 
                       help="Папка для сохранения индекса")
    parser.add_argument("--collection", "-c", default="soviet_mountaineering", 
                       help="Название коллекции в ChromaDB")
    parser.add_argument("--chunk-size", type=int, default=1000, 
                       help="Размер чанка в символах")
    parser.add_argument("--chunk-overlap", type=int, default=150, 
                       help="Перекрытие между чанками")
    
    args = parser.parse_args()
    
    INPUT_FOLDER = args.input
    PERSIST_DIRECTORY = args.output
    COLLECTION_NAME = args.collection
    
    print("Построение индекса:\n\n")
    
    processor = ChromaDocumentProcessor(
        persist_directory=PERSIST_DIRECTORY,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    try:
        vectorstore, chunks = processor.process_documents(
            INPUT_FOLDER, 
            COLLECTION_NAME
        )
        
        print(f"\n Создано чанков: {len(chunks)}")
        
    except Exception as e:
        print(f"\n Ошибка при обработке: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()